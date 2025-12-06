"""
U-Net Diffusion Model for Image Inpainting

This module implements a complete U-Net architecture for diffusion-based image inpainting,
including the noise scheduler for the forward diffusion process. The model predicts noise
in masked regions of images during training, enabling high-quality inpainting through
iterative denoising during inference. The implementation follows the Denoising Diffusion
Probabilistic Model (DDPM) framework adapted for conditional inpainting tasks.

The file contains two main components: (1) UNetDiffusion - a U-Net model with encoder-decoder
architecture, skip connections, optional self-attention at specified resolutions, and time
conditioning via sinusoidal embeddings injected at each level; (2) NoiseScheduler - manages
the forward diffusion process with support for linear and cosine beta schedules, computes
derived quantities needed for training and sampling, and handles mask-aware noise addition
that only applies noise to regions marked for inpainting.

Supporting classes include EncoderBlock and DecoderBlock for downsampling/upsampling with
time conditioning, SelfAttention for capturing long-range spatial dependencies, and
SinusoidalPositionEmbeddings for encoding timestep information. The architecture processes
images concatenated with binary masks as input, ensuring predictions are confined to the
masked regions that need inpainting while preserving known pixels.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDiffusion(nn.Module):
    """
    U-Net architecture for diffusion-based image inpainting.
    
    This model predicts noise in masked regions of images during the diffusion process.
    It uses an encoder-decoder architecture with skip connections and optional self-attention
    mechanisms at specified resolutions. The model incorporates time embeddings to condition
    on the diffusion timestep and processes masked regions by concatenating the mask as an
    additional input channel.
    
    Architecture:
        - Encoder: Downsamples the input through multiple blocks, extracting hierarchical features
        - Decoder: Upsamples features back to original resolution using skip connections
        - Time embedding: Sinusoidal embeddings processed through MLP to condition on timestep
        - Attention: Optional self-attention at specified spatial resolutions for long-range dependencies
    
    The model only predicts noise within masked regions, zeroing out predictions elsewhere.
    """
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dims: List[int] = None,
        use_attention: bool = True,
        use_skip_connections: bool = True,
        input_size: int = 256,
        attention_resolutions: List[int] = None
    ):
        super().__init__()
        
        self.use_skip_connections = use_skip_connections
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512, 512]
        
        # Set default attention resolutions if not provided
        if attention_resolutions is None:
            attention_resolutions = [16]  # Default: only at 16x16 will there be attention
        
        # Calculate which encoder blocks should have attention
        attention_blocks = []
        for i in range(len(hidden_dims)):
            resolution = input_size // (2 ** (i + 1))
            if resolution in attention_resolutions:
                attention_blocks.append(i)

        # ========== TIME EMBEDDING (MINIMAL VERSION) ==========
        time_emb_base = 256
        time_emb_dim = time_emb_base * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_base),  # Input: [B] → Output: [B, 256]
            nn.Linear(time_emb_base, time_emb_dim),        # [B, 256] → [B, 1024]
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),         # [B, 1024] → [B, 1024]
            nn.GELU(),
        )
        
        self.encoder = UNetEncoder(
            input_channels=input_channels,
            hidden_dims=hidden_dims,
            time_emb_dim=time_emb_dim,
            use_attention=use_attention,
            attention_blocks=attention_blocks
        )
        
        # Decoder
        self.decoder = UNetDecoder(
            output_channels=input_channels,
            hidden_dims=list(reversed(hidden_dims)),
            time_emb_dim=time_emb_dim,
            use_attention=use_attention,
            attention_blocks=attention_blocks
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Compute time embedding ONCE
        t_emb = self.time_mlp(t)  # [B, 256]
        
        # apply mask
        # masked_input = x * (1 - mask)
        # Concatenate image with mask
        x_input = torch.cat([x, mask], dim=1)
        
        # Pass time embedding to encoder and decoder
        features, skip_connections = self.encoder(x_input, t_emb)
        predicted_noise = self.decoder(features, t_emb, skip_connections)

        # Mask the output - zero out predictions outside masked region
        mask_3c = mask.repeat(1, predicted_noise.size(1), 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
        predicted_noise = predicted_noise * mask_3c

        return predicted_noise

class NoiseScheduler:
    """
    Noise scheduler for the diffusion process in image inpainting.
    
    Manages the forward diffusion process by defining a noise schedule (beta values)
    and computing derived quantities needed for both training and sampling. The scheduler
    controls how noise is gradually added to images during the forward process and supports
    both linear and cosine scheduling strategies.
    
    Key quantities:
        - betas (β_t): Variance schedule controlling noise addition at each timestep
        - alphas (α_t): 1 - β_t, the signal retention factor
        - alpha_bars (ᾱ_t): Cumulative product of alphas, used for direct t-step noising
        - alpha_bars_prev (ᾱ_{t-1}): Previous timestep's alpha_bar, used in reverse process
    
    The forward diffusion process follows:
        q(x_t | x_0) = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε, where ε ~ N(0, I)
    
    For inpainting, noise is only applied to masked regions while preserving known pixels.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = 'linear'
    ):
        """
        Args:
            num_timesteps: Total number of diffusion steps (T)
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule_type: 'linear' or 'cosine'
        """
        self.num_timesteps = num_timesteps
        
        # Create beta schedule
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Compute alpha values
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # Cumulative product
        
        # For sampling (reverse process)
        alpha_bars_prev_np = np.append(1.0, self.alpha_bars[:-1].cpu().numpy())
        self.alpha_bars_prev = torch.from_numpy(alpha_bars_prev_np).float()
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def to(self, device):
        """Move all tensors to the specified device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.alpha_bars_prev = self.alpha_bars_prev.to(device)
        return self
    
    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, mask: torch.Tensor):
        """
        Add noise to the masked region of the image at timestep t.
        
        Args:
            x_0: Original clean image [B, C, H, W]
            t: Timestep for each sample in batch [B]
            mask: Binary mask [B, 1, H, W], 1 = region to inpaint
        
        Returns:
            x_t: Noisy masked image [B, C, H, W]
            noise: The noise that was added [B, C, H, W] (0 outside mask)
        """
        # Generate random noise
        noise = torch.randn_like(x_0)
        
        # Get alpha_bar for the given timesteps
        alpha_bar_t = self.alpha_bars[t]  # [B]
        
        # Reshape for broadcasting: [B] -> [B, 1, 1, 1]
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
        
        # Forward diffusion: q(x_t | x_0)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        # Apply noise only to masked region
        noisy_region = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        x_t = mask * noisy_region + (1 - mask) * x_0
        
        # Zero out noise outside masked region
        noise_masked = noise * mask  # Only keep noise where mask = 1
        
        return x_t, noise_masked


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for diffusion timesteps.
    
    Encodes timestep information using sinusoidal functions at different frequencies,
    allowing the model to learn temporal patterns in the diffusion process. This approach
    is adapted from the positional encoding in "Attention is All You Need" (Vaswani et al., 2017).
    
    The embedding for timestep t is computed as:
        PE(t, 2i) = sin(t / 10000^(2i/dim))
        PE(t, 2i+1) = cos(t / 10000^(2i/dim))
    
    where i ranges from 0 to dim/2-1. This creates a spectrum of frequencies that helps
    the model distinguish between different timesteps and interpolate between them.
    
    Key properties:
        - Each dimension uses a different frequency
        - Produces continuous, smooth embeddings
        - Allows the model to learn relative temporal positions
        - No learnable parameters (deterministic transformation)
    
    Reference:
        "Attention is All You Need" - https://arxiv.org/abs/1706.03762
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Timesteps, shape [B] - integers from 0 to num_timesteps
        
        Returns:
            embeddings: Sinusoidal embeddings, shape [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        
        # Create frequency spectrum
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Apply to timesteps
        embeddings = time[:, None] * embeddings[None, :]
        
        # Concatenate sin and cos
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings


class UNetEncoder(nn.Module):
    """
    U-Net style encoder for progressive downsampling with optional self-attention.
    
    The encoder processes the input image (concatenated with mask) through multiple
    downsampling blocks, extracting hierarchical feature representations at different
    spatial scales. Each block halves the spatial resolution while increasing the
    number of channels. Time embeddings are injected at each level to condition the
    features on the diffusion timestep.
    
    Architecture flow:
        Input [B, C+1, H, W] 
        → Block 1 [B, D1, H/2, W/2] → (optional attention) → skip connection
        → Block 2 [B, D2, H/4, W/4] → (optional attention) → skip connection
        → ...
        → Block N [B, DN, H/(2^N), W/(2^N)] → (optional attention)
    
    where D1, D2, ..., DN are specified by hidden_dims.
    
    Skip connections from all but the final block are returned for use in the decoder,
    enabling the network to recover fine-grained spatial details during upsampling.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_dims: List[int],
        time_emb_dim: int,
        use_attention: bool = True,
        attention_blocks: List[int] = None
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        if attention_blocks is None:
            attention_blocks = []
        
        in_channels = input_channels + 1  # +1 for mask channel
        
        for i, out_channels in enumerate(hidden_dims):
            self.blocks.append(
                EncoderBlock(in_channels, out_channels, time_emb_dim)
            )
            
            # Only apply attention at specified block indices
            if use_attention and i in attention_blocks:
                self.attention_blocks.append(
                    SelfAttention(out_channels)
                )
            else:
                self.attention_blocks.append(nn.Identity())
                
            in_channels = out_channels
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        #(f"[DEBUG] Encoder received: {x.shape}")
        skip_connections = []
        
        for block, attention in zip(self.blocks, self.attention_blocks):
            x = block(x, t_emb)  # Pass time embedding
            x = attention(x)
            skip_connections.append(x)
            
        return x, skip_connections[:-1]


class UNetDecoder(nn.Module):
    """
    U-Net style decoder for progressive upsampling with skip connections.
    
    The decoder reconstructs the output from encoded features through multiple upsampling
    blocks, progressively increasing spatial resolution while decreasing channel dimensions.
    Skip connections from the encoder are concatenated at corresponding levels to recover
    fine-grained spatial details lost during downsampling. Time embeddings are injected at
    each level to maintain temporal conditioning throughout the reconstruction.
    
    Architecture flow:
        Bottleneck [B, D0, H/(2^N), W/(2^N)]
        → Block 1 [B, D1, H/(2^(N-1)), W/(2^(N-1))] → (optional attention)
        → Concatenate skip → Block 2 [B, D2, H/(2^(N-2)), W/(2^(N-2))] → (optional attention)
        → ...
        → Concatenate skip → Block N [B, DN, H, W] → (optional attention)
        → Final Conv [B, output_channels, H, W]
    
    Skip connections are concatenated channel-wise, doubling the input channels for blocks
    that receive them. The attention blocks mirror the encoder's attention pattern to maintain
    architectural symmetry.
    """
    
    def __init__(
        self,
        output_channels: int,
        hidden_dims: List[int],
        time_emb_dim: int,
        use_attention: bool = True,
        attention_blocks: List[int] = None
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        # Default to no attention if not specified
        if attention_blocks is None:
            attention_blocks = []
        
        # Mirror encoder attention
        decoder_attention_indices = [len(hidden_dims) - 1 - i for i in attention_blocks]
        
        # Create decoder blocks to match encoder blocks
        for i in range(len(hidden_dims)):
            if i == 0:
                in_channels = hidden_dims[0]
                out_channels = hidden_dims[0] if len(hidden_dims) == 1 else hidden_dims[1]
            elif i == len(hidden_dims) - 1:
                in_channels = hidden_dims[i] * 2  # With skip connection
                out_channels = hidden_dims[i]
            else:
                in_channels = hidden_dims[i] * 2
                out_channels = hidden_dims[i + 1]
            
            self.blocks.append(
                DecoderBlock(in_channels, out_channels, time_emb_dim)
            )
            
            # Apply attention at mirrored positions
            if use_attention and i in decoder_attention_indices:
                self.attention_blocks.append(SelfAttention(out_channels))
            else:
                self.attention_blocks.append(nn.Identity())
        
        # Final 1x1 conv to get output channels
        self.final_conv = nn.Conv2d(hidden_dims[-1], output_channels, 1)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, skip_connections: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
            skip_connections: List of skip connection tensors
        """
        if skip_connections is not None:
            skip_connections = list(reversed(skip_connections))
        
        for i, (block, attention) in enumerate(zip(self.blocks, self.attention_blocks)):
            if skip_connections is not None and i > 0 and i <= len(skip_connections):
                # Concatenate skip connection
                skip = skip_connections[i - 1]
                # Resize if necessary
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
                
            x = block(x, t_emb)
            x = attention(x)
        
        return self.final_conv(x)


class EncoderBlock(nn.Module):
    """
    Single encoder block for U-Net with downsampling and time conditioning.
    
    Performs spatial downsampling by factor of 2 while increasing channel dimensions,
    with time embedding injection between two convolutional layers. This block is the
    fundamental building unit of the U-Net encoder.
    
    Architecture:
        Input [B, in_channels, H, W]
        → Strided Conv (4x4, stride=2) + BatchNorm + GELU → [B, out_channels, H/2, W/2]
        → Add time embedding (broadcast spatially) → [B, out_channels, H/2, W/2]
        → Conv (3x3) + BatchNorm + GELU → [B, out_channels, H/2, W/2]
    
    The time embedding is projected to match the output channel dimension and added
    element-wise to inject temporal information into the feature maps.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        """
        Initialize the encoder block.
        
        Args:
            in_channels: Number of input channels from previous layer or input image.
            out_channels: Number of output channels. This becomes the channel dimension
                         after downsampling.
            time_emb_dim: Dimensionality of the time embedding vector to be injected.
                         Typically 1024 in standard configurations. Will be linearly
                         projected to match out_channels.
        
        Components:
            - conv1: Downsampling convolution (kernel=4, stride=2) that halves spatial dims
            - time_proj: Linear projection mapping time embeddings to feature channels
            - conv2: Refinement convolution (kernel=3, stride=1) that preserves spatial dims
        """
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # Time projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # Add time embedding
        x = x + self.time_proj(t_emb)[:, :, None, None]
        x = self.conv2(x)
        return x


class DecoderBlock(nn.Module):
    """
    Single decoder block for U-Net with upsampling and time conditioning.
    
    Performs spatial upsampling by factor of 2 while typically decreasing channel dimensions,
    with time embedding injection between two convolutional layers. This block is the
    fundamental building unit of the U-Net decoder, mirroring the encoder block's structure
    but with upsampling instead of downsampling.
    
    Architecture:
        Input [B, in_channels, H, W]
        → Transposed Conv (4x4, stride=2) + BatchNorm + GELU → [B, out_channels, 2H, 2W]
        → Add time embedding (broadcast spatially) → [B, out_channels, 2H, 2W]
        → Conv (3x3) + BatchNorm + GELU → [B, out_channels, 2H, 2W]
    
    The time embedding is projected to match the output channel dimension and added
    element-wise to inject temporal information into the feature maps, maintaining
    consistency with the encoder's time conditioning.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # Time projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # Add time embedding
        x = x + self.time_proj(t_emb)[:, :, None, None]
        x = self.conv2(x)
        return x


class SelfAttention(nn.Module):
    """
    Self-attention module for capturing long-range spatial dependencies in feature maps.
    
    Implements self-attention mechanism adapted for 2D spatial feature maps, allowing each
    position to attend to all other positions regardless of distance. This enables the model
    to capture global context and long-range dependencies that convolutions alone cannot
    efficiently model due to their limited receptive fields.
    
    The module uses a scaled-down attention mechanism where query and key dimensions are
    reduced by a factor of 8 for computational efficiency, while value projections maintain
    full channel dimensionality. A learnable scalar γ (gamma) controls the contribution of
    attention features via a residual connection, starting from 0 to allow the network to
    initially rely on convolutional features.
    
    Architecture:
        Input [B, C, H, W]
        → Query: 1×1 Conv → [B, C/8, H×W] (reshaped)
        → Key: 1×1 Conv → [B, C/8, H×W] (reshaped)
        → Value: 1×1 Conv → [B, C, H×W] (reshaped)
        → Attention weights: softmax(Q @ K^T) → [B, H×W, H×W]
        → Attended features: V @ Attention^T → [B, C, H×W] → [B, C, H, W]
        → Output: Input + γ × Attended features
    
    Note:
        Attention is computationally expensive (O(H²W²) complexity), so this module is
        typically only applied at lower spatial resolutions (e.g., 16×16 or 32×32).
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        
        # Generate query, key, value
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)
        
        # Attention
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection with learnable weight
        return x + self.gamma * out