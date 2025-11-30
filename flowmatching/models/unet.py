"""U-Net architecture for flow matching image inpainting.

This module implements the main U-Net model that takes an image, mask, and time
as input and predicts a velocity field for flow matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .embeddings import TimestepEmbedding
from .blocks import EncoderBlock, DecoderBlock, SelfAttention


class UNetEncoder(nn.Module):
    """U-Net style encoder with skip connections."""
    
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
        
        in_channels = input_channels
        
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
        skip_connections = []
        
        for block, attention in zip(self.blocks, self.attention_blocks):
            x = block(x, t_emb)  # Pass time embedding
            x = attention(x)
            skip_connections.append(x)
            
        return x, skip_connections[:-1]


class UNetDecoder(nn.Module):
    """U-Net style decoder with skip connections."""
    
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


class UNet(nn.Module):
    """U-Net model for flow matching inpainting.
    
    A time-conditioned U-Net that takes a masked image (RGB + mask channel)
    and predicts a velocity field for flow matching.
    
    Args:
        in_channels: Number of input channels (default: 4 for RGB + mask)
        out_channels: Number of output channels (default: 3 for RGB velocity)
        hidden_dims: List of channel dimensions for each level
        time_embed_dim: Dimension of time embedding (default: 256)
        use_attention: Whether to use self-attention (default: True)
        attention_resolutions: List of resolutions to apply attention at
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        hidden_dims: List[int] = None,
        time_embed_dim: int = 256,
        use_attention: bool = True,
        attention_resolutions: List[int] = None,
        image_size: int = 128
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
            
        if attention_resolutions is None:
            attention_resolutions = [16]
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.time_embed_dim = time_embed_dim
        
        # Calculate which encoder blocks should have attention
        attention_blocks = []
        for i in range(len(hidden_dims)):
            resolution = image_size // (2 ** (i + 1))
            if resolution in attention_resolutions:
                attention_blocks.append(i)
        
        # Time embedding
        # Note: TimestepEmbedding internally uses SinusoidalPositionEmbeddings(scale=1000.0)
        # and TimeEmbeddingMLP to match DDPM structure
        self.time_embedding = TimestepEmbedding(
            embedding_dim=time_embed_dim,
            hidden_dim=time_embed_dim * 4,
            output_dim=time_embed_dim * 4  # Match diffusion model's time_emb_dim
        )
        
        # Encoder
        self.encoder = UNetEncoder(
            input_channels=in_channels,
            hidden_dims=hidden_dims,
            time_emb_dim=time_embed_dim * 4,
            use_attention=use_attention,
            attention_blocks=attention_blocks
        )
        
        # Decoder
        self.decoder = UNetDecoder(
            output_channels=out_channels,
            hidden_dims=list(reversed(hidden_dims)),
            time_emb_dim=time_embed_dim * 4,
            use_attention=use_attention,
            attention_blocks=attention_blocks
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.
        
        Args:
            x: Input tensor of shape [B, in_channels, H, W]
               Expected to be concatenation of image and mask: [image, mask]
            t: Time values of shape [B] or [B, 1], values in [0, 1]
            
        Returns:
            Velocity field of shape [B, out_channels, H, W]
        """
        # Generate time embeddings
        time_emb = self.time_embedding(t)
        
        # Encoder path
        features, skip_connections = self.encoder(x, time_emb)
        
        # Decoder path
        output = self.decoder(features, time_emb, skip_connections)
        
        return output
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get the model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


def create_unet(
    hidden_dims: List[int],
    image_size: int = 128,
    in_channels: int = 4,
    out_channels: int = 3,
    time_embed_dim: int = 256,
    attention_resolutions: Optional[List[int]] = None
) -> UNet:
    """Factory function to create a U-Net model.
    
    Args:
        hidden_dims: List of channel dimensions
        image_size: Input image size (default: 128)
        in_channels: Number of input channels (default: 4)
        out_channels: Number of output channels (default: 3)
        time_embed_dim: Time embedding dimension (default: 256)
        attention_resolutions: List of resolutions for attention (optional)
    
    Returns:
        Configured U-Net model
    """
    if attention_resolutions is None:
        attention_resolutions = [16]
    
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_dims=hidden_dims,
        time_embed_dim=time_embed_dim,
        use_attention=True,
        attention_resolutions=attention_resolutions,
        image_size=image_size
    )
    
    return model