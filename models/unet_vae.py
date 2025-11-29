import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from models.pretrained_encoders import (
    PretrainedResNetEncoder, 
    PretrainedVAEEncoder,
    PretrainedStyleGANEncoder
)


class UNetVAE(nn.Module):
    """U-Net based VAE with optional pretrained encoder."""
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 512,
        hidden_dims: List[int] = None,
        image_size: int = 128,
        use_attention: bool = True,
        use_skip_connections: bool = True,
        pretrained_encoder: Optional[str] = None,
        encoder_checkpoint: Optional[str] = None,
        freeze_encoder_stages: int = 0
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.use_skip_connections = use_skip_connections
        self.image_size = image_size
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512, 512]
        
        self.hidden_dims = hidden_dims
        
        # Choose encoder based on pretrained option
        if pretrained_encoder == 'resnet':
            self.encoder = PretrainedResNetEncoder(
                model_name='resnet50',
                pretrained='imagenet',
                frozen_stages=freeze_encoder_stages,
                output_channels=hidden_dims
            )
        elif pretrained_encoder == 'vggface':
            self.encoder = PretrainedResNetEncoder(
                model_name='resnet50',
                pretrained='vggface2',
                frozen_stages=freeze_encoder_stages,
                output_channels=hidden_dims
            )
        elif pretrained_encoder == 'vae' and encoder_checkpoint:
            self.encoder = PretrainedVAEEncoder(
                checkpoint_path=encoder_checkpoint,
                frozen=(freeze_encoder_stages > 0)
            )
        elif pretrained_encoder == 'stylegan':
            self.encoder = PretrainedStyleGANEncoder(
                model_path=encoder_checkpoint,
                frozen_layers=freeze_encoder_stages
            )
        else:
            self.encoder = UNetEncoder(
                input_channels=input_channels,
                hidden_dims=hidden_dims,
                use_attention=use_attention
            )
        
        # Based on the error message, we know the actual flattened size is 32768
        # This corresponds to 512 channels * 8 * 8 spatial dimensions
        self.flattened_size = 32768
        self.encoder_output_channels = hidden_dims[-1]  # 512
        self.encoder_output_size = 8  # 8x8 spatial dimensions
        
        # Latent space - use the correct size
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)
        
        # Decoder
        self.decoder = UNetDecoder(
            output_channels=input_channels,
            hidden_dims=list(reversed(hidden_dims)),
            use_attention=use_attention,
            target_size=self.image_size  # Pass the target size
)
    
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Encode input to latent distribution parameters."""
        # Concatenate mask if provided
        if mask is not None:
            x = torch.cat([x, mask], dim=1)
            
        features, skip_connections = self.encoder(x)
        features = features.flatten(start_dim=1)
        
        mu = self.fc_mu(features)
        log_var = self.fc_var(features)
        
        return mu, log_var, skip_connections
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, skip_connections: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Decode latent vector to image."""
        features = self.decoder_input(z)
        
        # Reshape to (batch, 512, 8, 8) based on actual encoder output
        features = features.view(
            -1, 
            self.encoder_output_channels,  # 512
            self.encoder_output_size,      # 8
            self.encoder_output_size        # 8
        )
        
        if self.use_skip_connections and skip_connections is not None:
            output = self.decoder(features, skip_connections)
        else:
            output = self.decoder(features, None)

        # IMPORTANT: Ensure output matches the configured image size
        # The decoder might only output 128x128, so we need to upsample to 256x256
        if output.shape[2] != self.image_size or output.shape[3] != self.image_size:
            output = F.interpolate(output, size=(self.image_size, self.image_size), 
                                mode='bilinear', align_corners=False)
            
        return torch.tanh(output)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE."""
        mu, log_var, skip_connections = self.encode(x, mask)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z, skip_connections)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }


class UNetEncoder(nn.Module):
    """U-Net style encoder with skip connections."""
    
    def __init__(
        self,
        input_channels: int,
        hidden_dims: List[int],
        use_attention: bool = True
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        in_channels = input_channels + 1  # +1 for mask channel
        
        for i, out_channels in enumerate(hidden_dims):
            self.blocks.append(
                EncoderBlock(in_channels, out_channels)
            )
            
            if use_attention and i >= len(hidden_dims) - 2:
                self.attention_blocks.append(
                    SelfAttention(out_channels)
                )
            else:
                self.attention_blocks.append(nn.Identity())
                
            in_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_connections = []
        
        for block, attention in zip(self.blocks, self.attention_blocks):
            x = block(x)
            x = attention(x)
            skip_connections.append(x)
            
        return x, skip_connections[:-1]  # Don't include last feature as skip


class UNetDecoder(nn.Module):
    """U-Net style decoder with skip connections."""
    
    def __init__(
        self,
        output_channels: int,
        hidden_dims: List[int],
        use_attention: bool = True,
        target_size: int = 128  # Add target size parameter
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.target_size = target_size
        
        for i, (in_channels, out_channels) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            # Account for skip connections (doubles channels)
            actual_in_channels = in_channels * 2 if i > 0 else in_channels
            
            self.blocks.append(
                DecoderBlock(actual_in_channels, out_channels)
            )
            
            if use_attention and i <= 1:
                self.attention_blocks.append(
                    SelfAttention(out_channels)
                )
            else:
                self.attention_blocks.append(nn.Identity())
        
        # Add an extra upsampling block to get back to original size
        self.extra_upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims[-1], output_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, skip_connections: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
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
                
            x = block(x)
            x = attention(x)
        
        # Extra upsampling to reach target size
        x = self.extra_upsample(x)
        
        # Ensure we reach the target size
        if x.shape[2] != self.target_size or x.shape[3] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), 
                            mode='bilinear', align_corners=False)
        
        return self.final_conv(x)


class EncoderBlock(nn.Module):
    """Encoder block with downsampling."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Decoder block with upsampling."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SelfAttention(nn.Module):
    """Self-attention module for capturing long-range dependencies."""
    
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