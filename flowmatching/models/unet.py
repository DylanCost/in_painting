"""U-Net architecture for flow matching image inpainting.

This module implements the main U-Net model that takes an image, mask, and time
as input and predicts a velocity field for flow matching.
"""

import torch
import torch.nn as nn
from typing import Optional

from .embeddings import TimestepEmbedding
from .blocks import DoubleConv, Down, Up


class UNet(nn.Module):
    """U-Net model for flow matching inpainting.
    
    A time-conditioned U-Net that takes a masked image (RGB + mask channel)
    and predicts a velocity field for flow matching. The architecture uses
    skip connections between encoder and decoder.
    
    Architecture:
        - Input: 4 channels (RGB + binary mask)
        - Output: 3 channels (RGB velocity field)
        - Encoder: 64 -> 128 -> 256 -> 512 channels
        - Bottleneck: 512 channels with convolutions
        - Decoder: 512 -> 256 -> 128 -> 64 channels
        - Time conditioning via sinusoidal embeddings
    
    Args:
        in_channels: Number of input channels (default: 4 for RGB + mask)
        out_channels: Number of output channels (default: 3 for RGB velocity)
        base_channels: Base number of channels (default: 64)
        channel_multipliers: Channel multipliers for each level (default: [1, 2, 4, 8])
        num_res_blocks: Number of residual blocks per level (default: 2)
        time_embed_dim: Dimension of time embedding (default: 256)
        num_groups: Number of groups for GroupNorm (default: 8)
    
    Input:
        x: Input tensor of shape [B, in_channels, H, W]
        t: Time values of shape [B] or [B, 1], values in [0, 1]
    
    Output:
        Velocity field of shape [B, out_channels, H, W]
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: list[int] = None,
        num_res_blocks: int = 2,
        time_embed_dim: int = 256,
        num_groups: int = 8
    ):
        super().__init__()
        
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_embedding = TimestepEmbedding(
            embedding_dim=time_embed_dim,
            hidden_dim=time_embed_dim * 4,
            output_dim=time_embed_dim
        )
        
        # Calculate channel dimensions for each level
        channels = [base_channels * mult for mult in channel_multipliers]
        
        # Initial convolution
        self.init_conv = DoubleConv(
            in_channels,
            channels[0],
            time_embed_dim=time_embed_dim,
            num_groups=num_groups
        )
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder_blocks.append(
                Down(
                    channels[i],
                    channels[i + 1],
                    time_embed_dim=time_embed_dim,
                    num_groups=num_groups
                )
            )
        
        # Bottleneck with convolutions
        self.bottleneck = nn.ModuleList([
            DoubleConv(
                channels[-1],
                channels[-1],
                time_embed_dim=time_embed_dim,
                num_groups=num_groups
            ),
            DoubleConv(
                channels[-1],
                channels[-1],
                time_embed_dim=time_embed_dim,
                num_groups=num_groups
            ),
            DoubleConv(
                channels[-1],
                channels[-1],
                time_embed_dim=time_embed_dim,
                num_groups=num_groups
            )
        ])
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoder_blocks.append(
                Up(
                    channels[i],
                    channels[i - 1],
                    channels[i - 1],
                    time_embed_dim=time_embed_dim,
                    num_groups=num_groups,
                    bilinear=True
                )
            )
        
        # Output convolution
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
    
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
        time_emb = self.time_embedding(t)  # [B, time_embed_dim]
        
        # Initial convolution
        x = self.init_conv(x, time_emb)  # [B, base_channels, H, W]
        
        # Encoder path with skip connections
        skip_connections = [x]
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, time_emb)
            skip_connections.append(x)
        
        # Remove the last skip connection (it's the input to bottleneck)
        skip_connections = skip_connections[:-1]
        
        # Bottleneck
        for block in self.bottleneck:
            x = block(x, time_emb)
        
        # Decoder path with skip connections
        for decoder_block in self.decoder_blocks:
            skip = skip_connections.pop()
            x = decoder_block(x, skip, time_emb)
        
        # Output convolution
        x = self.out_conv(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters.
        
        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get the model size in megabytes.
        
        Returns:
            Model size in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


def create_unet(
    image_size: int = 128,
    in_channels: int = 4,
    out_channels: int = 3,
    base_channels: int = 64,
    time_embed_dim: int = 256
) -> UNet:
    """Factory function to create a U-Net model with default settings.
    
    Args:
        image_size: Input image size (default: 128)
        in_channels: Number of input channels (default: 4)
        out_channels: Number of output channels (default: 3)
        base_channels: Base number of channels (default: 64)
        time_embed_dim: Time embedding dimension (default: 256)
    
    Returns:
        Configured U-Net model
    """
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_multipliers=[1, 2, 4, 8],
        num_res_blocks=2,
        time_embed_dim=time_embed_dim,
        num_groups=8
    )
    
    return model