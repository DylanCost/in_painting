"""Building blocks for U-Net architecture.

This module contains reusable components for the U-Net model including
convolutional blocks, downsampling, and upsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoubleConv(nn.Module):
    """Double convolution block with GroupNorm and SiLU activation.
    
    Applies two consecutive convolution operations with normalization
    and activation. Includes optional time embedding conditioning.
    
    Architecture:
        Conv2d -> GroupNorm -> SiLU -> Conv2d -> GroupNorm -> SiLU
        + Time embedding injection (if provided)
        + Residual connection (if in_channels == out_channels)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_embed_dim: Dimension of time embedding (optional)
        num_groups: Number of groups for GroupNorm (default: 8)
        residual: Whether to use residual connection (default: True)
    
    Input:
        x: Input tensor of shape [B, in_channels, H, W]
        time_emb: Optional time embedding of shape [B, time_embed_dim]
    
    Output:
        Output tensor of shape [B, out_channels, H, W]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: Optional[int] = None,
        num_groups: int = 8,
        residual: bool = True
    ):
        super().__init__()
        
        self.residual = residual and (in_channels == out_channels)
        
        # First convolution block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act1 = nn.SiLU()
        
        # Second convolution block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()
        
        # Time embedding projection
        if time_embed_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, out_channels)
            )
        else:
            self.time_mlp = None
    
    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through double convolution block.
        
        Args:
            x: Input tensor of shape [B, in_channels, H, W]
            time_emb: Optional time embedding of shape [B, time_embed_dim]
            
        Returns:
            Output tensor of shape [B, out_channels, H, W]
        """
        identity = x if self.residual else None
        
        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        # Add time embedding if provided
        if time_emb is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(time_emb)
            # Reshape to [B, C, 1, 1] for broadcasting
            h = h + time_emb[:, :, None, None]
        
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Add residual connection
        if self.residual:
            h = h + identity
        
        h = self.act2(h)
        
        return h


class Down(nn.Module):
    """Downsampling block with double convolution.
    
    Performs spatial downsampling followed by double convolution.
    Uses MaxPool2d for downsampling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_embed_dim: Dimension of time embedding (optional)
        num_groups: Number of groups for GroupNorm (default: 8)
    
    Input:
        x: Input tensor of shape [B, in_channels, H, W]
        time_emb: Optional time embedding of shape [B, time_embed_dim]
    
    Output:
        Output tensor of shape [B, out_channels, H/2, W/2]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: Optional[int] = None,
        num_groups: int = 8
    ):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(
            in_channels,
            out_channels,
            time_embed_dim=time_embed_dim,
            num_groups=num_groups,
            residual=False
        )
    
    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through downsampling block.
        
        Args:
            x: Input tensor of shape [B, in_channels, H, W]
            time_emb: Optional time embedding of shape [B, time_embed_dim]
            
        Returns:
            Output tensor of shape [B, out_channels, H/2, W/2]
        """
        x = self.maxpool(x)
        x = self.conv(x, time_emb)
        return x


class Up(nn.Module):
    """Upsampling block with skip connection and double convolution.
    
    Performs spatial upsampling, concatenates with skip connection from encoder,
    and applies double convolution.
    
    Args:
        in_channels: Number of input channels (from previous layer)
        skip_channels: Number of channels in skip connection
        out_channels: Number of output channels
        time_embed_dim: Dimension of time embedding (optional)
        num_groups: Number of groups for GroupNorm (default: 8)
        bilinear: Use bilinear upsampling instead of transposed conv (default: True)
    
    Input:
        x: Input tensor of shape [B, in_channels, H, W]
        skip: Skip connection tensor of shape [B, skip_channels, H*2, W*2]
        time_emb: Optional time embedding of shape [B, time_embed_dim]
    
    Output:
        Output tensor of shape [B, out_channels, H*2, W*2]
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_embed_dim: Optional[int] = None,
        num_groups: int = 8,
        bilinear: bool = True
    ):
        super().__init__()
        
        # Upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_up = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=2,
                stride=2
            )
            self.conv_up = nn.Identity()
        
        # Double convolution after concatenation
        self.conv = DoubleConv(
            in_channels + skip_channels,
            out_channels,
            time_embed_dim=time_embed_dim,
            num_groups=num_groups,
            residual=False
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through upsampling block.
        
        Args:
            x: Input tensor of shape [B, in_channels, H, W]
            skip: Skip connection of shape [B, skip_channels, H*2, W*2]
            time_emb: Optional time embedding of shape [B, time_embed_dim]
            
        Returns:
            Output tensor of shape [B, out_channels, H*2, W*2]
        """
        x = self.up(x)
        x = self.conv_up(x)
        
        # Handle potential size mismatch due to odd dimensions
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply double convolution
        x = self.conv(x, time_emb)
        
        return x

