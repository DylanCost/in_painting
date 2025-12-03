"""Building blocks for U-Net architecture.

This module contains reusable components for the U-Net model including
encoder/decoder blocks and self-attention mechanisms, adapted from
diffusion model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """
    Encoder block with double convolution and time embedding injection.
    
    Architecture:
        Conv(stride=2) -> BN -> GELU -> Conv -> BN -> GELU
        + Time embedding injection
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_emb_dim: Dimension of time embedding
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
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
        """
        Args:
            x: Input tensor [B, in_channels, H, W]
            t_emb: Time embedding [B, time_emb_dim]
        """
        x = self.conv1(x)
        # Add time embedding
        x = x + self.time_proj(t_emb)[:, :, None, None]
        x = self.conv2(x)
        return x


class DecoderBlock(nn.Module):
    """
    Decoder block with transposed convolution and time embedding injection.
    
    Architecture:
        ConvTranspose(stride=2) -> BN -> ReLU -> Conv -> BN -> ReLU
        + Time embedding injection
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_emb_dim: Dimension of time embedding
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Time projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, in_channels, H, W]
            t_emb: Time embedding [B, time_emb_dim]
        """
        x = self.conv1(x)
        # Add time embedding
        x = x + self.time_proj(t_emb)[:, :, None, None]
        x = self.conv2(x)
        return x


class SelfAttention(nn.Module):
    """
    Self-attention module for capturing long-range dependencies.
    
    Args:
        channels: Number of input/output channels
    """
    def __init__(self, channels: int):
        super().__init__()
        
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        """
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
