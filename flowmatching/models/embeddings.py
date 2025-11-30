"""Time embedding module for flow matching models.

This module implements sinusoidal positional encodings for time conditioning
in the U-Net architecture, adapted from DDPM-style embeddings.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timesteps.
    Used to encode timestep information in diffusion models.
    Based on the positional encoding from "Attention is All You Need".
    
    Args:
        dim: Dimension of the output embedding
        scale: Scaling factor for time inputs (default: 1000.0).
               Since flow matching uses t in [0, 1], we scale it up to match
               the typical range of DDPM timesteps (0-1000).
    """
    def __init__(self, dim: int, scale: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Timesteps, shape [B] or [B, 1] - values typically in [0, 1]
        
        Returns:
            embeddings: Sinusoidal embeddings, shape [B, dim]
        """
        device = time.device
        
        # Ensure time is shape [B]
        if time.dim() == 2:
            time = time.squeeze(-1)
            
        # Scale time to match DDPM range
        time = time * self.scale
        
        half_dim = self.dim // 2
        
        # Create frequency spectrum
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Apply to timesteps
        embeddings = time[:, None] * embeddings[None, :]
        
        # Concatenate sin and cos
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))
            
        return embeddings


class TimeEmbeddingMLP(nn.Module):
    """
    MLP for processing time embeddings.
    Matches the structure used in UNetDiffusion (Linear -> GELU -> Linear -> GELU).
    
    Args:
        embedding_dim: Input dimension from sinusoidal embedding
        hidden_dim: Hidden dimension for MLP
        output_dim: Output dimension
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 1024,
        output_dim: int = 1024
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )
    
    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_emb: Time embeddings of shape [B, embedding_dim]
            
        Returns:
            Processed embeddings of shape [B, output_dim]
        """
        return self.mlp(t_emb)


class TimestepEmbedding(nn.Module):
    """
    Complete timestep embedding module.
    Combines sinusoidal encoding with MLP processing.
    
    Args:
        embedding_dim: Dimension of sinusoidal embedding (default: 256)
        hidden_dim: Hidden dimension for MLP (default: 1024)
        output_dim: Final output dimension (default: 1024)
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 1024,
        output_dim: int = 1024
    ):
        super().__init__()
        
        self.sinusoidal = SinusoidalPositionEmbeddings(embedding_dim)
        self.mlp = TimeEmbeddingMLP(embedding_dim, hidden_dim, output_dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values of shape [B] or [B, 1]
            
        Returns:
            Processed time embeddings of shape [B, output_dim]
        """
        emb = self.sinusoidal(t)
        emb = self.mlp(emb)
        return emb