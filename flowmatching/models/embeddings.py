"""Time embedding module for flow matching models.

This module implements sinusoidal positional encodings for time conditioning
in the U-Net architecture.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding layer.
    
    Converts scalar time values into high-dimensional embeddings using
    sinusoidal functions at different frequencies, similar to positional
    encodings in transformers.
    
    Args:
        embedding_dim: Dimension of the output embedding (default: 256)
        max_period: Maximum period for the sinusoidal functions (default: 10000)
    
    Input:
        t: Time values of shape [B] or [B, 1], values typically in [0, 1]
    
    Output:
        embeddings: Time embeddings of shape [B, embedding_dim]
    """
    
    def __init__(self, embedding_dim: int = 256, max_period: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal time embeddings.
        
        Args:
            t: Time values of shape [B] or [B, 1]
            
        Returns:
            Time embeddings of shape [B, embedding_dim]
        """
        # Ensure t is shape [B]
        if t.dim() == 2:
            t = t.squeeze(-1)
        
        half_dim = self.embedding_dim // 2
        
        # Compute frequency scaling factors
        # emb = log(max_period) / (half_dim - 1)
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        
        # Compute sinusoidal embeddings
        # emb shape: [half_dim]
        # t[:, None] shape: [B, 1]
        # emb[None, :] shape: [1, half_dim]
        # Result shape: [B, half_dim]
        emb = t[:, None] * emb[None, :]
        
        # Concatenate sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Handle odd embedding dimensions
        if self.embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1), mode='constant', value=0)
        
        return emb


class TimeEmbeddingMLP(nn.Module):
    """MLP for processing time embeddings.
    
    Projects sinusoidal time embeddings to the desired dimension and
    applies non-linear transformations.
    
    Args:
        time_embed_dim: Input dimension from sinusoidal embedding (default: 256)
        hidden_dim: Hidden dimension for MLP (default: 1024)
        output_dim: Output dimension (default: 256)
    
    Input:
        t: Time embeddings of shape [B, time_embed_dim]
    
    Output:
        embeddings: Processed time embeddings of shape [B, output_dim]
    """
    
    def __init__(
        self,
        time_embed_dim: int = 256,
        hidden_dim: int = 1024,
        output_dim: int = 256
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Process time embeddings through MLP.
        
        Args:
            t: Time embeddings of shape [B, time_embed_dim]
            
        Returns:
            Processed embeddings of shape [B, output_dim]
        """
        return self.mlp(t)


class TimestepEmbedding(nn.Module):
    """Complete timestep embedding module.
    
    Combines sinusoidal encoding with MLP processing for time conditioning.
    
    Args:
        embedding_dim: Dimension of sinusoidal embedding (default: 256)
        hidden_dim: Hidden dimension for MLP (default: 1024)
        output_dim: Final output dimension (default: 256)
    
    Input:
        t: Time values of shape [B] or [B, 1], values typically in [0, 1]
    
    Output:
        embeddings: Processed time embeddings of shape [B, output_dim]
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 1024,
        output_dim: int = 256
    ):
        super().__init__()
        
        self.sinusoidal = SinusoidalTimeEmbedding(embedding_dim)
        self.mlp = TimeEmbeddingMLP(embedding_dim, hidden_dim, output_dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Generate and process time embeddings.
        
        Args:
            t: Time values of shape [B] or [B, 1]
            
        Returns:
            Processed time embeddings of shape [B, output_dim]
        """
        emb = self.sinusoidal(t)
        emb = self.mlp(emb)
        return emb