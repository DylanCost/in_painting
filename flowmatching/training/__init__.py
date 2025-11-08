"""Training package for flow matching image inpainting.

This package contains training utilities including the main trainer,
metrics computation, and checkpoint management.
"""

from .metrics import compute_psnr, compute_ssim, compute_metrics
from .checkpoint import CheckpointManager, save_checkpoint, load_checkpoint
from .trainer import Trainer

__all__ = [
    # Metrics
    "compute_psnr",
    "compute_ssim",
    "compute_metrics",
    
    # Checkpointing
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    
    # Training
    "Trainer",
]