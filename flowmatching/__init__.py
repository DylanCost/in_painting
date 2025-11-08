"""Flow matching image inpainting package.

This package provides a complete implementation of flow matching for
conditional image inpainting on the CelebA dataset.

Main components:
    - data: Dataset loading and masking strategies
    - models: U-Net architecture for velocity prediction
    - flow: Flow matching formulation and ODE sampling
    - training: Training loop, metrics, and checkpointing

Example usage:
    >>> from flowmatching.data import CelebAInpainting
    >>> from flowmatching.models import create_unet
    >>> from flowmatching.training import Trainer
    >>>
    >>> # Create dataset
    >>> dataset = CelebAInpainting(root='./data', split='train')
    >>>
    >>> # Create model
    >>> model = create_unet(in_channels=4, out_channels=3)
    >>>
    >>> # Train
    >>> trainer = Trainer(model, train_loader, val_loader, optimizer)
    >>> trainer.train(num_epochs=100)
"""

# Version info
__version__ = "0.1.0"

# Import key classes for convenience
from .data import CelebAInpainting, RandomRectangularMask
from .models import UNet, create_unet
from .flow import FlowMatching, ODESampler, HeunSampler
from .training import Trainer, CheckpointManager

__all__ = [
    # Data
    "CelebAInpainting",
    "RandomRectangularMask",
    # Models
    "UNet",
    "create_unet",
    # Flow
    "FlowMatching",
    "ODESampler",
    "HeunSampler",
    # Training
    "Trainer",
    "CheckpointManager",
]
