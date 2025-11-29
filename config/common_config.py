"""Common configuration system for inpainting project.

This module contains configuration classes that are shared across all model variants
(VAE, FlowMatching, Diffusion, etc.). Model-specific configurations should be defined
in separate config files (e.g., vae_config.py).
"""

from dataclasses import dataclass, field
from copy import deepcopy
from typing import List


@dataclass
class UNetConfig:
    """U-Net architecture configuration - shared across all models."""

    input_channels: int = 3
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 512])
    use_attention: bool = True
    use_skip_connections: bool = True
    dropout: float = 0.1


@dataclass
class DataConfig:
    """Dataset configuration - common across all models."""

    dataset: str = "celeba"
    data_path: str = "./assets/datasets"
    image_size: int = 128
    num_workers: int = 4
    augmentation: bool = True
    batch_size: int = 64
    learning_rate: float = 0.0002
    epochs: int = 5


@dataclass
class MaskConfig:
    """Masking strategy configuration - common across all models."""

    type: str = "random"  # random, center, irregular
    mask_ratio: float = 0.4
    min_size: int = 32
    max_size: int = 128
    seed: int = 42
    cache_dir: str = "./assets/masks"


@dataclass
class LoggingConfig:
    """Experiment tracking and logging configuration - common across all models."""

    use_wandb: bool = True
    use_tensorboard: bool = True
    log_interval: int = 100
    save_interval: int = 5
    sample_interval: int = 500
    checkpoint_dir: str = "./weights"
    log_dir: str = "./logs"


@dataclass
class Config:
    """Base configuration class combining common sub-configs.

    Model-specific configurations should inherit from this class and add
    their own model and training config classes.
    """

    unet: UNetConfig = field(default_factory=UNetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def copy(self) -> "Config":
        """Create a deep copy of the config."""
        return deepcopy(self)
