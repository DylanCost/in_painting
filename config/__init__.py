"""Configuration module for VAE inpainting project."""

from config.config import (
    Config,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    MaskConfig,
    LoggingConfig,
)

# Pre-instantiated default configuration
default_config = Config.get_default()

# Pre-instantiated pretrained configuration
pretrained_config = Config.get_pretrained()

__all__ = [
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "MaskConfig",
    "LoggingConfig",
    "default_config",
    "pretrained_config",
]
