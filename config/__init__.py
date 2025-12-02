"""Configuration module for VAE inpainting project."""

from config.vae_config import (
    VAEConfig,
    VAEModelConfig,
    VAETrainingConfig,
    VAEDataConfig,
    VAEMaskConfig,
    VAELoggingConfig,
)

# Pre-instantiated default configuration
default_config = VAEConfig.get_default()

# Pre-instantiated pretrained configuration
pretrained_config = VAEConfig.get_pretrained()

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
