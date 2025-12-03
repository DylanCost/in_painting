"""Configuration module for VAE inpainting project."""

from config.vae_config import (
    VAEConfig,
    VAEModelConfig,
    VAETrainingConfig,
    VAEDataConfig,
    VAEMaskConfig,
    VAELoggingConfig,
)

from config.common_config import (
    Config,
    UNetConfig,
    DataConfig,
    MaskConfig,
    LoggingConfig,
    ManualMaskSpec,
    get_triptych_mask_specs,
    TRIPTYCH_MASK_VERSION,
)

# Pre-instantiated default configuration
default_config = VAEConfig.get_default()

# Pre-instantiated pretrained configuration
pretrained_config = VAEConfig.get_pretrained()

__all__ = [
    "Config",
    "UNetConfig",
    "DataConfig",
    "MaskConfig",
    "LoggingConfig",
    "ManualMaskSpec",
    "get_triptych_mask_specs",
    "TRIPTYCH_MASK_VERSION",
    "VAEConfig",
    "VAEModelConfig",
    "VAETrainingConfig",
    "VAEDataConfig",
    "VAEMaskConfig",
    "VAELoggingConfig",
    "default_config",
    "pretrained_config",
]
