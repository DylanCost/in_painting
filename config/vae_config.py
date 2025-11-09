"""VAE-specific configuration for inpainting project."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from config.common_config import (
    Config,
    UNetConfig,
    DataConfig,
    MaskConfig,
    LoggingConfig,
)


@dataclass
class VAEModelConfig:
    """VAE-specific model configuration (non-U-Net settings)."""

    name: str = "unet_vae"
    latent_dim: int = 512
    pretrained_encoder: Optional[str] = None
    encoder_checkpoint: Optional[str] = None
    freeze_encoder_stages: int = 0


@dataclass
class VAETrainingConfig:
    """VAE training hyperparameters configuration."""

    batch_size: int = 32
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    epochs: int = 100
    gradient_clip: float = 1.0
    kl_weight: float = 0.001
    perceptual_weight: float = 0.1
    adversarial_weight: float = 0.001
    encoder_lr: Optional[float] = None
    decoder_lr: Optional[float] = None
    warmup_epochs: int = 0
    unfreeze_schedule: Dict[str, int] = field(default_factory=dict)


@dataclass
class VAEConfig(Config):
    """VAE-specific configuration combining all sub-configs."""

    model: VAEModelConfig = field(default_factory=VAEModelConfig)
    training: VAETrainingConfig = field(default_factory=VAETrainingConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @staticmethod
    def get_default() -> "VAEConfig":
        """Get default VAE configuration."""
        return VAEConfig(
            model=VAEModelConfig(),
            training=VAETrainingConfig(),
            unet=UNetConfig(),
            data=DataConfig(),
            mask=MaskConfig(),
            logging=LoggingConfig(),
        )

    @staticmethod
    def get_pretrained() -> "VAEConfig":
        """Get configuration for training with pretrained weights."""
        config = VAEConfig.get_default()

        # Model settings for pretrained
        config.model.pretrained_encoder = "resnet"
        config.model.encoder_checkpoint = None
        config.model.freeze_encoder_stages = 2

        # Modified training settings for transfer learning
        config.training.learning_rate = 0.0001
        config.training.encoder_lr = 0.00001
        config.training.decoder_lr = 0.0002
        config.training.warmup_epochs = 5
        config.training.unfreeze_schedule = {
            "epoch_5": 3,
            "epoch_10": 2,
            "epoch_15": 1,
            "epoch_20": 0,
        }

        return config
