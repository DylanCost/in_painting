"""Common configuration system for inpainting project.

This module contains configuration classes that are shared across all model variants
(VAE, FlowMatching, Diffusion, etc.). Model-specific configurations should be defined
in separate config files (e.g., vae_config.py).
"""

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Dict, List, Optional, Sequence


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
    learning_rate: float = 0.0002 #0.0002
    epochs: int = 100


@dataclass
class MaskConfig:
    """Masking strategy configuration - common across all models."""

    type: str = "random"  # random, center, irregular
    mask_ratio: float = 0.4
    min_size: int = 16
    max_size: int = 64
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


@dataclass(frozen=True)
class ManualMaskSpec:
    """Fixed rectangular mask definition bound to a dataset index."""

    index: int
    top: int
    left: int
    height: int
    width: int

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def right(self) -> int:
        return self.left + self.width

    def as_dict(self) -> Dict[str, int]:
        return {
            "index": self.index,
            "top": self.top,
            "left": self.left,
            "height": self.height,
            "width": self.width,
        }


TRIPTYCH_MASK_VERSION = "v1"

DEFAULT_TRIPTYCH_MASKS: Dict[int, ManualMaskSpec] = {
    0: ManualMaskSpec(index=0, top=28, left=20, height=56, width=64),
    1: ManualMaskSpec(index=1, top=48, left=32, height=56, width=56),
    2: ManualMaskSpec(index=2, top=18, left=56, height=46, width=54),
    3: ManualMaskSpec(index=3, top=10, left=24, height=60, width=60),
    4: ManualMaskSpec(index=4, top=44, left=8, height=68, width=52),
    5: ManualMaskSpec(index=5, top=30, left=72, height=54, width=44),
    6: ManualMaskSpec(index=6, top=12, left=36, height=64, width=42),
    7: ManualMaskSpec(index=7, top=58, left=48, height=54, width=60),
}


def get_triptych_mask_specs(indices: Optional[Sequence[int]] = None) -> List[ManualMaskSpec]:
    """Return manual mask specs filtered by dataset indices (defaults to all)."""

    if indices is None:
        indices = sorted(DEFAULT_TRIPTYCH_MASKS.keys())

    specs: List[ManualMaskSpec] = []
    for idx in indices:
        if idx not in DEFAULT_TRIPTYCH_MASKS:
            raise KeyError(
                f"No manual mask specification found for dataset index {idx}."
            )
        specs.append(DEFAULT_TRIPTYCH_MASKS[idx])
    return specs


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
