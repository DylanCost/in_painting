"""Masking strategies for image inpainting.

This module provides various masking strategies for creating training
and evaluation data for image inpainting tasks.
"""

import torch
import random
from typing import Tuple, Optional
import sys
import os

# Add parent directory to path to import MaskGenerator
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from masking.mask_generator import MaskGenerator


class RandomRectangularMask:
    """Generate random rectangular masks for image inpainting.

    Creates binary masks with random rectangular regions marked for inpainting.
    The mask has value 1 in regions to be inpainted and 0 in observed regions.

    Args:
        image_size: Size of the image (assumes square images), default: 128
        min_size: Minimum size of the rectangular mask (default: 16)
        max_size: Maximum size of the rectangular mask (default: 64)
        channels: Number of mask channels (default: 1)

    Example:
        >>> masker = RandomRectangularMask(image_size=128, min_size=16, max_size=64)
        >>> mask = masker.generate_mask(batch_size=4)
        >>> print(mask.shape)  # [4, 1, 128, 128]
        >>> print(mask.unique())  # tensor([0., 1.])

    Note:
        - Mask value 1 indicates pixels to be inpainted (masked/unknown)
        - Mask value 0 indicates observed pixels (known)
        - This convention is consistent with the flow matching formulation
    """

    def __init__(
        self,
        image_size: int = 128,
        min_size: int = 16,
        max_size: int = 64,
        channels: int = 1,
        mask_type: str = "random",
        seed: Optional[int] = None,
    ):
        """Initialize the random rectangular mask generator.

        Args:
            image_size: Size of the square image
            min_size: Minimum dimension of the rectangular mask
            max_size: Maximum dimension of the rectangular mask
            channels: Number of channels in the mask (typically 1)
            mask_type: Type of mask to generate ('random', 'center', 'irregular')
            seed: Random seed for reproducible mask generation (optional)

        Raises:
            ValueError: If min_size > max_size or sizes are invalid
        """
        if min_size > max_size:
            raise ValueError(f"min_size ({min_size}) must be <= max_size ({max_size})")

        if min_size < 1 or max_size > image_size:
            raise ValueError(
                f"Invalid mask sizes: min_size={min_size}, max_size={max_size}, "
                f"image_size={image_size}"
            )

        self.image_size = image_size
        self.min_size = min_size
        self.max_size = max_size
        self.channels = channels
        self.mask_type = mask_type
        self.seed = seed

        # Create internal MaskGenerator instance
        self._mask_generator = MaskGenerator(
            mask_type=mask_type, min_size=min_size, max_size=max_size, seed=seed
        )

    def generate_mask(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate random rectangular masks for a batch.

        Each mask in the batch has a randomly sized and positioned rectangle.

        Args:
            batch_size: Number of masks to generate
            device: Device to create tensor on (default: None, uses CPU)

        Returns:
            Binary mask tensor of shape [batch_size, channels, image_size, image_size]
            with 1s in the masked region and 0s elsewhere

        Example:
            >>> masker = RandomRectangularMask(128, 16, 64)
            >>> masks = masker.generate_mask(batch_size=8)
            >>> print(masks.shape)  # [8, 1, 128, 128]
        """
        if device is None:
            device = torch.device("cpu")

        # Use internal MaskGenerator to generate masks
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        masks = self._mask_generator.generate(shape)

        # Move to specified device if needed
        if masks.device != device:
            masks = masks.to(device)

        return masks

    def generate_fixed_mask(
        self,
        batch_size: int = 1,
        height: int = None,
        width: int = None,
        top: int = None,
        left: int = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate fixed rectangular masks with specified dimensions and position.

        Useful for evaluation or testing with consistent masks.

        Args:
            batch_size: Number of identical masks to generate
            height: Height of the rectangle (default: uses max_size)
            width: Width of the rectangle (default: uses max_size)
            top: Top position of the rectangle (default: centered)
            left: Left position of the rectangle (default: centered)
            device: Device to create tensor on

        Returns:
            Binary mask tensor of shape [batch_size, channels, image_size, image_size]

        Example:
            >>> masker = RandomRectangularMask(128, 16, 64)
            >>> # Create centered 32x32 mask
            >>> masks = masker.generate_fixed_mask(4, height=32, width=32)
        """
        if device is None:
            device = torch.device("cpu")

        # Use defaults if not specified
        if height is None:
            height = self.max_size
        if width is None:
            width = self.max_size
        if top is None:
            top = (self.image_size - height) // 2
        if left is None:
            left = (self.image_size - width) // 2

        # Validate dimensions
        if top + height > self.image_size or left + width > self.image_size:
            raise ValueError(
                f"Mask extends beyond image: top={top}, left={left}, "
                f"height={height}, width={width}, image_size={self.image_size}"
            )

        # Initialize masks with zeros
        masks = torch.zeros(
            batch_size, self.channels, self.image_size, self.image_size, device=device
        )

        # Set masked region to 1 for all masks in batch
        masks[:, :, top : top + height, left : left + width] = 1.0

        return masks

    def get_mask_ratio(self, mask: torch.Tensor) -> float:
        """Calculate the ratio of masked pixels to total pixels.

        Args:
            mask: Binary mask tensor of shape [B, C, H, W]

        Returns:
            Ratio of masked pixels (value 1) to total pixels

        Example:
            >>> masker = RandomRectangularMask(128, 16, 64)
            >>> mask = masker.generate_mask(1)
            >>> ratio = masker.get_mask_ratio(mask)
            >>> print(f"Masked ratio: {ratio:.2%}")
        """
        return mask.mean().item()

    def __repr__(self) -> str:
        """String representation of the masker."""
        return (
            f"RandomRectangularMask(image_size={self.image_size}, "
            f"min_size={self.min_size}, max_size={self.max_size}, "
            f"channels={self.channels})"
        )


class MultiRectangularMask(RandomRectangularMask):
    """Generate multiple random rectangular masks per image.

    Extension of RandomRectangularMask that can create multiple
    non-overlapping rectangular masks on a single image.

    Args:
        image_size: Size of the image (assumes square images)
        min_size: Minimum size of each rectangular mask
        max_size: Maximum size of each rectangular mask
        num_masks: Number of rectangular masks per image (default: 2)
        channels: Number of mask channels (default: 1)

    Example:
        >>> masker = MultiRectangularMask(128, 16, 32, num_masks=3)
        >>> mask = masker.generate_mask(batch_size=4)
        >>> # Each image will have 3 rectangular masked regions
    """

    def __init__(
        self,
        image_size: int = 128,
        min_size: int = 16,
        max_size: int = 64,
        num_masks: int = 2,
        channels: int = 1,
    ):
        """Initialize multi-rectangular mask generator.

        Args:
            image_size: Size of the square image
            min_size: Minimum dimension of each rectangular mask
            max_size: Maximum dimension of each rectangular mask
            num_masks: Number of masks per image
            channels: Number of channels in the mask
        """
        super().__init__(image_size, min_size, max_size, channels)
        self.num_masks = num_masks

    def generate_mask(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate masks with multiple rectangular regions.

        Args:
            batch_size: Number of masks to generate
            device: Device to create tensor on

        Returns:
            Binary mask tensor with multiple masked regions per image
        """
        if device is None:
            device = torch.device("cpu")

        # Initialize masks with zeros
        masks = torch.zeros(
            batch_size, self.channels, self.image_size, self.image_size, device=device
        )

        # Generate multiple rectangles for each mask in the batch
        for i in range(batch_size):
            for _ in range(self.num_masks):
                # Random mask dimensions
                mask_h = random.randint(self.min_size, self.max_size)
                mask_w = random.randint(self.min_size, self.max_size)

                # Random position
                top = random.randint(0, self.image_size - mask_h)
                left = random.randint(0, self.image_size - mask_w)

                # Set masked region to 1 (can overlap with previous masks)
                masks[i, :, top : top + mask_h, left : left + mask_w] = 1.0

        return masks

    def __repr__(self) -> str:
        """String representation of the masker."""
        return (
            f"MultiRectangularMask(image_size={self.image_size}, "
            f"min_size={self.min_size}, max_size={self.max_size}, "
            f"num_masks={self.num_masks}, channels={self.channels})"
        )
