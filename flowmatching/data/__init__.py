"""Data pipeline module for CelebA inpainting.

This module provides dataset loading, masking strategies, and data
transformations for the image inpainting task.
"""

from .dataset import CelebAInpainting

__all__ = ["CelebAInpainting"]
