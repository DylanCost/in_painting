"""CelebA dataset wrapper for image inpainting.

This module provides a PyTorch dataset for loading CelebA images with
random masking for the inpainting task.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import CelebA
from typing import Dict, Optional, Tuple, Literal
import os
import sys

# Add parent directory to path to import CelebADataset
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data.celeba_dataset import CelebADataset
from masking.mask_generator import MaskGenerator


class CelebAInpainting(Dataset):
    """CelebA dataset wrapper for image inpainting with random masking.

    Loads CelebA images, applies transformations, and generates random
    rectangular masks for training or evaluation. Returns a dictionary
    containing the original image, masked image, and mask.

    Args:
        root: Root directory where CelebA dataset will be downloaded/stored
        split: Dataset split - 'train', 'valid', or 'test' (default: 'train')
        image_size: Size to resize images to (default: 128)
        min_mask_size: Minimum size of rectangular mask (default: 16)
        max_mask_size: Maximum size of rectangular mask (default: 64)
        download: Whether to download the dataset if not found (default: True)
        transform: Optional custom transform (overrides default)
        normalize: Whether to normalize images to [-1, 1] (default: True)

    Returns:
        Dictionary with keys:
            - 'image': Original complete image, shape [3, H, W]
            - 'mask': Binary mask (1 = masked, 0 = observed), shape [1, H, W]

    Example:
        >>> dataset = CelebAInpainting(
        ...     root='./data',
        ...     split='train',
        ...     image_size=128
        ... )
        >>> sample = dataset[0]
        >>> print(sample['image'].shape)  # [3, 128, 128]
        >>> print(sample['mask'].shape)   # [1, 128, 128]
    """

    def __init__(
        self,
        root: str = "./assets/datasets",
        split: Literal["train", "valid", "test"] = "train",
        image_size: int = 128,
        min_mask_size: int = 16,
        max_mask_size: int = 64,
        download: bool = True,
        transform: Optional[transforms.Compose] = None,
        normalize: bool = True,
        mask_type: str = "random",
        mask_seed: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize CelebA inpainting dataset.

        Args:
            root: Root directory for dataset
            split: Dataset split to use
            image_size: Target image size
            min_mask_size: Minimum mask dimension
            max_mask_size: Maximum mask dimension
            download: Whether to download dataset
            transform: Custom transform (overrides default)
            normalize: Whether to normalize to [-1, 1]
            mask_type: Type of mask to generate ('random', 'center', 'irregular')
            mask_seed: Random seed for reproducible mask generation (optional)
            cache_dir: Directory for caching deterministic masks (optional)
        """
        self.root = root
        self.split = split
        self.image_size = image_size
        self.normalize = normalize

        # Create transform if not provided
        if transform is None:
            transform_list = [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]

            # Add horizontal flip for training
            if split == "train":
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

            transform_list.append(transforms.ToTensor())

            # Normalize to [-1, 1] if requested
            if normalize:
                transform_list.append(
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                )

            transform = transforms.Compose(transform_list)

        self.transform = transform

        # Load CelebA dataset using CelebADataset wrapper
        self.celeba = CelebADataset(
            root_dir=root,
            split=(
                split if split != "valid" else "val"
            ),  # Map 'valid' to 'val' for CelebADataset
            image_size=image_size,
            transform=transform,
            download=download,
        )

        # Initialize mask generator
        # Use deterministic masks for validation and test splits
        deterministic = split in ["valid", "test"]
        self.mask_generator = MaskGenerator(
            mask_type=mask_type,
            min_size=min_mask_size,
            max_size=max_mask_size,
            seed=mask_seed,
            cache_dir=cache_dir,
            deterministic=deterministic,
        )

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.celeba)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing:
                - 'image': Original image [3, H, W]
                - 'mask': Binary mask [1, H, W]
        """
        # Load image from CelebADataset (returns dict with 'image', 'filename', 'idx')
        sample = self.celeba[idx]
        image = sample["image"]
        filename = sample["filename"]

        # Generate mask - deterministic for val/test, random for train
        if self.mask_generator.deterministic:
            # Use filename-based deterministic mask generation
            # Returns [1, 1, H, W] for single filename
            mask = self.mask_generator.generate_for_filenames(
                [filename], shape=(1, self.image_size, self.image_size)
            )
            mask = mask.squeeze(0)  # Remove batch dimension: [1, 1, H, W] -> [1, H, W]
        else:
            # Random mask generation for training
            # generate() with shape (1, H, W) returns [1, 1, H, W]
            mask = self.mask_generator.generate(
                shape=(1, self.image_size, self.image_size)
            )
            mask = mask.squeeze(0)  # Remove extra dimension: [1, 1, H, W] -> [1, H, W]

        # Move mask to same device as image
        if mask.device != image.device:
            mask = mask.to(image.device)

        # Return original image - flow matching will handle the masking
        # by interpolating masked regions with noise
        return {"image": image, "mask": mask}

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = None,
        subsample_fraction: float = 1.0,
    ) -> DataLoader:
        """Create a DataLoader for this dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle (default: True for train, False otherwise)
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop last incomplete batch (default: True for train)
            subsample_fraction: Fraction of dataset to use (default: 1.0 for full dataset)

        Returns:
            DataLoader instance

        Example:
            >>> dataset = CelebAInpainting(root='./data', split='train')
            >>> loader = dataset.get_dataloader(batch_size=32)
            >>> for batch in loader:
            ...     images = batch['image']
            ...     masks = batch['mask']
            ...     # Training code here

            >>> # Use only 10% of data for debugging
            >>> loader = dataset.get_dataloader(batch_size=32, subsample_fraction=0.1)
        """
        # Set defaults based on split
        if shuffle is None:
            shuffle = self.split == "train"
        if drop_last is None:
            drop_last = self.split == "train"

        # Apply subsampling if requested
        dataset = self
        if subsample_fraction < 1.0:
            total_samples = len(self)
            subsample_size = int(total_samples * subsample_fraction)
            indices = list(range(subsample_size))
            dataset = Subset(self, indices)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"CelebAInpainting(split='{self.split}', "
            f"size={len(self)}, image_size={self.image_size}, "
            f"mask_range=[{self.mask_generator.min_size}, "
            f"{self.mask_generator.max_size}])"
        )


def create_train_val_datasets(
    root: str = "./assets/datasets",
    image_size: int = 128,
    min_mask_size: int = 16,
    max_mask_size: int = 64,
    train_ratio: float = 0.9,
    download: bool = True,
    seed: int = 42,
) -> Tuple[CelebAInpainting, CelebAInpainting]:
    """Create train and validation datasets with a custom split.

    This function creates a custom train/validation split from the CelebA
    training set, which is useful when you want more control over the split
    ratio than the default CelebA splits provide.

    Args:
        root: Root directory for dataset
        image_size: Target image size
        min_mask_size: Minimum mask dimension
        max_mask_size: Maximum mask dimension
        train_ratio: Ratio of training data (default: 0.9 for 90/10 split)
        download: Whether to download dataset
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_dataset, val_dataset)

    Example:
        >>> train_ds, val_ds = create_train_val_datasets(
        ...     root='./data',
        ...     train_ratio=0.9,
        ...     seed=42
        ... )
        >>> print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    Note:
        This uses the official CelebA 'train' split and further divides it.
        For the official validation set, use split='valid' directly.
    """
    # Create full training dataset
    train_dataset = CelebAInpainting(
        root=root,
        split="train",
        image_size=image_size,
        min_mask_size=min_mask_size,
        max_mask_size=max_mask_size,
        download=download,
    )

    val_dataset = CelebAInpainting(
        root=root,
        split="val",
        image_size=image_size,
        min_mask_size=min_mask_size,
        max_mask_size=max_mask_size,
        download=download,
    )

    # Wrap in CelebAInpainting-like interface
    # Note: The subsets will still have the same transform (with augmentation)
    # If you want different transforms for train/val, create separate datasets

    return train_dataset, val_dataset


def create_dataloaders(
    root: str = "./assets/datasets",
    image_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with default settings.

    Convenience function that creates both datasets and dataloaders
    with sensible defaults for training.

    Args:
        root: Root directory for dataset
        image_size: Target image size
        batch_size: Batch size for both loaders
        num_workers: Number of worker processes
        train_ratio: Ratio of training data
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        >>> train_loader, val_loader = create_dataloaders(
        ...     root='./data',
        ...     batch_size=32,
        ...     num_workers=4
        ... )
        >>> for batch in train_loader:
        ...     # Training code
        ...     pass
    """
    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(
        root=root, image_size=image_size, train_ratio=train_ratio, seed=seed
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
