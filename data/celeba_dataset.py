import os
from typing import Dict, Any, Optional, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CelebADataset(Dataset):
    """
    CelebA dataset loader that:
    - Returns clean images and minimal metadata (no masking)
    - Uses official CelebA partition file for reproducible splits
    - Optionally auto-downloads via torchvision when data is missing
    - Provides class-level factory methods for config-driven creation
    """

    SPLIT_MAP = {
        'train': 0,
        'val': 1,
        'test': 2,
        'all': -1,
    }

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
        verify_integrity: bool = True,  # kept for API completeness; not used extensively
    ):
        """
        Args:
            root_dir: Path to CelebA dataset directory (expects torchvision layout: {root_dir}/img_align_celeba, list_eval_partition.txt)
            split: One of ['train', 'val', 'test', 'all']
            image_size: Target image size for resizing
            transform: Optional torchvision transform pipeline to apply to images
            download: If True and data is missing, attempt auto-download via torchvision
            verify_integrity: Optionally verify dataset presence (lightweight)
        """
        super().__init__()
        if split not in self.SPLIT_MAP:
            raise ValueError(f"Invalid split '{split}'. Must be one of {list(self.SPLIT_MAP.keys())}.")

        # Normalize root_dir to absolute path
        self.root_dir = os.path.abspath(root_dir)
        self.split = split
        self.image_size = image_size

        # Ensure data exists or download via torchvision
        self._ensure_data(download=download, verify=verify_integrity)

        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform

        # Determine image directory and partition file paths
        self.images_dir = os.path.join(self.root_dir, 'img_align_celeba')
        self.partition_file = self._resolve_partition_file(self.root_dir)

        # Load split membership from official partition file
        self._filename_list = self._load_filenames_for_split(self.partition_file, self.split)

    # -------------------------
    # Factory methods
    # -------------------------
    @classmethod
    def from_config(cls, config, split: str = 'train', download: bool = True) -> "CelebADataset":
        """
        Construct a dataset from Config object.

        Expected config attributes:
            config.data.data_path: str
            config.data.image_size: int
        """
        root_dir = config.data.data_path
        image_size = int(config.data.image_size)

        # Optionally support transform customization in future; default for now
        return cls(
            root_dir=root_dir,
            split=split,
            image_size=image_size,
            transform=None,
            download=download,
            verify_integrity=True,
        )

    @classmethod
    def create_splits_from_config(cls, config: Dict[str, Any], download: bool = True) -> Tuple["CelebADataset", "CelebADataset", "CelebADataset"]:
        """
        Convenience constructor that returns (train, val, test) datasets
        with consistent configuration and optional auto-download.
        """
        train_ds = cls.from_config(config, split='train', download=download)
        val_ds = cls.from_config(config, split='val', download=False)   # Already downloaded by train_ds
        test_ds = cls.from_config(config, split='test', download=False) # Already downloaded by train_ds
        return train_ds, val_ds, test_ds

    # -------------------------
    # Dataset protocol
    # -------------------------
    def __len__(self) -> int:
        return len(self._filename_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename = self._filename_list[idx]
        img_path = os.path.join(self.images_dir, filename)

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        image_tensor = self.transform(image)

        return {
            'image': image_tensor,
            'filename': filename,
            'idx': idx,
        }

    # -------------------------
    # Internal helpers
    # -------------------------
    def _ensure_data(self, download: bool, verify: bool) -> None:
        """
        Ensure CelebA data exists at self.root_dir. If not and download=True,
        attempt to download via torchvision using the parent directory as root.
        """
        images_dir = os.path.join(self.root_dir, 'img_align_celeba')
        partition_path = self._resolve_partition_file(self.root_dir)

        have_images = os.path.isdir(images_dir) and len(os.listdir(images_dir)) > 0
        have_partition = partition_path is not None and os.path.isfile(partition_path)

        if have_images and have_partition:
            return

        if not download:
            missing = []
            if not have_images:
                missing.append(f"images at {images_dir}")
            if not have_partition:
                missing.append("list_eval_partition.txt")
            raise FileNotFoundError(
                "CelebA dataset not found. Missing: " + ", ".join(missing) +
                f". Set download=True or prepare dataset at {self.root_dir}."
            )

        # Attempt auto-download via torchvision
        try:
            import torchvision
            parent = os.path.dirname(self.root_dir)
            # This will place files under {parent}/celeba/
            torchvision.datasets.CelebA(root=parent, split='all', download=True)
            # If self.root_dir wasn't {parent}/celeba, prefer the torchvision layout
            torchvision_dir = os.path.join(parent, 'celeba')
            if os.path.abspath(torchvision_dir) != self.root_dir:
                # Prefer torchvision dir to ensure consistent structure
                self.root_dir = os.path.abspath(torchvision_dir)
        except Exception as e:
            raise RuntimeError(f"Automatic download with torchvision failed: {e}")

        # Final simple verification
        images_dir = os.path.join(self.root_dir, 'img_align_celeba')
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Downloaded CelebA but missing images directory: {images_dir}")
        if self._resolve_partition_file(self.root_dir) is None:
            raise FileNotFoundError("Downloaded CelebA but could not locate list_eval_partition.txt")

    @staticmethod
    def _resolve_partition_file(base_dir: str) -> Optional[str]:
        """
        Locate the official partition file (list_eval_partition.txt) under torchvision layout.
        Common locations observed:
            - {base_dir}/list_eval_partition.txt
            - {base_dir}/Anno/list_eval_partition.txt
        """
        candidates = [
            os.path.join(base_dir, 'list_eval_partition.txt'),
            os.path.join(base_dir, 'Anno', 'list_eval_partition.txt'),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    @classmethod
    def _load_filenames_for_split(cls, partition_file: str, split: str) -> List[str]:
        """
        Parse the official partition file and return filenames belonging to the requested split.
        """
        target = cls.SPLIT_MAP[split]
        filenames: List[str] = []
        with open(partition_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                filename, partition_str = parts
                try:
                    partition = int(partition_str)
                except ValueError:
                    continue

                if target == -1:  # 'all'
                    filenames.append(filename)
                else:
                    if partition == target:
                        filenames.append(filename)

        if not filenames:
            raise ValueError(f"No filenames found for split='{split}' using '{partition_file}'.")
        return sorted(filenames)