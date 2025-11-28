import os
from typing import Dict, Any, Optional
import torch
from torchvision import transforms
from torchvision.datasets import CelebA as TorchvisionCelebA


class CelebADataset(TorchvisionCelebA):
    """
    Wrapper around torchvision.datasets.CelebA that:
    - Returns dict format compatible with existing pipeline
    - Provides default transforms matching original behavior
    - Implements a custom split scheme:
        * Validation: fixed subset of 1024 images from the official CelebA
          validation partition (first 1024 in partition order)
        * Training: all official train images plus the remaining validation
          images not used for validation
        * Test: official CelebA test split (unchanged)
    - Enables subclassing for specialized datasets
    """

    # Number of images to keep in the (small) validation split
    VAL_SET_SIZE = 1024

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ):
        """
        Args:
            root_dir: Path to CelebA dataset directory
            split: One of ['train', 'val', 'test', 'all']
            image_size: Target image size for resizing
            transform: Optional torchvision transform pipeline
            download: If True, download dataset if missing
            verify_integrity: Verify dataset integrity (unused, kept for API compatibility)
        """
        # Validate and normalize split
        allowed_splits = {"train", "val", "test", "all"}
        if split not in allowed_splits:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of {sorted(allowed_splits)}."
            )

        # Store for later use
        self.split = split  # user-facing split string
        self.normalized_split = split
        self.image_size = image_size

        # Default transforms if none provided
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

        # Initialize parent with 'all' split so we can re-partition ourselves
        super().__init__(
            root=root_dir,
            split="all",
            target_type="attr",  # Required param, we don't use attributes
            transform=transform,
            download=download,
        )

        # Build custom train/val/test indices from the official CelebA
        # evaluation partition file.
        partitions = self._load_partitions()

        # 0 = official train, 1 = official val, 2 = official test
        train_indices_official = [i for i, p in enumerate(partitions) if p == 0]
        val_indices_official = [i for i, p in enumerate(partitions) if p == 1]
        test_indices_official = [i for i, p in enumerate(partitions) if p == 2]

        if len(val_indices_official) < self.VAL_SET_SIZE:
            raise RuntimeError(
                f"Official CelebA validation split has only {len(val_indices_official)} "
                f"samples, cannot create a fixed {self.VAL_SET_SIZE}-sample validation subset."
            )

        # Fixed small validation subset: first VAL_SET_SIZE validation samples
        val_indices_small = val_indices_official[: self.VAL_SET_SIZE]
        # Remaining validation samples are merged into the training split
        val_indices_to_train = val_indices_official[self.VAL_SET_SIZE :]

        # Extended training split: official train + leftover validation
        train_indices_extended = train_indices_official + val_indices_to_train

        # 'all' always refers to the full dataset
        all_indices = list(range(len(partitions)))

        # Select indices for this instance based on requested split
        if self.normalized_split == "train":
            self._indices = train_indices_extended
        elif self.normalized_split == "val":
            self._indices = val_indices_small
        elif self.normalized_split == "test":
            self._indices = test_indices_official
        elif self.normalized_split == "all":
            self._indices = all_indices
        else:
            # This should never happen due to earlier validation
            raise RuntimeError(
                f"Unhandled split '{self.normalized_split}' in CelebADataset."
            )

    def __len__(self) -> int:
        """Return number of samples in this logical split."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Override to return dict format instead of tuple.

        Returns:
            Dict with keys:
                - 'image': Transformed image tensor
                - 'filename': Original filename (e.g., '000001.jpg')
                - 'idx': Global index in the full CelebA dataset ('all' split)
        """
        # Map local index within the split to the global index in the
        # underlying 'all' CelebA dataset.
        real_idx = self._indices[idx]

        # Call parent to get (image, attr)
        image, _ = super().__getitem__(real_idx)

        # Get filename from internal filename list using global index
        filename = self.filename[real_idx]

        return {
            "image": image,
            "filename": filename,
            "idx": real_idx,
        }

    def _load_partitions(self):
        """
        Load the official CelebA evaluation partition information.

        Returns:
            List of integers of length N (number of images in 'all' split),
            where each entry is:
                0 = train, 1 = validation, 2 = test
        """
        # Torchvision stores CelebA under root/base_folder; the
        # list_eval_partition.txt file can be located by walking that tree.
        base_dir = os.path.join(self.root, self.base_folder)
        partition_file = None

        for dirpath, _, filenames in os.walk(base_dir):
            if "list_eval_partition.txt" in filenames:
                partition_file = os.path.join(dirpath, "list_eval_partition.txt")
                break

        if partition_file is None:
            raise FileNotFoundError(
                f"Could not locate 'list_eval_partition.txt' under '{base_dir}'. "
                "Please ensure the CelebA dataset is correctly downloaded."
            )

        partitions = []
        with open(partition_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # Expected format: '<filename> <partition_id>'
                try:
                    partition_id = int(parts[-1])
                except (ValueError, IndexError) as e:
                    raise RuntimeError(
                        f"Malformed line in list_eval_partition.txt: {line!r}"
                    ) from e
                partitions.append(partition_id)

        if len(partitions) != len(self.filename):
            raise RuntimeError(
                "Mismatch between number of partition entries "
                f"({len(partitions)}) and number of images ({len(self.filename)})."
            )

        return partitions
