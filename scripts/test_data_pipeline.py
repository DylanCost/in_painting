#!/usr/bin/env python
"""
Visual verification script for the data pipeline.

- Builds CelebA datasets via class factory (auto-download with torchvision if missing)
- Samples a few images from each split
- Generates masks:
    - Train: dynamic masks
    - Val/Test: deterministic per-filename masks with optional on-disk caching
- Saves side-by-side grids [clean, masked] for manual inspection

Usage:
    python scripts/test_data_pipeline.py --config default --num-samples 8 --device cuda
"""

import argparse
import os
import random
import sys
from typing import Dict

import torch
import torchvision.utils as vutils

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import default_config, pretrained_config
from data.celeba_dataset import CelebADataset  # noqa: E402
from masking import MaskGenerator  # noqa: E402


def denormalize_for_vis(x: torch.Tensor) -> torch.Tensor:
    """
    Expect input in [-1, 1]; map to [0, 1] for saving.
    """
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def dump_samples(
    ds,
    mask_gen: MaskGenerator,
    out_dir: str,
    deterministic: bool = False,
    n: int = 8,
    device: str = "cpu"
):
    os.makedirs(out_dir, exist_ok=True)
    length = len(ds)
    if length == 0:
        print(f"Warning: dataset has 0 samples, skipping dump for {out_dir}")
        return

    idxs = random.sample(range(length), min(n, length))

    for i in idxs:
        sample = ds[i]
        img = sample["image"].unsqueeze(0)  # [1,3,H,W]
        filename = sample["filename"]
        _, _, H, W = img.shape

        if deterministic:
            masks = mask_gen.generate_for_filenames(
                [filename],
                (1, H, W),
                cache_dir=mask_gen.cache_dir or "./assets/masks"
            )
        else:
            masks = mask_gen.generate((1, 1, H, W))

        # Compose masked image
        masked = img * (1.0 - masks)

        # Grid: [clean, masked]
        grid = vutils.make_grid(
            torch.cat([img, masked], dim=0),
            nrow=2,
            normalize=False
        )
        grid = denormalize_for_vis(grid)

        save_path = os.path.join(out_dir, f"{i:06d}.png")
        vutils.save_image(grid, save_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize data pipeline with masks")
    parser.add_argument("--config", type=str, choices=['default', 'pretrained'], default="default")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    # Load config
    if args.config == 'pretrained':
        config = pretrained_config.copy()
    else:
        config = default_config.copy()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Build datasets (auto-download if missing)
    train_ds = CelebADataset(
        root_dir=config.data.data_path,
        split='train',
        image_size=config.data.image_size,
        download=True
    )
    val_ds = CelebADataset(
        root_dir=config.data.data_path,
        split='val',
        image_size=config.data.image_size,
        download=False
    )
    test_ds = CelebADataset(
        root_dir=config.data.data_path,
        split='test',
        image_size=config.data.image_size,
        download=False
    )

    # Build mask generators
    train_mask_gen = MaskGenerator.for_train(config.mask)
    eval_mask_gen = MaskGenerator.for_eval(
        config.mask,
        cache_dir=config.mask.cache_dir
    )

    # Output directories
    base_out = "results/data_samples"
    os.makedirs(base_out, exist_ok=True)

    # Dump samples
    dump_samples(train_ds, train_mask_gen, os.path.join(base_out, "train"), deterministic=False, n=args.num_samples, device=str(device))
    dump_samples(val_ds, eval_mask_gen, os.path.join(base_out, "val"), deterministic=True, n=args.num_samples, device=str(device))
    dump_samples(test_ds, eval_mask_gen, os.path.join(base_out, "test"), deterministic=True, n=args.num_samples, device=str(device))

    print(f"Saved visual samples under {base_out}/{{train,val,test}}")


if __name__ == "__main__":
    main()