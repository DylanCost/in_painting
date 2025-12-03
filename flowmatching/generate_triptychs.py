#!/usr/bin/env python
"""Generate reproducible triptych figures for flow-matching inpainting models.

This utility loads the first N images from the CelebA test split, applies manual
masks drawn from :mod:`config.common_config`, runs reconstruction with a trained
flow-matching checkpoint, and exports two marginless canvases (4 columns ×
3 rows) suitable for paper-ready qualitative figures.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torchvision.utils import save_image

from config.common_config import (
    ManualMaskSpec,
    TRIPTYCH_MASK_VERSION,
    get_triptych_mask_specs,
)
from flowmatching.data import CelebAInpainting
from flowmatching.flow.sampler import ODESampler
from flowmatching.models import create_unet
from flowmatching.pipeline import PipelineConfig
from flowmatching.training.metrics import denormalize_image

LOGGER = logging.getLogger("flowmatching.generate_triptychs")
PANEL_COLUMNS = 4
PANEL_ROWS = 3
IMAGE_SIZE = 128
DEFAULT_NUM_EXAMPLES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible triptych canvases for qualitative inspection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/triptychs",
        help="Directory where canvases and metadata will be stored",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help="Number of deterministic test samples to render (must be multiple of 4)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional explicit checkpoint path; otherwise the latest best/last is used",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device override (auto-detects CUDA if available)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./assets/datasets",
        help="Root directory containing CelebA data",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of ODE integration steps for the sampler",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for torch/numpy to keep noise generation deterministic",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download CelebA if it is missing",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a tqdm progress bar during sampling",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(user_device: Optional[str]) -> torch.device:
    if user_device:
        return torch.device(user_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint_path(explicit_path: Optional[str]) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
        return path

    runs_dir = Path("runs/flowmatching")
    if not runs_dir.exists():
        raise FileNotFoundError("runs/flowmatching does not exist and no checkpoint was provided")

    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], reverse=True)
    for ckpt_name in ("best.ckpt", "last.ckpt"):
        for run_dir in run_dirs:
            candidate = run_dir / "checkpoints" / ckpt_name
            if candidate.exists():
                return candidate
    raise FileNotFoundError("No best.ckpt or last.ckpt found under runs/flowmatching")


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    pipeline_config = PipelineConfig()
    hidden_dims = list(pipeline_config.common.unet.hidden_dims)
    image_size = pipeline_config.common.data.image_size

    model = create_unet(
        hidden_dims=hidden_dims,
        image_size=image_size,
        in_channels=4,
        out_channels=3,
        time_embed_dim=256,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def manual_mask_tensor(spec: ManualMaskSpec, image_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, image_size, image_size, dtype=torch.float32)
    bottom = min(spec.bottom, image_size)
    right = min(spec.right, image_size)
    mask[:, spec.top:bottom, spec.left:right] = 1.0
    return mask.unsqueeze(0).to(device)


def prepare_triptych(original: torch.Tensor, masked: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    original_vis = denormalize_image(original).squeeze(0).cpu()
    masked_vis = denormalize_image(masked).squeeze(0).cpu()
    recon_vis = denormalize_image(recon).squeeze(0).cpu()
    return torch.cat([original_vis, masked_vis, recon_vis], dim=1)


def save_canvases(triptychs: Sequence[torch.Tensor], output_dir: Path) -> List[Path]:
    if len(triptychs) % PANEL_COLUMNS != 0:
        raise ValueError(
            f"Triptych count ({len(triptychs)}) must be divisible by {PANEL_COLUMNS} to fill panels"
        )
    panel_paths: List[Path] = []
    for chunk_idx in range(0, len(triptychs), PANEL_COLUMNS):
        chunk = triptychs[chunk_idx : chunk_idx + PANEL_COLUMNS]
        if len(chunk) != PANEL_COLUMNS:
            raise ValueError("Incomplete panel chunk encountered; ensure num_examples is a multiple of 4")
        canvas = torch.cat(list(chunk), dim=2)
        panel_id = chunk_idx // PANEL_COLUMNS + 1
        panel_path = output_dir / f"triptych_panel_{panel_id:02d}.png"
        save_image(canvas, panel_path)
        panel_paths.append(panel_path)
    return panel_paths


def main() -> None:
    args = parse_args()

    if args.num_examples % PANEL_COLUMNS != 0:
        raise ValueError("num-examples must be divisible by 4 so each panel has exactly four columns")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = resolve_device(args.device)
    LOGGER.info("Using device %s", device)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    LOGGER.info("Loading checkpoint from %s", checkpoint_path)
    model = load_model(checkpoint_path, device)
    sampler = ODESampler(
        model=model,
        num_steps=args.num_steps,
        preserve_observed=True,
        device=device,
        show_progress=args.progress,
    )

    dataset = CelebAInpainting(
        root=args.data_root,
        split="test",
        image_size=IMAGE_SIZE,
        download=args.download,
        normalize=True,
        mask_type="random",
    )

    available_specs = get_triptych_mask_specs(list(range(DEFAULT_NUM_EXAMPLES)))
    if args.num_examples > len(available_specs):
        raise ValueError(
            f"Requested {args.num_examples} examples but only {len(available_specs)} manual masks are defined"
        )
    specs = available_specs[: args.num_examples]

    LOGGER.info("Generating %d triptychs", len(specs))
    triptych_tensors: List[torch.Tensor] = []
    metadata_samples: List[Dict[str, object]] = []

    for spec in specs:
        sample = dataset[spec.index]
        image = sample["image"].unsqueeze(0).to(device)
        mask = manual_mask_tensor(spec, IMAGE_SIZE, device)
        masked_input = (1 - mask) * image
        with torch.no_grad():
            recon = sampler.sample(image, mask)
        triptych = prepare_triptych(image, masked_input, recon)
        triptych_tensors.append(triptych)
        metadata_samples.append(
            {
                "local_index": sample.get("local_idx"),
                "global_index": sample.get("global_idx"),
                "filename": sample.get("filename"),
                "mask": asdict(spec),
            }
        )

    panel_paths = save_canvases(triptych_tensors, output_dir)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "num_examples": len(triptych_tensors),
        "panel_dimensions_px": {
            "width": PANEL_COLUMNS * IMAGE_SIZE,
            "height": PANEL_ROWS * IMAGE_SIZE,
        },
        "mask_version": TRIPTYCH_MASK_VERSION,
        "samples": metadata_samples,
        "panels": [str(path) for path in panel_paths],
    }

    metadata_path = output_dir / "triptych_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    LOGGER.info("Saved panels: %s", ", ".join(path.name for path in panel_paths))
    LOGGER.info("Metadata written to %s", metadata_path)


if __name__ == "__main__":
    main()
