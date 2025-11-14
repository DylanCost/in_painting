"""Flow Matching Inpainting Pipeline.

Self-contained pipeline for training and evaluating the flow matching model
on CelebA dataset. Conforms to the specification in docs/layout_and_outputs.md.

This pipeline:
- Loads and merges configuration from common_config.py
- Trains the flow matching model using existing Trainer
- Evaluates on test set with masked-region metrics
- Generates standardized outputs in runs/flowmatching/{timestamp}/

Usage:
    python -m flowmatching.pipeline [--options]

Example:
    python -m flowmatching.pipeline --epochs 100 --batch_size 32
"""

import os
import sys
import json
import logging
import argparse
import shutil
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# Project imports
from config.common_config import Config, DataConfig, MaskConfig, LoggingConfig
from flowmatching.models.unet import create_unet
from flowmatching.flow.sampler import ODESampler
from flowmatching.training.trainer import Trainer
from flowmatching.training.metrics import compute_metrics, denormalize_image
from flowmatching.data.dataset import CelebAInpainting


@dataclass
class FlowMatchingConfig:
    """Flow matching specific configuration."""

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    subsample_fraction: float = 1.0  # Fraction of training data to use (for debugging)

    # Validation parameters
    val_timesteps: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])

    # Sampling parameters
    num_sampling_steps: int = 100
    num_eval_samples: int = 1000
    num_example_images: int = 8

    # Optional metrics
    compute_lpips: bool = False

    # Device
    device: str = "cuda"
    seed: int = 42


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    common: Config = field(default_factory=Config)
    flowmatching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)


def setup_logging(run_dir: Path) -> logging.Logger:
    """Configure logging to file and console.

    Args:
        run_dir: Run directory for log file

    Returns:
        Configured logger
    """
    log_file = run_dir / "training.log"

    # Create logger
    logger = logging.getLogger("flowmatching_pipeline")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def create_run_directory() -> Tuple[Path, str]:
    """Create timestamped run directory.

    Returns:
        Tuple of (run_dir, timestamp)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs/flowmatching") / timestamp

    # Create subdirectories
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "examples").mkdir(parents=True, exist_ok=True)

    return run_dir, timestamp


def save_config_snapshot(config: PipelineConfig, run_dir: Path):
    """Save complete configuration snapshot.

    Args:
        config: Pipeline configuration
        run_dir: Run directory
    """
    config_dict = {
        "common": asdict(config.common),
        "flowmatching": asdict(config.flowmatching),
    }

    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def create_data_loaders(
    config: PipelineConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.

    Args:
        config: Pipeline configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Training dataset
    train_dataset = CelebAInpainting(
        root=config.common.data.data_path,
        split="train",
        image_size=config.common.data.image_size,
        mask_type=config.common.mask.type,
        min_mask_size=config.common.mask.min_size,
        max_mask_size=config.common.mask.max_size,
        mask_seed=config.common.mask.seed,
        cache_dir=config.common.mask.cache_dir,
    )

    # Validation dataset
    val_dataset = CelebAInpainting(
        root=config.common.data.data_path,
        split="valid",
        image_size=config.common.data.image_size,
        mask_type=config.common.mask.type,
        min_mask_size=config.common.mask.min_size,
        max_mask_size=config.common.mask.max_size,
        mask_seed=config.common.mask.seed,
        cache_dir=config.common.mask.cache_dir,
    )

    # Test dataset
    test_dataset = CelebAInpainting(
        root=config.common.data.data_path,
        split="test",
        image_size=config.common.data.image_size,
        mask_type=config.common.mask.type,
        min_mask_size=config.common.mask.min_size,
        max_mask_size=config.common.mask.max_size,
        mask_seed=config.common.mask.seed,
        cache_dir=config.common.mask.cache_dir,
    )

    # Create data loaders
    train_loader = train_dataset.get_dataloader(
        batch_size=config.flowmatching.batch_size,
        shuffle=True,
        num_workers=config.common.data.num_workers,
        subsample_fraction=config.flowmatching.subsample_fraction,
    )

    val_loader = val_dataset.get_dataloader(
        batch_size=config.flowmatching.batch_size,
        shuffle=False,
        num_workers=config.common.data.num_workers,
    )

    test_loader = test_dataset.get_dataloader(
        batch_size=config.flowmatching.batch_size,
        shuffle=False,
        num_workers=config.common.data.num_workers,
    )

    return train_loader, val_loader, test_loader


def train_model(
    config: PipelineConfig,
    run_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[str, Dict[str, List[float]]]:
    """Train the flow matching model.

    Args:
        config: Pipeline configuration
        run_dir: Run directory
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Training device
        logger: Logger instance

    Returns:
        Tuple of (best_checkpoint_path, training_history)
    """
    logger.info("Initializing model...")

    # Create model using common config
    model = create_unet(
        image_size=config.common.data.image_size,
        in_channels=4,  # RGB + mask
        out_channels=3,  # RGB velocity
        base_channels=config.common.unet.hidden_dims[0],
        time_embed_dim=256,  # Flow matching specific parameter
    )

    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    logger.info(f"Model size: {model.get_model_size_mb():.2f} MB")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.flowmatching.learning_rate,
        weight_decay=config.flowmatching.weight_decay,
    )

    # Create scheduler
    total_steps = len(train_loader) * config.flowmatching.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - config.flowmatching.warmup_steps,
        eta_min=config.flowmatching.min_lr,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=str(run_dir / "checkpoints"),
        log_dir=str(run_dir / "logs"),
        save_every=config.common.logging.save_interval,
        val_timesteps=config.flowmatching.val_timesteps,
        gradient_clip=1.0,
        warmup_steps=config.flowmatching.warmup_steps,
    )

    # Train and get history
    logger.info(f"Starting training for {config.flowmatching.epochs} epochs...")
    history = trainer.train(num_epochs=config.flowmatching.epochs)

    # Get best checkpoint
    best_checkpoint = trainer.checkpoint_manager.get_best_checkpoint()

    return best_checkpoint, history


def plot_learning_curves(history: Dict[str, List[float]], run_dir: Path):
    """Generate learning curves plot.

    Args:
        history: Training history dictionary
        run_dir: Run directory
    """
    if not history["train_loss"]:
        # No history available, skip plotting
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Training loss
    if history["train_loss"]:
        axes[0, 0].plot(history["train_loss"])
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)

    # Validation loss
    if history["val_loss"]:
        axes[0, 1].plot(history["val_loss"])
        axes[0, 1].set_title("Validation Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True)

    # Validation PSNR
    if history["val_psnr"]:
        axes[1, 0].plot(history["val_psnr"])
        axes[1, 0].set_title("Validation PSNR")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("PSNR (dB)")
        axes[1, 0].grid(True)

    # Validation SSIM
    if history["val_ssim"]:
        axes[1, 1].plot(history["val_ssim"])
        axes[1, 1].set_title("Validation SSIM")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("SSIM")
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(run_dir / "learning_curves.png", dpi=150)
    plt.close()


def save_history_csv(history: Dict[str, List[float]], run_dir: Path):
    """Save per-epoch metrics to CSV.

    Args:
        history: Training history dictionary
        run_dir: Run directory
    """
    if not history["train_loss"]:
        return

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "learning_rate"]
        )

        # Data rows
        num_epochs = len(history["train_loss"])
        for epoch in range(num_epochs):
            writer.writerow(
                [
                    epoch + 1,
                    (
                        history["train_loss"][epoch]
                        if epoch < len(history["train_loss"])
                        else ""
                    ),
                    (
                        history["val_loss"][epoch]
                        if epoch < len(history["val_loss"])
                        else ""
                    ),
                    (
                        history["val_psnr"][epoch]
                        if epoch < len(history["val_psnr"])
                        else ""
                    ),
                    (
                        history["val_ssim"][epoch]
                        if epoch < len(history["val_ssim"])
                        else ""
                    ),
                    (
                        history["learning_rate"][epoch]
                        if epoch < len(history["learning_rate"])
                        else ""
                    ),
                ]
            )


def organize_checkpoints(best_checkpoint: str, run_dir: Path, logger: logging.Logger):
    """Rename checkpoints to match specification.

    Args:
        best_checkpoint: Path to best checkpoint
        run_dir: Run directory
        logger: Logger instance
    """
    checkpoints_dir = run_dir / "checkpoints"

    # Copy best model as best.ckpt
    if best_checkpoint and os.path.exists(best_checkpoint):
        shutil.copy(best_checkpoint, checkpoints_dir / "best.ckpt")
        logger.info(f"Copied best checkpoint to best.ckpt")

    # Find latest checkpoint and copy as last.ckpt
    checkpoints = list(checkpoints_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoints:
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
        latest = checkpoints[-1]
        shutil.copy(latest, checkpoints_dir / "last.ckpt")
        logger.info(f"Copied latest checkpoint to last.ckpt")


def load_model_from_checkpoint(
    checkpoint_path: str, config: PipelineConfig, device: torch.device
) -> nn.Module:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        config: Pipeline configuration
        device: Device to load model on

    Returns:
        Loaded model
    """
    # Create model using common config
    model = create_unet(
        image_size=config.common.data.image_size,
        in_channels=4,
        out_channels=3,
        base_channels=config.common.unet.hidden_dims[0],
        time_embed_dim=256,  # Flow matching specific parameter
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    config: PipelineConfig,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Evaluate model on test set with masked-region metrics.

    Args:
        model: Trained model
        test_loader: Test data loader
        config: Pipeline configuration
        device: Device
        logger: Logger instance

    Returns:
        Dictionary with evaluation results
    """
    model.eval()

    # Create sampler
    sampler = ODESampler(
        model=model,
        num_steps=config.flowmatching.num_sampling_steps,
        preserve_observed=True,
        device=device,
        show_progress=False,
    )

    # Optional LPIPS metric
    lpips_metric = None
    if config.flowmatching.compute_lpips:
        try:
            from evaluation.metrics import InpaintingMetrics

            lpips_metric = InpaintingMetrics(device=str(device))
            logger.info("LPIPS metric enabled")
        except ImportError:
            logger.warning("LPIPS not available. Install with: pip install lpips")

    all_psnr = []
    all_ssim = []
    all_lpips = []

    num_batches = (
        config.flowmatching.num_eval_samples + config.flowmatching.batch_size - 1
    ) // config.flowmatching.batch_size

    logger.info(f"Evaluating on {config.flowmatching.num_eval_samples} samples...")

    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(test_loader, desc="Evaluation", total=num_batches)
        ):
            if i >= num_batches:
                break

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Sample inpainted images
            inpainted = sampler.sample(images, masks)

            # Compute masked-region metrics
            metrics = compute_metrics(
                pred=inpainted,
                target=images,
                mask=masks,  # Only compute on masked regions
                max_val=2.0,  # Images in [-1, 1]
                data_range=2.0,
            )

            all_psnr.append(metrics["psnr"])
            all_ssim.append(metrics["ssim"])

            # Optional LPIPS
            if lpips_metric is not None:
                # Denormalize for LPIPS
                inpainted_denorm = denormalize_image(inpainted)
                images_denorm = denormalize_image(images)
                lpips_val = lpips_metric.lpips_distance(inpainted_denorm, images_denorm)
                all_lpips.append(lpips_val)

    # Compute averages
    results = {
        "evaluated_count": len(all_psnr) * config.flowmatching.batch_size,
        "metrics_masked": {
            "psnr": float(np.mean(all_psnr)),
            "ssim": float(np.mean(all_ssim)),
        },
    }

    if all_lpips:
        results["metrics_masked"]["lpips"] = float(np.mean(all_lpips))

    return results


def save_eval_results(results: Dict[str, Any], run_dir: Path, timestamp: str):
    """Save evaluation results in schema v1 format.

    Args:
        results: Evaluation results
        run_dir: Run directory
        timestamp: Run timestamp
    """
    eval_results = {
        "timestamp": datetime.now().isoformat(),
        "evaluated_count": results["evaluated_count"],
        "metrics_masked": results["metrics_masked"],
    }

    with open(run_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)


def generate_triptychs(
    model: nn.Module,
    test_loader: DataLoader,
    config: PipelineConfig,
    run_dir: Path,
    device: torch.device,
    logger: logging.Logger,
):
    """Generate example triptychs: masked input | prediction | ground truth.

    Args:
        model: Trained model
        test_loader: Test data loader
        config: Pipeline configuration
        run_dir: Run directory
        device: Device
        logger: Logger instance
    """
    model.eval()

    # Create sampler
    sampler = ODESampler(
        model=model,
        num_steps=config.flowmatching.num_sampling_steps,
        preserve_observed=True,
        device=device,
        show_progress=False,
    )

    examples_dir = run_dir / "examples"
    all_triptychs = []

    logger.info(
        f"Generating {config.flowmatching.num_example_images} example triptychs..."
    )

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= config.flowmatching.num_example_images:
                break

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Generate inpainted image
            inpainted = sampler.sample(images, masks)

            # Create masked input (for visualization)
            masked_input = images * (1 - masks) + masks * torch.randn_like(images)

            # Denormalize for visualization
            masked_input = denormalize_image(masked_input)
            inpainted = denormalize_image(inpainted)
            ground_truth = denormalize_image(images)

            # Create triptych for first image in batch
            triptych = torch.cat(
                [masked_input[0], inpainted[0], ground_truth[0]], dim=2
            )  # Concatenate horizontally

            # Clamp to [0, 1]
            triptych = torch.clamp(triptych, 0, 1)

            # Save individual triptych
            save_image(triptych, examples_dir / f"triptych_{i+1:04d}.png")

            all_triptychs.append(triptych)

    # Create grid of all examples
    if all_triptychs:
        grid = make_grid(torch.stack(all_triptychs), nrow=2, padding=10, pad_value=1.0)
        save_image(grid, examples_dir / "examples_grid.png")

    logger.info(f"Saved {len(all_triptychs)} triptychs to {examples_dir}")


def parse_args(config: PipelineConfig) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        config: Pipeline configuration with default values

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Flow Matching Inpainting Pipeline")

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default=config.common.data.data_path,
        help="Root directory for dataset",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config.common.data.image_size,
        help="Image size",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.flowmatching.epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.flowmatching.batch_size,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.flowmatching.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--subsample_fraction",
        type=float,
        default=config.flowmatching.subsample_fraction,
        help="Fraction of training data to use (for debugging)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=config.flowmatching.num_eval_samples,
        help="Number of test samples to evaluate",
    )
    parser.add_argument(
        "--num_example_images",
        type=int,
        default=config.flowmatching.num_example_images,
        help="Number of example triptychs to generate",
    )
    parser.add_argument(
        "--compute_lpips",
        action="store_true",
        default=config.flowmatching.compute_lpips,
        help="Compute LPIPS metric (requires lpips package)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=config.flowmatching.device,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.flowmatching.seed,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    """Main pipeline execution."""

    # Print header
    print("=" * 80)
    print("Flow Matching Inpainting Pipeline")
    print("=" * 80)

    # Create configuration
    config = PipelineConfig()

    # Parse arguments with config defaults
    args = parse_args(config)

    # Update config with parsed arguments
    config.common.data.data_path = args.data_root
    config.common.data.image_size = args.image_size
    config.flowmatching.epochs = args.epochs
    config.flowmatching.batch_size = args.batch_size
    config.flowmatching.learning_rate = args.lr
    config.flowmatching.num_eval_samples = args.num_eval_samples
    config.flowmatching.num_example_images = args.num_example_images
    config.flowmatching.compute_lpips = args.compute_lpips
    config.flowmatching.device = args.device
    config.flowmatching.seed = args.seed
    config.flowmatching.subsample_fraction = args.subsample_fraction

    # Set random seed
    torch.manual_seed(config.flowmatching.seed)
    np.random.seed(config.flowmatching.seed)

    # Create run directory
    run_dir, timestamp = create_run_directory()
    print(f"Run directory: {run_dir}")

    # Setup logging
    logger = setup_logging(run_dir)
    logger.info("Starting flow matching pipeline")
    logger.info(f"Run directory: {run_dir}")

    # Save configuration
    save_config_snapshot(config, run_dir)
    logger.info("Configuration saved")

    # Set device
    device = torch.device(
        config.flowmatching.device if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    if config.flowmatching.subsample_fraction < 1.0:
        logger.info(
            f"  (subsampled to {config.flowmatching.subsample_fraction:.1%} of full dataset)"
        )
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Train model
    logger.info("=" * 80)
    logger.info("TRAINING PHASE")
    logger.info("=" * 80)

    best_checkpoint, history = train_model(
        config, run_dir, train_loader, val_loader, device, logger
    )

    # Generate learning curves
    logger.info("Generating learning curves...")
    plot_learning_curves(history, run_dir)
    save_history_csv(history, run_dir)

    # Organize checkpoints
    logger.info("Organizing checkpoints...")
    organize_checkpoints(best_checkpoint, run_dir, logger)

    # Load best model for evaluation
    logger.info("=" * 80)
    logger.info("EVALUATION PHASE")
    logger.info("=" * 80)

    logger.info("Loading best model...")
    model = load_model_from_checkpoint(best_checkpoint, config, device)

    # Evaluate on test set
    eval_results = evaluate_model(model, test_loader, config, device, logger)
    save_eval_results(eval_results, run_dir, timestamp)

    logger.info(f"Evaluation Results:")
    logger.info(f"  Evaluated samples: {eval_results['evaluated_count']}")
    logger.info(f"  PSNR: {eval_results['metrics_masked']['psnr']:.2f} dB")
    logger.info(f"  SSIM: {eval_results['metrics_masked']['ssim']:.4f}")
    if "lpips" in eval_results["metrics_masked"]:
        logger.info(f"  LPIPS: {eval_results['metrics_masked']['lpips']:.4f}")

    # Generate example triptychs
    logger.info("Generating example triptychs...")
    generate_triptychs(model, test_loader, config, run_dir, device, logger)

    # Summary
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {run_dir}")
    logger.info(f"  - eval_results.json")
    logger.info(f"  - config_snapshot.json")
    logger.info(f"  - training.log")
    logger.info(f"  - learning_curves.png")
    logger.info(f"  - checkpoints/best.ckpt")
    logger.info(f"  - checkpoints/last.ckpt")
    logger.info(f"  - examples/ ({config.flowmatching.num_example_images} triptychs)")

    print("=" * 80)
    print(f"✓ Pipeline completed successfully!")
    print(f"✓ Results: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
