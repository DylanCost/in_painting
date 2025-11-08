"""Main training script for flow matching image inpainting.

This script provides a command-line interface for training the flow matching
model on the CelebA dataset with configurable hyperparameters.

Example usage:
    # Train with default settings
    python train.py

    # Train with custom settings
    python train.py --epochs 100 --batch_size 32 --lr 1e-4

    # Resume from checkpoint
    python train.py --resume checkpoints/checkpoint_epoch_50.pt

    # Train on CPU
    python train.py --device cpu
"""

import argparse
import logging

# Import from root-level data module
import sys
from pathlib import Path

import torch
import torch.nn as nn

from .models import create_unet
from .training.trainer import Trainer

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
from data.celeba_dataset import create_dataloaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train flow matching model for image inpainting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for AdamW optimizer",
    )

    # Data parameters
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for CelebA dataset",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Image size (will be resized to this)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of training data in train/val split",
    )

    # Model parameters
    parser.add_argument(
        "--base_channels", type=int, default=64, help="Base number of channels in U-Net"
    )
    parser.add_argument(
        "--time_emb_dim", type=int, default=256, help="Time embedding dimension"
    )

    # Scheduler parameters
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps for learning rate",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine annealing",
    )

    # Checkpointing parameters
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--save_every", type=int, default=5, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Validation parameters
    parser.add_argument(
        "--val_timesteps",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="Timesteps to use for validation",
    )

    # Training parameters
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (None to disable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np

    np.random.seed(seed)
    import random

    random.seed(seed)

    # Make cudnn deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Log configuration
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("=" * 80)

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Create model
    logger.info("Creating model...")
    model = create_unet(
        in_channels=4,  # RGB + mask
        out_channels=3,  # RGB velocity
        base_channels=args.base_channels,
        time_embed_dim=args.time_emb_dim,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Create scheduler
    logger.info("Creating learning rate scheduler...")
    # Calculate total steps for cosine annealing
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - args.warmup_steps, eta_min=args.min_lr
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_every=args.save_every,
        val_timesteps=args.val_timesteps,
        gradient_clip=args.gradient_clip,
        warmup_steps=args.warmup_steps,
    )

    # Train
    logger.info("Starting training...")
    logger.info("=" * 80)

    try:
        trainer.train(num_epochs=args.epochs, resume_from=args.resume)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")
    logger.info(f"TensorBoard logs saved to: {args.log_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
