"""Checkpoint management for model training.

This module provides utilities for saving and loading model checkpoints,
including model state, optimizer state, scheduler state, and training metadata.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manager for saving and loading training checkpoints.

    Handles checkpoint saving with configurable frequency, keeping track of
    best models, and managing checkpoint directory structure.

    Args:
        checkpoint_dir: Directory to save checkpoints
        keep_last: Number of recent checkpoints to keep (default: 3)
        save_best: Whether to save best model separately (default: True)

    Example:
        >>> manager = CheckpointManager('./checkpoints', keep_last=3)
        >>> manager.save_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     loss=0.5,
        ...     metrics={'psnr': 25.0}
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        keep_last: int = 3,
        save_best: bool = True,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last: Number of recent checkpoints to keep
            save_best: Whether to track and save best model
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.save_best = save_best
        self.best_loss = float("inf")
        self.checkpoints = []

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """Save a training checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch number
            loss: Current loss value
            scheduler: Optional learning rate scheduler
            metrics: Optional dictionary of metrics
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint file
        """
        # Prepare checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "metrics": metrics or {},
        }

        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Track checkpoints for cleanup
        self.checkpoints.append(checkpoint_path)
        self._cleanup_old_checkpoints()

        # Save best model if applicable
        if self.save_best and (is_best or loss < self.best_loss):
            self.best_loss = loss
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path} (loss: {loss:.6f})")

        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        if len(self.checkpoints) > self.keep_last:
            # Sort by modification time
            self.checkpoints.sort(key=lambda p: p.stat().st_mtime)

            # Remove oldest checkpoints
            while len(self.checkpoints) > self.keep_last:
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint and restore model/optimizer state.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load checkpoint to

        Returns:
            Dictionary containing checkpoint metadata (epoch, loss, metrics)

        Example:
            >>> manager = CheckpointManager('./checkpoints')
            >>> metadata = manager.load_checkpoint(
            ...     'checkpoints/best_model.pt',
            ...     model=model,
            ...     optimizer=optimizer,
            ...     device=torch.device('cuda')
            ... )
            >>> print(f"Resuming from epoch {metadata['epoch']}")
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model state from {checkpoint_path}")

        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Loaded optimizer state")

        # Load scheduler state if provided
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Loaded scheduler state")

        # Return metadata
        metadata = {
            "epoch": checkpoint.get("epoch", 0),
            "loss": checkpoint.get("loss", float("inf")),
            "metrics": checkpoint.get("metrics", {}),
        }

        logger.info(
            f"Resumed from epoch {metadata['epoch']} with loss {metadata['loss']:.6f}"
        )

        return metadata

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the most recent checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None

        # Sort by epoch number
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
        return str(checkpoints[-1])

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to the best model checkpoint.

        Returns:
            Path to best checkpoint, or None if it doesn't exist
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        return str(best_path) if best_path.exists() else None


def save_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save a checkpoint to a specific file path.

    Convenience function for saving checkpoints without using CheckpointManager.

    Args:
        filepath: Path where checkpoint will be saved
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        scheduler: Optional learning rate scheduler
        metrics: Optional dictionary of metrics

    Example:
        >>> save_checkpoint(
        ...     'my_checkpoint.pt',
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     loss=0.5
        ... )
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics or {},
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint from a file path.

    Convenience function for loading checkpoints without using CheckpointManager.

    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint to

    Returns:
        Dictionary containing checkpoint metadata

    Example:
        >>> metadata = load_checkpoint(
        ...     'my_checkpoint.pt',
        ...     model=model,
        ...     optimizer=optimizer,
        ...     device=torch.device('cuda')
        ... )
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "metrics": checkpoint.get("metrics", {}),
    }
