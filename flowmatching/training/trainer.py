"""Main training loop for flow matching image inpainting.

This module implements the Trainer class that handles the complete training
pipeline including training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Optional, Dict, List
import time

from ..flow.flow_matching import FlowMatching
from .metrics import compute_metrics
from .checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for flow matching image inpainting.
    
    Handles the complete training pipeline including:
    - Training loop with progress tracking
    - Validation with multiple timesteps
    - Checkpointing and model saving
    - TensorBoard logging
    - Learning rate scheduling with warmup
    - Gradient clipping
    - Resume from checkpoint
    
    Args:
        model: U-Net model for velocity prediction
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer (e.g., AdamW)
        scheduler: Learning rate scheduler (e.g., CosineAnnealingLR)
        device: Device to train on
        checkpoint_dir: Directory for saving checkpoints
        log_dir: Directory for TensorBoard logs
        save_every: Save checkpoint every N epochs
        val_timesteps: List of timesteps to validate on
        gradient_clip: Maximum gradient norm (None to disable)
        warmup_steps: Number of warmup steps for scheduler
    
    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     device=device
        ... )
        >>> trainer.train(num_epochs=100)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs',
        save_every: int = 5,
        val_timesteps: List[float] = None,
        gradient_clip: Optional[float] = 1.0,
        warmup_steps: int = 1000
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Training device
            checkpoint_dir: Checkpoint directory
            log_dir: TensorBoard log directory
            save_every: Checkpoint frequency
            val_timesteps: Timesteps for validation
            gradient_clip: Gradient clipping value
            warmup_steps: Warmup steps
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_every = save_every
        self.val_timesteps = val_timesteps or [0.25, 0.5, 0.75]
        self.gradient_clip = gradient_clip
        self.warmup_steps = warmup_steps
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize flow matching
        self.flow_matching = FlowMatching()
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            keep_last=3,
            save_best=True
        )
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary with training metrics (loss, etc.)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Prepare training batch (get x_t, t, v_gt, x_0)
            # x_0 has noise in masked regions, original pixels in unmasked regions
            x_t, t, v_gt, x0 = self.flow_matching.prepare_training_batch(
                images, masks
            )
            
            # Create model input (concatenate x_t with mask)
            model_input = torch.cat([x_t, masks], dim=1)  # [B, 4, H, W]
            
            # Forward pass
            v_pred = self.model(model_input, t)
            
            # Compute loss (only on masked regions)
            loss = self.flow_matching.compute_loss(v_pred, v_gt, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step (with warmup)
            if self.scheduler is not None:
                if self.global_step < self.warmup_steps:
                    # Linear warmup
                    lr_scale = (self.global_step + 1) / self.warmup_steps
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['initial_lr'] * lr_scale
                else:
                    self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / num_batches,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to TensorBoard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Run validation on multiple timesteps.
        
        Returns:
            Dictionary with validation metrics (loss, PSNR, SSIM)
        """
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Validate on multiple timesteps
                batch_loss = 0.0
                batch_psnr = 0.0
                batch_ssim = 0.0
                
                for t_val in self.val_timesteps:
                    # Create timestep tensor
                    t = torch.full((images.shape[0],), t_val, device=self.device)
                    
                    # Prepare batch
                    # x_0 has noise in masked regions, original pixels in unmasked regions
                    x_t, _, v_gt, x0 = self.flow_matching.prepare_training_batch(
                        images, masks, t
                    )
                    
                    # Forward pass
                    model_input = torch.cat([x_t, masks], dim=1)
                    v_pred = self.model(model_input, t)
                    
                    # Compute loss
                    loss = self.flow_matching.compute_loss(v_pred, v_gt, masks)
                    batch_loss += loss.item()
                    
                    # Compute metrics on predicted image
                    # x_1 = x_t + v_pred * (1 - t) (approximate)
                    pred_image = x_t + v_pred * (1 - t.view(-1, 1, 1, 1))
                    
                    # Compute PSNR and SSIM on masked regions
                    metrics = compute_metrics(
                        pred_image,
                        images,
                        mask=masks,
                        max_val=2.0,  # Images are normalized to [-1, 1]
                        data_range=2.0
                    )
                    
                    batch_psnr += metrics['psnr']
                    batch_ssim += metrics['ssim']
                
                # Average over timesteps
                batch_loss /= len(self.val_timesteps)
                batch_psnr /= len(self.val_timesteps)
                batch_ssim /= len(self.val_timesteps)
                
                total_loss += batch_loss
                total_psnr += batch_psnr
                total_ssim += batch_ssim
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': batch_loss,
                    'psnr': batch_psnr,
                    'ssim': batch_ssim
                })
        
        # Average over all batches
        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        
        return {
            'loss': avg_loss,
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
    
    def train(
        self,
        num_epochs: int,
        resume_from: Optional[str] = None
    ) -> None:
        """Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Optional path to checkpoint to resume from
        
        Example:
            >>> trainer.train(num_epochs=100)
            >>> # Or resume from checkpoint
            >>> trainer.train(num_epochs=100, resume_from='checkpoints/best_model.pt')
        """
        # Resume from checkpoint if provided
        if resume_from is not None:
            metadata = self.checkpoint_manager.load_checkpoint(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.device
            )
            self.current_epoch = metadata['epoch']
            self.best_val_loss = metadata['loss']
            logger.info(f"Resumed training from epoch {self.current_epoch}")
        
        # Store initial learning rate for warmup
        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']
        
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                
                # Train for one epoch
                train_metrics = self.train_epoch()
                logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.6f}")
                
                # Validate
                val_metrics = self.validate()
                logger.info(
                    f"Epoch {epoch}: Val Loss = {val_metrics['loss']:.6f}, "
                    f"PSNR = {val_metrics['psnr']:.2f} dB, "
                    f"SSIM = {val_metrics['ssim']:.4f}"
                )
                
                # Log to TensorBoard
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('val/psnr', val_metrics['psnr'], epoch)
                self.writer.add_scalar('val/ssim', val_metrics['ssim'], epoch)
                
                # Save checkpoint
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                
                if (epoch + 1) % self.save_every == 0 or is_best:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        loss=val_metrics['loss'],
                        scheduler=self.scheduler,
                        metrics=val_metrics,
                        is_best=is_best
                    )
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save checkpoint on interrupt
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.current_epoch,
                loss=self.best_val_loss,
                scheduler=self.scheduler,
                metrics={'interrupted': True}
            )
        
        finally:
            elapsed_time = time.time() - start_time
            logger.info(f"Training completed in {elapsed_time / 3600:.2f} hours")
            self.writer.close()
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save a checkpoint to a specific file.
        
        Args:
            filepath: Path to save checkpoint
        """
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            loss=self.best_val_loss,
            scheduler=self.scheduler
        )
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load a checkpoint from a file.
        
        Args:
            filepath: Path to checkpoint file
        """
        metadata = self.checkpoint_manager.load_checkpoint(
            filepath,
            self.model,
            self.optimizer,
            self.scheduler,
            self.device
        )
        self.current_epoch = metadata['epoch']
        self.best_val_loss = metadata['loss']