import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import wandb
from tqdm import tqdm
import os
import csv
import math
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter
from masking.mask_generator import MaskGenerator
import torchvision.utils as vutils


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio only on masked regions.
    
    Args:
        pred: Predicted images (B, C, H, W) in range [-1, 1]
        target: Target images (B, C, H, W) in range [-1, 1]
        mask: Binary mask (B, 1, H, W) where 1 indicates regions to evaluate
        data_range: The difference between max and min values (2.0 for [-1, 1] range)
    
    Returns:
        Mean PSNR value across the batch for masked regions only
    """
    # Expand mask to match image channels if needed
    if mask.shape[1] == 1 and pred.shape[1] > 1:
        mask = mask.expand_as(pred)
    
    # Compute MSE only on masked regions
    diff = (pred - target) * mask
    mse_per_image = (diff ** 2).sum(dim=[1, 2, 3]) / (mask.sum(dim=[1, 2, 3]) + 1e-8)
    
    # Compute PSNR for each image
    psnr = 10 * torch.log10((data_range ** 2) / (mse_per_image + 1e-8))
    
    return psnr.mean()


def compute_ssim(
    pred: torch.Tensor, 
    target: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 11,
    data_range: float = 2.0,
    channel: int = 3
) -> torch.Tensor:
    """
    Compute Structural Similarity Index Measure only on masked regions.
    
    Args:
        pred: Predicted images (B, C, H, W) in range [-1, 1]
        target: Target images (B, C, H, W) in range [-1, 1]
        mask: Binary mask (B, 1, H, W) where 1 indicates regions to evaluate
        window_size: Size of the Gaussian window
        data_range: The difference between max and min values
        channel: Number of channels
    
    Returns:
        Mean SSIM value across the batch for masked regions only
    """
    # Create Gaussian window
    def gaussian_window(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g
    
    # Create 2D window
    sigma = 1.5
    _1d_window = gaussian_window(window_size, sigma)
    _2d_window = _1d_window.unsqueeze(1) @ _1d_window.unsqueeze(0)  # Outer product
    _2d_window = _2d_window.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, window_size, window_size)
    _2d_window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    window = _2d_window.to(pred.device, dtype=pred.dtype)
    
    # Expand mask to match channels
    if mask.shape[1] == 1:
        mask_expanded = mask.expand(-1, channel, -1, -1)
    else:
        mask_expanded = mask
    
    # Apply mask to inputs
    pred_masked = pred * mask_expanded
    target_masked = target * mask_expanded
    
    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Compute means (weighted by mask)
    mu1 = F.conv2d(pred_masked, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target_masked, window, padding=window_size // 2, groups=channel)
    
    # Normalize by mask area
    mask_sum = F.conv2d(mask_expanded, window, padding=window_size // 2, groups=channel) + 1e-8
    mu1 = mu1 / mask_sum
    mu2 = mu2 / mask_sum
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance (weighted by mask)
    sigma1_sq = F.conv2d(pred_masked ** 2, window, padding=window_size // 2, groups=channel) / mask_sum - mu1_sq
    sigma2_sq = F.conv2d(target_masked ** 2, window, padding=window_size // 2, groups=channel) / mask_sum - mu2_sq
    sigma12 = F.conv2d(pred_masked * target_masked, window, padding=window_size // 2, groups=channel) / mask_sum - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Apply mask to SSIM map and compute mean only over masked regions
    ssim_map_masked = ssim_map * mask_expanded
    return ssim_map_masked.sum() / (mask_expanded.sum() + 1e-8)


class MetricsTracker:
    """Track and compute image quality metrics for masked regions."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.mse_sum = 0.0
        self.mae_sum = 0.0
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.count = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """
        Update metrics with a new batch (computed only on masked regions).
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            mask: Binary mask for masked region metrics (B, 1, H, W)
        """
        batch_size = pred.shape[0]
        
        with torch.no_grad():
            # Expand mask to match channels if needed
            if mask.shape[1] == 1 and pred.shape[1] > 1:
                mask_expanded = mask.expand_as(pred)
            else:
                mask_expanded = mask
            
            # MSE on masked regions
            diff = (pred - target) * mask_expanded
            mse = (diff ** 2).sum() / (mask_expanded.sum() + 1e-8)
            self.mse_sum += mse.item() * batch_size
            
            # MAE on masked regions
            mae = (diff.abs()).sum() / (mask_expanded.sum() + 1e-8)
            self.mae_sum += mae.item() * batch_size
            
            # PSNR on masked regions
            psnr = compute_psnr(pred, target, mask)
            self.psnr_sum += psnr.item() * batch_size
            
            # SSIM on masked regions
            ssim = compute_ssim(pred, target, mask, channel=pred.shape[1])
            self.ssim_sum += ssim.item() * batch_size
            
            self.count += batch_size
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics."""
        if self.count == 0:
            return {'mse': 0.0, 'mae': 0.0, 'psnr': 0.0, 'ssim': 0.0}
        
        return {
            'mse': self.mse_sum / self.count,
            'mae': self.mae_sum / self.count,
            'psnr': self.psnr_sum / self.count,
            'ssim': self.ssim_sum / self.count
        }


class Trainer:
    """Enhanced trainer with pretrained model support and masked region quality metrics."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        config: Dict,
        mask_config: Dict,
        device: str = 'cuda',
        test_loader: DataLoader = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn.to(device)
        self.config = config
        self.mask_config = mask_config
        self.device = device

        # Mask generators
        self.train_mask_gen = MaskGenerator.for_train(self.mask_config)
        self.eval_mask_gen = MaskGenerator.for_eval(
            self.mask_config,
            cache_dir=self.mask_config.cache_dir if hasattr(self.mask_config, 'cache_dir') else './assets/masks'
        )
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker(device=device)
        
        # Setup optimizer with differential learning rates
        self.setup_optimizer()
        
        # Setup progressive unfreezing if configured
        self.setup_progressive_unfreezing()
        
        # Learning rate scheduler
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=1e-6
        )
        
        # Logging
        if config.logging.use_wandb:
            wandb.init(project="vae-inpainting", config=config.to_dict())
        
        if config.logging.use_tensorboard:
            self.writer = SummaryWriter(config.logging.log_dir)
        else:
            self.writer = None
        
        self.checkpoint_dir = config.logging.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # CSV metrics logging
        self.metrics_csv_path = os.path.join(self.checkpoint_dir, 'metrics.csv')
        self.csv_initialized = False
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_psnr = 0.0
        self.best_val_ssim = 0.0
    
    def setup_optimizer(self):
        """Setup optimizer with different LRs for encoder/decoder."""
        
        # Check if using differential learning rates
        if self.config.training.encoder_lr is not None:
            encoder_params = []
            decoder_params = []
            
            for name, param in self.model.named_parameters():
                if 'encoder' in name:
                    encoder_params.append(param)
                else:
                    decoder_params.append(param)
            
            self.optimizer = Adam([
                {'params': encoder_params, 'lr': self.config.training.encoder_lr},
                {'params': decoder_params, 'lr': self.config.training.decoder_lr}
            ], betas=(self.config.training.beta1, self.config.training.beta2))
        else:
            # Standard optimizer
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=(self.config.training.beta1, self.config.training.beta2)
            )
    
    def setup_progressive_unfreezing(self):
        """Setup schedule for progressive unfreezing."""
        self.unfreeze_schedule = self.config.training.unfreeze_schedule
    
    def maybe_unfreeze_layers(self, epoch: int):
        """Unfreeze layers according to schedule."""
        epoch_key = f'epoch_{epoch}'
        if epoch_key in self.unfreeze_schedule:
            stages_to_unfreeze = self.unfreeze_schedule[epoch_key]
            
            # Unfreeze encoder stages
            if hasattr(self.model.encoder, '_freeze_stages'):
                self.model.encoder._freeze_stages(stages_to_unfreeze)
                print(f"Unfroze encoder up to stage {stages_to_unfreeze}")
    
    def compute_batch_metrics(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, float]:
        """Compute MSE, MAE, PSNR, and SSIM for masked regions in a single batch."""
        with torch.no_grad():
            # Expand mask if needed
            if mask.shape[1] == 1 and pred.shape[1] > 1:
                mask_expanded = mask.expand_as(pred)
            else:
                mask_expanded = mask
            
            # MSE on masked regions
            diff = (pred - target) * mask_expanded
            mse = (diff ** 2).sum() / (mask_expanded.sum() + 1e-8)
            
            # MAE on masked regions
            mae = diff.abs().sum() / (mask_expanded.sum() + 1e-8)
            
            # PSNR and SSIM on masked regions
            psnr = compute_psnr(pred, target, mask)
            ssim = compute_ssim(pred, target, mask, channel=pred.shape[1])
        
        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'psnr': psnr.item(),
            'ssim': ssim.item()
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Modified train epoch with progressive unfreezing and masked region quality metrics."""
        
        # Check if we should unfreeze layers
        self.maybe_unfreeze_layers(self.current_epoch)
        
        self.model.train()
        epoch_losses = {'total': 0, 'reconstruction': 0, 'kl': 0, 'perceptual': 0}
        self.metrics_tracker.reset()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            image = batch['image'].to(self.device)
            B, C, H, W = image.shape
            mask = self.train_mask_gen.generate((B, 1, H, W)).to(self.device)
            masked_image = image * (1.0 - mask)

            # Forward pass
            outputs = self.model(masked_image, mask)
            
            # Calculate loss
            losses = self.loss_fn(
                outputs['reconstruction'],
                image,
                outputs['mu'],
                outputs['log_var'],
                mask
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            
            self.optimizer.step()
            
            # Update loss metrics
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Update quality metrics (for masked regions)
            self.metrics_tracker.update(outputs['reconstruction'], image, mask)
            batch_metrics = self.compute_batch_metrics(outputs['reconstruction'], image, mask)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'rec': losses['reconstruction'].item(),
                'psnr_mask': batch_metrics['psnr'],
                'ssim_mask': batch_metrics['ssim']
            })
            
            # Logging at batch level
            if self.global_step % self.config.logging.log_interval == 0:
                # Merge losses and metrics for logging
                all_metrics = {**losses, **batch_metrics}
                self.log_metrics(all_metrics, 'train_batch', step=self.global_step)
            
            # Sample generation
            if self.global_step % self.config.logging.sample_interval == 0:
                self.generate_samples(image, outputs, masked_image, mask)
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        # Add averaged quality metrics (for masked regions)
        quality_metrics = self.metrics_tracker.compute()
        epoch_losses.update(quality_metrics)
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model with masked region quality metrics."""
        self.model.eval()
        val_losses = {'total': 0, 'reconstruction': 0, 'kl': 0, 'perceptual': 0}
        self.metrics_tracker.reset()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                image = batch['image'].to(self.device)
                filenames = batch['filename']  # list[str]
                B, C, H, W = image.shape
                mask = self.eval_mask_gen.generate_for_filenames(
                    filenames=filenames,
                    shape=(1, H, W)
                ).to(self.device)
                masked_image = image * (1.0 - mask)

                # Forward pass
                outputs = self.model(masked_image, mask)
                
                # Calculate loss
                losses = self.loss_fn(
                    outputs['reconstruction'],
                    image,
                    outputs['mu'],
                    outputs['log_var'],
                    mask
                )
                
                # Update loss metrics
                for key in val_losses:
                    if key in losses:
                        val_losses[key] += losses[key].item()
                
                # Update quality metrics (for masked regions)
                self.metrics_tracker.update(outputs['reconstruction'], image, mask)
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)
        
        # Add averaged quality metrics (for masked regions)
        quality_metrics = self.metrics_tracker.compute()
        val_losses.update(quality_metrics)
        
        return val_losses
    
    def test_model(self) -> Dict[str, float]:
        """Evaluate on test set with masked region quality metrics."""
        if self.test_loader is None:
            return {}
        
        self.model.eval()
        test_losses = {'total': 0, 'reconstruction': 0, 'kl': 0, 'perceptual': 0}
        self.metrics_tracker.reset()
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                image = batch['image'].to(self.device)
                filenames = batch['filename']  # list[str]
                B, C, H, W = image.shape
                mask = self.eval_mask_gen.generate_for_filenames(
                    filenames=filenames,
                    shape=(1, H, W)
                ).to(self.device)
                masked_image = image * (1.0 - mask)

                outputs = self.model(masked_image, mask)
                losses = self.loss_fn(
                    outputs['reconstruction'],
                    image,
                    outputs['mu'],
                    outputs['log_var'],
                    mask
                )
                
                for key in test_losses:
                    if key in losses:
                        test_losses[key] += losses[key].item()
                
                # Update quality metrics (for masked regions)
                self.metrics_tracker.update(outputs['reconstruction'], image, mask)
        
        # Average losses
        for key in test_losses:
            test_losses[key] /= len(self.test_loader)
        
        # Add averaged quality metrics (for masked regions)
        quality_metrics = self.metrics_tracker.compute()
        test_losses.update(quality_metrics)
        
        return test_losses
    
    def save_metrics_to_csv(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Optional[Dict[str, float]] = None,
        learning_rate: float = None
    ):
        """Save epoch metrics to CSV file."""
        # Build row data
        row_data = {'epoch': epoch}
        
        # Add learning rate
        if learning_rate is not None:
            row_data['learning_rate'] = learning_rate
        
        # Add train metrics with prefix
        for key, value in train_metrics.items():
            if torch.is_tensor(value):
                value = value.item()
            row_data[f'train_{key}'] = value
        
        # Add val metrics with prefix
        for key, value in val_metrics.items():
            if torch.is_tensor(value):
                value = value.item()
            row_data[f'val_{key}'] = value
        
        # Add test metrics with prefix (if provided)
        if test_metrics is not None:
            for key, value in test_metrics.items():
                if torch.is_tensor(value):
                    value = value.item()
                row_data[f'test_{key}'] = value
        
        # Write to CSV
        file_exists = os.path.exists(self.metrics_csv_path)
        
        with open(self.metrics_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            
            # Write header only if file is new or empty
            if not file_exists or not self.csv_initialized:
                writer.writeheader()
                self.csv_initialized = True
            
            writer.writerow(row_data)
    
    def train(self):
        """Main training loop with masked region evaluation."""
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Log epoch-level metrics
            self.log_metrics(train_losses, 'train_epoch', step=epoch)
            self.log_metrics(val_losses, 'val_epoch', step=epoch)
            
            # Test every N epochs
            test_losses = None
            if epoch % 10 == 0 and self.test_loader is not None:
                test_losses = self.test_model()
                print(f"Test Loss: {test_losses['total']:.4f} | "
                      f"Masked PSNR: {test_losses['psnr']:.2f} | Masked SSIM: {test_losses['ssim']:.4f}")
                self.log_metrics(test_losses, 'test_epoch', step=epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Log learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            self.log_metrics({'learning_rate': current_lr}, 'train_epoch', step=epoch)
            
            # Save metrics to CSV
            self.save_metrics_to_csv(
                epoch=epoch,
                train_metrics=train_losses,
                val_metrics=val_losses,
                test_metrics=test_losses,
                learning_rate=current_lr
            )
            
            # Logging (now showing masked region metrics)
            print(f"\nEpoch {epoch}/{self.config.training.epochs}")
            print(f"Train Loss: {train_losses['total']:.4f} | "
                  f"Masked MSE: {train_losses['mse']:.4f} | Masked MAE: {train_losses['mae']:.4f} | "
                  f"Masked PSNR: {train_losses['psnr']:.2f} | Masked SSIM: {train_losses['ssim']:.4f}")
            print(f"Val Loss: {val_losses['total']:.4f} | "
                  f"Masked MSE: {val_losses['mse']:.4f} | Masked MAE: {val_losses['mae']:.4f} | "
                  f"Masked PSNR: {val_losses['psnr']:.2f} | Masked SSIM: {val_losses['ssim']:.4f}")
            
            # Save checkpoint
            if epoch % self.config.logging.save_interval == 0:
                self.save_checkpoint(epoch, val_losses['total'])
            
            # Save best model (based on loss)
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint(epoch, val_losses['total'], is_best=True, suffix='best_loss')
            
            # Save best model based on masked PSNR
            if val_losses['psnr'] > self.best_val_psnr:
                self.best_val_psnr = val_losses['psnr']
                self.save_checkpoint(epoch, val_losses['total'], is_best=True, suffix='best_psnr')
            
            # Save best model based on masked SSIM
            if val_losses['ssim'] > self.best_val_ssim:
                self.best_val_ssim = val_losses['ssim']
                self.save_checkpoint(epoch, val_losses['total'], is_best=True, suffix='best_ssim')
    
    def save_checkpoint(
        self, 
        epoch: int, 
        val_loss: float, 
        is_best: bool = False,
        suffix: str = 'best_model'
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'best_val_psnr': self.best_val_psnr,
            'best_val_ssim': self.best_val_ssim,
            'config': self.config,
            'global_step': self.global_step
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, f'{suffix}.pt')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_psnr = checkpoint.get('best_val_psnr', 0.0)
        self.best_val_ssim = checkpoint.get('best_val_ssim', 0.0)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str, step: int = None):
        """Log metrics to wandb and tensorboard."""
        # Use provided step or default to global_step
        log_step = step if step is not None else self.global_step
        
        # Handle both tensor and float values
        processed_metrics = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                processed_metrics[key] = value.item()
            else:
                processed_metrics[key] = value
        
        if self.config.logging.use_wandb:
            wandb.log({f"{prefix}/{k}": v for k, v in processed_metrics.items()}, step=log_step)
        
        if self.writer is not None:
            for key, value in processed_metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}", value, log_step)
    
    def generate_samples(
        self, 
        image: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        masked_image: torch.Tensor,
        mask: torch.Tensor
    ):
        """Generate sample reconstructions for visualization with masked region metrics."""
        with torch.no_grad():
            # Ensure all tensors are on the same device
            device = self.device
            
            # Take first 8 samples
            n_samples = min(8, image.size(0))
            
            # Move all tensors to device and select samples
            original = image[:n_samples].to(device)
            masked = masked_image[:n_samples].to(device)
            reconstruction = outputs['reconstruction'][:n_samples].to(device)
            mask_viz = mask[:n_samples].repeat(1, 3, 1, 1).to(device)  # Convert to 3-channel
            
            # Create comparison grid with all images
            comparison = torch.cat([
                original,        # Original images
                masked,          # Masked input images
                mask_viz,        # Mask visualization
                reconstruction   # Reconstructed images
            ], dim=0)
            
            # Create grid
            grid = vutils.make_grid(comparison, nrow=n_samples, normalize=True, value_range=(-1, 1))
            
            # Compute metrics for these samples (masked regions only)
            sample_metrics = self.compute_batch_metrics(reconstruction, original, mask[:n_samples])
            
            if self.config.logging.use_wandb:
                wandb.log({
                    "samples": wandb.Image(grid),
                    "sample_masked_psnr": sample_metrics['psnr'],
                    "sample_masked_ssim": sample_metrics['ssim']
                }, step=self.global_step)
            
            if self.writer is not None:
                self.writer.add_image("samples", grid, self.global_step)
                self.writer.add_scalar("sample_metrics/masked_psnr", sample_metrics['psnr'], self.global_step)
                self.writer.add_scalar("sample_metrics/masked_ssim", sample_metrics['ssim'], self.global_step)
            
            # Save to file periodically
            if self.global_step % (self.config.logging.sample_interval * 10) == 0:
                os.makedirs('results/samples', exist_ok=True)
                save_path = f'results/samples/step_{self.global_step}.png'
                vutils.save_image(grid, save_path, normalize=True, value_range=(-1, 1))
                print(f"Saved samples to {save_path} | Masked PSNR: {sample_metrics['psnr']:.2f}, Masked SSIM: {sample_metrics['ssim']:.4f}")
    
    def close(self):
        """Clean up logging resources."""
        if self.writer is not None:
            self.writer.close()
        
        if self.config.logging.use_wandb:
            wandb.finish()
