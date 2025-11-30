import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import wandb
from tqdm import tqdm
import os
import math
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter
from masking.mask_generator import MaskGenerator


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted images (B, C, H, W) in range [-1, 1]
        target: Target images (B, C, H, W) in range [-1, 1]
        data_range: The difference between max and min values (2.0 for [-1, 1] range)
    
    Returns:
        Mean PSNR value across the batch
    """
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10((data_range ** 2) / (mse + 1e-8))
    return psnr.mean()


def compute_ssim(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    window_size: int = 11,
    data_range: float = 2.0,
    channel: int = 3
) -> torch.Tensor:
    """
    Compute Structural Similarity Index Measure.
    
    Args:
        pred: Predicted images (B, C, H, W) in range [-1, 1]
        target: Target images (B, C, H, W) in range [-1, 1]
        window_size: Size of the Gaussian window
        data_range: The difference between max and min values
        channel: Number of channels
    
    Returns:
        Mean SSIM value across the batch
    """
    # Create Gaussian window
    def gaussian_window(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.view(1, 1, -1)
    
    # Create 2D window
    sigma = 1.5
    _1d_window = gaussian_window(window_size, sigma)
    _2d_window = _1d_window.t() @ _1d_window
    _2d_window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    window = _2d_window.to(pred.device, dtype=pred.dtype)
    
    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Compute means
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


class MetricsTracker:
    """Track and compute image quality metrics."""
    
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
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Update metrics with a new batch.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            mask: Optional mask for masked region metrics (B, 1, H, W)
        """
        batch_size = pred.shape[0]
        
        with torch.no_grad():
            # MSE
            mse = F.mse_loss(pred, target, reduction='mean')
            self.mse_sum += mse.item() * batch_size
            
            # MAE
            mae = F.l1_loss(pred, target, reduction='mean')
            self.mae_sum += mae.item() * batch_size
            
            # PSNR
            psnr = compute_psnr(pred, target)
            self.psnr_sum += psnr.item() * batch_size
            
            # SSIM
            ssim = compute_ssim(pred, target, channel=pred.shape[1])
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
    """Enhanced trainer with pretrained model support and quality metrics."""
    
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
        self.loss_fn = loss_fn
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
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute MSE, MAE, PSNR, and SSIM for a single batch."""
        with torch.no_grad():
            mse = F.mse_loss(pred, target, reduction='mean').item()
            mae = F.l1_loss(pred, target, reduction='mean').item()
            psnr = compute_psnr(pred, target).item()
            ssim = compute_ssim(pred, target, channel=pred.shape[1]).item()
        
        return {
            'mse': mse,
            'mae': mae,
            'psnr': psnr,
            'ssim': ssim
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Modified train epoch with progressive unfreezing and quality metrics."""
        
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
            
            # Update quality metrics
            self.metrics_tracker.update(outputs['reconstruction'], image, mask)
            batch_metrics = self.compute_batch_metrics(outputs['reconstruction'], image, mask)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'rec': losses['reconstruction'].item(),
                'psnr': batch_metrics['psnr'],
                'ssim': batch_metrics['ssim']
            })
            
            # Logging
            if self.global_step % self.config.logging.log_interval == 0:
                # Merge losses and metrics for logging
                all_metrics = {**losses, **batch_metrics}
                self.log_metrics(all_metrics, 'train')
            
            # Sample generation
            if self.global_step % self.config.logging.sample_interval == 0:
                self.generate_samples(batch, outputs)
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        # Add averaged quality metrics
        quality_metrics = self.metrics_tracker.compute()
        epoch_losses.update(quality_metrics)
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model with quality metrics."""
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
                
                # Update quality metrics
                self.metrics_tracker.update(outputs['reconstruction'], image, mask)
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)
        
        # Add averaged quality metrics
        quality_metrics = self.metrics_tracker.compute()
        val_losses.update(quality_metrics)
        
        return val_losses
    
    def test_model(self) -> Dict[str, float]:
        """Evaluate on test set during training with quality metrics."""
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
                
                # Update quality metrics
                self.metrics_tracker.update(outputs['reconstruction'], image, mask)
        
        # Average losses
        for key in test_losses:
            test_losses[key] /= len(self.test_loader)
        
        # Add averaged quality metrics
        quality_metrics = self.metrics_tracker.compute()
        test_losses.update(quality_metrics)
        
        return test_losses
    
    def train(self):
        """Main training loop with periodic test evaluation and quality metrics."""
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Test every N epochs
            if epoch % 10 == 0 and self.test_loader is not None:
                test_losses = self.test_model()
                print(f"Test Loss: {test_losses['total']:.4f} | "
                      f"PSNR: {test_losses['psnr']:.2f} | SSIM: {test_losses['ssim']:.4f}")
                self.log_metrics(test_losses, 'test_epoch')
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            print(f"\nEpoch {epoch}/{self.config.training.epochs}")
            print(f"Train Loss: {train_losses['total']:.4f} | "
                  f"MSE: {train_losses['mse']:.4f} | MAE: {train_losses['mae']:.4f} | "
                  f"PSNR: {train_losses['psnr']:.2f} | SSIM: {train_losses['ssim']:.4f}")
            print(f"Val Loss: {val_losses['total']:.4f} | "
                  f"MSE: {val_losses['mse']:.4f} | MAE: {val_losses['mae']:.4f} | "
                  f"PSNR: {val_losses['psnr']:.2f} | SSIM: {val_losses['ssim']:.4f}")
            
            self.log_metrics(train_losses, 'train_epoch')
            self.log_metrics(val_losses, 'val_epoch')
            
            # Save checkpoint
            if epoch % self.config.logging.save_interval == 0:
                self.save_checkpoint(epoch, val_losses['total'])
            
            # Save best model (based on loss)
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint(epoch, val_losses['total'], is_best=True, suffix='best_loss')
            
            # Save best model based on PSNR
            if val_losses['psnr'] > self.best_val_psnr:
                self.best_val_psnr = val_losses['psnr']
                self.save_checkpoint(epoch, val_losses['total'], is_best=True, suffix='best_psnr')
            
            # Save best model based on SSIM
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
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str):
        """Log metrics to wandb and tensorboard."""
        # Handle both tensor and float values
        processed_metrics = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                processed_metrics[key] = value.item()
            else:
                processed_metrics[key] = value
        
        if self.config.logging.use_wandb:
            wandb.log({f"{prefix}/{k}": v for k, v in processed_metrics.items()}, step=self.global_step)
        
        if self.writer is not None:
            for key, value in processed_metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}", value, self.global_step)
    
    def generate_samples(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]):
        """Generate sample reconstructions for visualization."""
        import torchvision.utils as vutils
        
        # Create grid of images
        n_samples = min(8, batch['image'].shape[0])

        # Prepare masked images for visualization (compute if not provided)
        imgs = batch['image'][:n_samples]
        try:
            B, C, H, W = imgs.shape
            vis_mask = self.train_mask_gen.generate((B, 1, H, W)).to(imgs.device)
            vis_masked = imgs * (1.0 - vis_mask)
        except Exception:
            vis_masked = imgs  # fallback if shape unexpected

        # Denormalize images for visualization
        def denormalize(x):
            return x  # Images are already in [-1, 1] range

        comparison = torch.cat([
            denormalize(imgs),
            denormalize(vis_masked),
            denormalize(outputs['reconstruction'][:n_samples])
        ], dim=0)
        
        grid = vutils.make_grid(comparison, nrow=n_samples, normalize=True, value_range=(-1, 1))
        
        # Compute metrics for these samples
        sample_metrics = self.compute_batch_metrics(
            outputs['reconstruction'][:n_samples].to(self.device),
            imgs.to(self.device)
        )
        
        if self.config.logging.use_wandb:
            wandb.log({
                "samples": wandb.Image(grid),
                "sample_psnr": sample_metrics['psnr'],
                "sample_ssim": sample_metrics['ssim']
            }, step=self.global_step)
        
        if self.writer is not None:
            self.writer.add_image("samples", grid, self.global_step)
            self.writer.add_scalar("sample_metrics/psnr", sample_metrics['psnr'], self.global_step)
            self.writer.add_scalar("sample_metrics/ssim", sample_metrics['ssim'], self.global_step)
        
        # Save to file periodically
        if self.global_step % (self.config.logging.sample_interval * 10) == 0:
            os.makedirs('/content/drive/MyDrive/vae_results/samples', exist_ok=True)
            save_path = f'/content/drive/MyDrive/vae_results/samples/step_{self.global_step}.png'
            vutils.save_image(grid, save_path)
    
    def close(self):
        """Clean up logging resources."""
        if self.writer is not None:
            self.writer.close()
        
        if self.config.logging.use_wandb:
            wandb.finish()