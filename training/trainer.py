import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import wandb
from tqdm import tqdm
import os
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """Main trainer class for VAE inpainting."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        
        # Learning rate scheduler
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=1e-6
        )
        
        # Logging
        if config['logging']['use_wandb']:
            wandb.init(project="vae-inpainting", config=config)
        
        if config['logging']['use_tensorboard']:
            self.writer = SummaryWriter(config['logging']['log_dir'])
        else:
            self.writer = None
        
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total': 0, 'reconstruction': 0, 'kl': 0, 'perceptual': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            image = batch['image'].to(self.device)
            masked_image = batch['masked_image'].to(self.device)
            mask = batch['mask'].to(self.device)
            
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
            if 'gradient_clip' in self.config['training']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'rec': losses['reconstruction'].item(),
                'kl': losses['kl'].item()
            })
            
            # Logging
            if self.global_step % self.config['logging']['log_interval'] == 0:
                self.log_metrics(losses, 'train')
            
            # Sample generation
            if self.global_step % self.config['logging']['sample_interval'] == 0:
                self.generate_samples(batch, outputs)
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = {'total': 0, 'reconstruction': 0, 'kl': 0, 'perceptual': 0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                image = batch['image'].to(self.device)
                masked_image = batch['masked_image'].to(self.device)
                mask = batch['mask'].to(self.device)
                
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
                
                # Update metrics
                for key in val_losses:
                    if key in losses:
                        val_losses[key] += losses[key].item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)
        
        return val_losses
    
    def train(self):
        """Main training loop."""
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            print(f"\nEpoch {epoch}/{self.config['training']['epochs']}")
            print(f"Train Loss: {train_losses['total']:.4f}")
            print(f"Val Loss: {val_losses['total']:.4f}")
            
            self.log_metrics(train_losses, 'train_epoch')
            self.log_metrics(val_losses, 'val_epoch')
            
            # Save checkpoint
            if epoch % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(epoch, val_losses['total'])
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint(epoch, val_losses['total'], is_best=True)
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str):
        """Log metrics to wandb and tensorboard."""
        if self.config['logging']['use_wandb']:
            wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=self.global_step)
        
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}", value, self.global_step)
    
    def generate_samples(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]):
        """Generate and log sample images."""
        import torchvision.utils as vutils
        
        # Create grid of images
        n_samples = min(8, batch['image'].shape[0])
        
        comparison = torch.cat([
            batch['image'][:n_samples],
            batch['masked_image'][:n_samples],
            outputs['reconstruction'][:n_samples]
        ], dim=0)
        
        grid = vutils.make_grid(comparison, nrow=n_samples, normalize=True, value_range=(-1, 1))
        
        if self.config['logging']['use_wandb']:
            wandb.log({"samples": wandb.Image(grid)}, step=self.global_step)
        
        if self.writer is not None:
            self.writer.add_image("samples", grid, self.global_step)