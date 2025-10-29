import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import wandb
from tqdm import tqdm
import os
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter
from masking.mask_generator import MaskGenerator


class Trainer:
    """Enhanced trainer with pretrained model support."""
    
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
            cache_dir=self.mask_config.cache_dir if hasattr(self.mask_config, 'cache_dir') else './data/masks'
        )
        
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
    
    def train_epoch(self) -> Dict[str, float]:
        """Modified train epoch with progressive unfreezing."""
        
        # Check if we should unfreeze layers
        self.maybe_unfreeze_layers(self.current_epoch)
        
        self.model.train()
        epoch_losses = {'total': 0, 'reconstruction': 0, 'kl': 0, 'perceptual': 0}
        
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
            if self.global_step % self.config.logging.log_interval == 0:
                self.log_metrics(losses, 'train')
            
            # Sample generation
            if self.global_step % self.config.logging.sample_interval == 0:
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
                
                # Update metrics
                for key in val_losses:
                    if key in losses:
                        val_losses[key] += losses[key].item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)
        
        return val_losses
    
    def test_model(self) -> Dict[str, float]:
        """Evaluate on test set during training."""
        if self.test_loader is None:
            return {}
        
        self.model.eval()
        test_losses = {'total': 0, 'reconstruction': 0, 'kl': 0, 'perceptual': 0}
        
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
        
        # Average losses
        for key in test_losses:
            test_losses[key] /= len(self.test_loader)
        
        return test_losses
    
    def train(self):
        """Main training loop with periodic test evaluation."""
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Test every N epochs
            if epoch % 10 == 0 and self.test_loader is not None:
                test_losses = self.test_model()
                print(f"Test Loss: {test_losses['total']:.4f}")
                self.log_metrics(test_losses, 'test_epoch')
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            print(f"\nEpoch {epoch}/{self.config.training.epochs}")
            print(f"Train Loss: {train_losses['total']:.4f}")
            print(f"Val Loss: {val_losses['total']:.4f}")
            
            self.log_metrics(train_losses, 'train_epoch')
            self.log_metrics(val_losses, 'val_epoch')
            
            # Save checkpoint
            if epoch % self.config.logging.save_interval == 0:
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
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'global_step': self.global_step
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
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
        """Generate and log sample images."""
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
        
        if self.config.logging.use_wandb:
            wandb.log({"samples": wandb.Image(grid)}, step=self.global_step)
        
        if self.writer is not None:
            self.writer.add_image("samples", grid, self.global_step)
        
        # Save to file periodically
        if self.global_step % (self.config.logging.sample_interval * 10) == 0:
            os.makedirs('results/samples', exist_ok=True)
            save_path = f'results/samples/step_{self.global_step}.png'
            vutils.save_image(grid, save_path)
    
    def close(self):
        """Clean up logging resources."""
        if self.writer is not None:
            self.writer.close()
        
        if self.config.logging.use_wandb:
            wandb.finish()