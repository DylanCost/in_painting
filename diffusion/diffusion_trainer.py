import os
import torch
import copy
import torchvision.utils as vutils
import numpy as np
import pandas as pd

from config import Config
from noise_scheduler_config import NoiseConfig
from data.celeba_dataset import CelebADataset
from unet_diffusion import UNetDiffusion, NoiseScheduler
from masking.mask_generator import MaskGenerator
from evaluation.metrics import InpaintingMetrics
from diffusion.diffusion_evaluate import sample_ddpm

# from diffusion_evaluate import run_evaluation

class DiffusionTrainer:
    """
    Trainer for diffusion inpainting model with optional Exponential Moving Average (EMA) stabilization.
    """

    def __init__(self, model, train_loader, val_loader, loss_fn, noise_scheduler,
                 config, device, train_mask_generator, val_mask_generator, patience,
                 use_ema=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.noise_scheduler = noise_scheduler.to(device)
        self.config = config
        self.device = device
        self.train_mask_generator = train_mask_generator
        self.val_mask_generator = val_mask_generator
        self.patience = patience
        self.use_ema = use_ema  # 👈 flag controls EMA usage

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.data.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # Cosine LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.data.epochs,
            eta_min=1e-6 #1e-6
        )

        # Training parameters
        self.num_epochs = config.data.epochs
        self.num_timesteps = noise_scheduler.num_timesteps

        # Internal counters
        self.global_step = 0

    # -------------------------------------------------------------------------
    # Training epoch
    # -------------------------------------------------------------------------
    def train_epoch(self, epoch):
        self.model.train()  # ✓ Set model to training mode
        total_loss = 0.0    # ✓ Initialize loss accumulator
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)  # ✓ Load images [B, C, H, W]
            B, C, H, W = images.shape                # ✓ Get dimensions
            
            # Generate random masks
            masks = self.train_mask_generator.generate((B, 1, H, W)).to(self.device)
            # ✓ Generate masks [B, 1, H, W] where 1 = inpaint region
            
            batch_size = images.size(0)  # ✓ Get batch size (same as B above)
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
            # ✓ Sample random timesteps for each image in batch
            
            # Add noise (only masked region), noise = noise in mask, 0 elsewhere
            noisy_images, noise = self.noise_scheduler.add_noise(images, t, masks)
            # ✓ noisy_images: masked region is noisy, unmasked is clean x_0
            # ✓ noise: contains actual noise values in mask, 0 elsewhere
            
            # Forward pass
            predicted_noise = self.model(noisy_images, t, masks)
            # ✓ Model predicts noise, masked to output 0 outside mask region
            
            loss = self.loss_fn(predicted_noise, noise, masks)
            # ⚠️ THIS is where we need to check the loss function!
            # Both predicted_noise and noise should be 0 outside mask
            # So loss_fn might need adjustment depending on implementation
            
            # Backpropagation
            self.optimizer.zero_grad()  # ✓ Clear gradients
            loss.backward()                              # ✓ Compute gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # ✓ Clip gradients to prevent explosion
            self.optimizer.step()                        # ✓ Update weights
            
            # self.update_ema()  # ✓ Update EMA model if enabled
            
            total_loss += loss.item()  # ✓ Accumulate loss
            self.global_step += 1      # ✓ Increment global step counter
            
            # Logging every 100 batches ✓
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{self.num_epochs}] "
                    f"Batch [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)  # ✓ Calculate average
        return avg_loss  # ✓ Return epoch loss

    # # -------------------------------------------------------------------------
    # # Validation
    # # -------------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch):
        """
        Run comprehensive evaluation on test set.
        """
        self.model.eval()

        # Use current directory's results folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, 'results')
        
        metrics_calc = InpaintingMetrics(device=self.device)
        # Collect metrics
        all_psnr = []
        all_ssim = []
        all_mse = []
        all_mae = []

        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx > 10:
                break
            images = batch['image'].to(self.device)
            filenames = batch['filename']
            
            B, C, H, W = images.shape
            
            # Generate deterministic masks
            masks = self.val_mask_generator.generate_for_filenames(
                filenames=filenames,
                shape=(1, H, W)
            ).to(self.device)
            
            # Add full noise to masked regions (start from complete noise)
            t = torch.full((B,), self.noise_scheduler.num_timesteps - 1, device=self.device)

            noisy_images, _ = self.noise_scheduler.add_noise(images, t, masks)

            # Denoise using DDPM sampling
            inpainted = sample_ddpm(self.model, self.noise_scheduler, noisy_images, masks, num_timesteps=1000)
            
            # Compute all metrics on full images
            psnr_val = metrics_calc.psnr(inpainted, images, masks)
            ssim_val = metrics_calc.ssim(inpainted, images)
            
            # Compute MSE and MAE
            mse_val = torch.mean((inpainted - images) ** 2).item()
            mae_val = torch.mean(torch.abs(inpainted - images)).item()
            
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_mse.append(mse_val)
            all_mae.append(mae_val)

            # Save first batch for visualization every 20 epochs
            if batch_idx == 0 and (epoch % 20 == 0 or epoch == 1):
                masked_input = images * (1 - masks)  # Zero out masked region
                
                comparison = torch.cat([
                    images[:8],           # Row 1: Original
                    masked_input[:8],     # Row 2: Input with black mask
                    inpainted[:8]         # Row 3: Inpainted result
                ], dim=0)
                
                vutils.save_image(
                    comparison,
                    os.path.join(save_dir, f'samples_epoch_{epoch}.png'),
                    nrow=8,
                    normalize=True,
                    value_range=(-1, 1)
                )
                print(f"\n✅ Saved sample images to {save_dir}/samples_epoch_{epoch}.png")
        
        # Compute summary statistics
        return {
            'psnr': np.mean(all_psnr),
            'ssim': np.mean(all_ssim),
            'mse': np.mean(all_mse),
            'mae': np.mean(all_mae)
        }

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    def train(self):
        """Full training loop with early stopping and checkpointing."""
        best_psnr = float('-inf')
        epochs_no_improve = 0
        # full_val_epoch = 25
        all_psnr = []
        all_ssim = []
        all_mse = []
        all_mae = []

        # Create DataFrame to store metrics
        metrics_data = {
            'epoch': [],
            'psnr': [],
            'ssim': [],
            'mae': []
        }
        
        # Create logs directory in the CURRENT directory (same folder as this file)
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "diffusion_training_log.txt")

        with open(log_file, 'w') as f:
            f.write("Training Log\n")
            f.write("=" * 60 + "\n\n")


        print("\n🚀 Starting training...")
        print("-" * 60)

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 50)

            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")

            metrics = self.validate(epoch)
            curr_psnr = metrics['psnr']
            print(f"Validation PSNR: {curr_psnr:.4f}")

            # Add metrics to DataFrame data
            metrics_data['epoch'].append(epoch)
            metrics_data['psnr'].append(metrics['psnr'])
            metrics_data['ssim'].append(metrics['ssim'])
            metrics_data['mae'].append(metrics['mae'])

            # Write to log file
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch}/{self.num_epochs}\n")
                f.write(f"Train Loss: {train_loss:.4f}\n")
                f.write(f"Validation PSNR: {curr_psnr:.4f}\n")
                f.write(f"Validation SSIM: {metrics['ssim']:.4f}\n")
                f.write(f"Validation MSE: {metrics['mse']:.6f}\n")
                f.write(f"Validation MAE: {metrics['mae']:.6f}\n")
                f.write("\n")

            self.scheduler.step()

            if curr_psnr > best_psnr:
                best_psnr = curr_psnr
                epochs_no_improve = 0
                self.save_checkpoint(epoch, "diffusion_best_model.pt")
                print(f"✅ New best model saved (psnr={best_psnr:.4f})")
            else:
                epochs_no_improve += 1

            # if epochs_no_improve >= self.patience:
            #     print(f"\n⏹️ Early stopping after {epoch} epochs (no improvement for {self.patience})")
            #     break

            all_psnr.append(metrics['psnr'])
            all_ssim.append(metrics['ssim'])
            all_mse.append(metrics['mse'])
            all_mae.append(metrics['mae'])
        
        metrics_df = pd.DataFrame(metrics_data)

        self.save_checkpoint(epoch, "diffusion_final_model.pt")
        print("💾 Final model saved after all epochs.")
        print(f"\n🏁 Training complete — Best PSNR: {best_psnr:.4f}")

        with open(log_file, 'a') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Training complete — Best PSNR: {best_psnr:.4f}\n")
            f.write("=" * 60 + "\n")

        return all_psnr, all_ssim, all_mse, all_mae, metrics_df

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------
    def save_checkpoint(self, epoch, filename):
        """Save model and EMA state (if available)."""
        # Use current directory instead of config path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(current_dir, "weights")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")
