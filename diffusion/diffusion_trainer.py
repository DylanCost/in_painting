"""
Diffusion Model Training Loop and Validation Logic

This module implements the DiffusionTrainer class which orchestrates the complete training
and validation workflow for the diffusion-based image inpainting model. The trainer manages
training epochs with random mask generation, timestep sampling, and noise prediction loss
computation, while validation performs full DDPM sampling from complete noise to evaluate
reconstruction quality via PSNR, SSIM, MSE, and MAE metrics. It handles optimizer configuration
(AdamW with cosine annealing), gradient clipping, automated checkpointing of best and final
models, comprehensive logging of per-epoch metrics, and periodic visual sample generation.
This file is used by diffusion_train.py for standalone training workflows.
"""

import os

import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils

from diffusion.diffusion_evaluate import sample_ddpm
from evaluation.metrics import InpaintingMetrics

class DiffusionTrainer:
    """
    Trainer for diffusion-based image inpainting with validation, checkpointing, and metric tracking.
    
    Manages the complete training workflow for the U-Net diffusion model, including training loops,
    validation with DDPM sampling, checkpointing based on PSNR, learning rate scheduling, and
    automated metric tracking. The trainer uses separate mask generators for training (random masks
    for diverse patterns) and validation (deterministic masks for consistent metric comparison).
    
    Training Process:
        1. For each batch: generate random masks, sample random timesteps, add noise to masked regions
        2. Model predicts noise in masked areas, loss computed only on masked regions
        3. Gradient clipping (max_norm=1.0) applied to prevent instability
        4. AdamW optimizer with cosine annealing learning rate schedule
    
    Validation Process:
        1. Generate deterministic masks based on filenames for reproducibility
        2. Start from complete noise (t=T-1) in masked regions
        3. Perform full DDPM sampling (1000 timesteps) to denoise
        4. Compute PSNR, SSIM, MSE, MAE metrics on full images
        5. Save visual comparisons every 20 epochs
    
    Outputs:
        - Checkpoints: runs/diffusion/checkpoints/{diffusion_best_model.pt, diffusion_final_model.pt}
        - Training log: runs/diffusion/diffusion_training_log.txt (per-epoch metrics)
        - Visual samples: runs/diffusion/examples/samples_epoch_{N}.png (every 20 epochs)
        - Metrics DataFrame: Returned from train() for CSV export
    
    Args:
        model: UNetDiffusion model instance to train
        train_loader: DataLoader for training set with 'image' and 'filename' keys
        val_loader: DataLoader for validation set with 'image' and 'filename' keys
        loss_fn: Loss function (typically DiffusionLoss computing MSE on masked regions)
        noise_scheduler: NoiseScheduler instance defining forward diffusion process
        config: Config object containing hyperparameters (epochs, lr, etc.)
        device: torch.device for computation ('cuda' or 'cpu')
        train_mask_generator: MaskGenerator for training (random masks)
        val_mask_generator: MaskGenerator for validation (deterministic masks)
    
    Attributes:
        optimizer: AdamW optimizer with lr from config, weight_decay=0.01
        scheduler: CosineAnnealingLR with T_max=num_epochs, eta_min=1e-6
        global_step: Training step counter (incremented per batch)
    """

    def __init__(self, model, train_loader, val_loader, loss_fn, noise_scheduler,
                 config, device, train_mask_generator, val_mask_generator):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.noise_scheduler = noise_scheduler.to(device)
        self.config = config
        self.device = device
        self.train_mask_generator = train_mask_generator
        self.val_mask_generator = val_mask_generator

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
        """
        Execute one training epoch over the entire training dataset.
        
        For each batch: generates random masks, samples random timesteps, adds noise to
        masked regions, predicts noise with the model, computes loss, and updates weights
        via backpropagation with gradient clipping. Logs progress every 100 batches.
        
        Args:
            epoch: Current epoch number (1-indexed) for logging
        
        Returns:
            float: Average loss across all batches in the epoch
        """

        self.model.train()  # ✓ Set model to training mode
        total_loss = 0.0    # ✓ Initialize loss accumulator
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)  # ✓ Load images [B, C, H, W]
            B, C, H, W = images.shape                # ✓ Get dimensions
            
            # Generate random masks
            masks = self.train_mask_generator.generate((B, 1, H, W)).to(self.device)
            # ✓ Generate masks [B, 1, H, W] where 1 = inpaint region 0 = outside inpaint region
            
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
        Run validation on a subset of the validation set using full DDPM sampling.
        
        Generates deterministic masks based on filenames, starts from complete noise (t=T-1),
        performs 1000-step DDPM denoising, and computes PSNR, SSIM, MSE, MAE metrics on
        full images. Saves visual comparisons every 20 epochs. Processes up to 16 batches.
        
        Args:
            epoch: Current epoch number for logging and conditional visualization
        
        Returns:
            dict: Summary statistics with keys 'psnr', 'ssim', 'mse', 'mae' (all averaged)
        """

        self.model.eval()

        # Go up one directory from diffusion/ to project root, then create runs/diffusion/examples
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(os.path.dirname(current_dir), 'runs', 'diffusion', 'examples')
        os.makedirs(save_dir, exist_ok=True)
        
        metrics_calc = InpaintingMetrics(device=self.device)
        # Collect metrics
        all_psnr = []
        all_ssim = []
        all_mse = []
        all_mae = []

        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx > 16: # Only use 1024 images for each validation run
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
            
            # Compute all metrics on masked portions of image
            psnr_val = metrics_calc.psnr(inpainted, images, masks)
            ssim_val = metrics_calc.ssim(inpainted, images, masks)
            mae_val = metrics_calc.compute_mae(inpainted, images, masks)
            
            # Compute MSE and MAE
            mse_val = torch.mean((inpainted - images) ** 2).item()
            
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
        """
        Execute the full training loop across all epochs with validation and checkpointing.
        
        Runs training and validation for the configured number of epochs, tracking metrics
        (PSNR, SSIM, MSE, MAE) and saving the best model based on PSNR. Logs all metrics
        to a text file, applies cosine learning rate scheduling, and saves both best and
        final model checkpoints. Creates a DataFrame of per-epoch metrics for CSV export.
        
        Returns:
            tuple: (all_psnr, all_ssim, all_mse, all_mae, metrics_df) containing lists of
                per-epoch metrics and a pandas DataFrame for external logging
        """
        best_psnr = float('-inf')
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
        
        # Go up one directory from diffusion/ to project root, then create runs/diffusion/
        current_dir = os.path.dirname(__file__)
        log_dir = os.path.join(os.path.dirname(current_dir), "runs", "diffusion")
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

            print(f"Running validation at epoch {epoch}")
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
                self.save_checkpoint(epoch, "diffusion_best_model.pt")
                print(f"✅ New best model saved (psnr={best_psnr:.4f})")

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
        """Save model to runs/diffusion/checkpoints/."""
        # Go up one directory from diffusion/ to project root, then create runs/diffusion/checkpoints
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(os.path.dirname(current_dir), "runs", "diffusion", "checkpoints")
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
