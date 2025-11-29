import os
import torch
import copy
import torchvision.utils as vutils
import numpy as np

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
            eta_min=1e-6
        )

        # Training parameters
        self.num_epochs = config.data.epochs
        self.num_timesteps = noise_scheduler.num_timesteps

        # --- EMA setup (optional) ---
        # if self.use_ema:
        #     self.ema_decay = 0.9999
        #     self.ema_model = copy.deepcopy(self.model).to(self.device)
        #     self.ema_model.eval()
        #     print("🟢 EMA is ENABLED — using exponential moving average with decay =", self.ema_decay)
        # else:
        #     self.ema_model = None
        #     print("🔴 EMA is DISABLED — training and validation will use raw model weights only.")

        # Internal counters
        self.global_step = 0

    # -------------------------------------------------------------------------
    # EMA helper
    # -------------------------------------------------------------------------
    # @torch.no_grad()
    # def update_ema(self):
    #     """Update exponential moving average of model parameters."""
    #     if not self.use_ema:
    #         return
    #     for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
    #         ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

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
            self.optimizer.zero_grad(set_to_none=True)  # ✓ Clear gradients
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

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    @torch.no_grad()  # ✓ Disable gradient computation for validation (saves memory/compute)
    def validate(self, epoch):
        """Validate using EMA weights if enabled, otherwise raw model."""
        
        # ✓ Select which model to evaluate (EMA provides more stable/better results)
        model_to_eval = self.model #self.ema_model if self.use_ema else self.model
        
        model_to_eval.eval()  # ✓ Set model to evaluation mode (disables dropout, batchnorm training)
        
        total_loss = 0.0  # ✓ Initialize loss accumulator
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)  # ✓ Load validation images [B, C, H, W]
            filenames = batch['filename']            # ✓ Get filenames for deterministic masking
            B, C, H, W = images.shape                # ✓ Extract dimensions
            
            # ✓ Generate deterministic masks based on filenames (same masks each validation)
            masks = self.val_mask_generator.generate_for_filenames(
                filenames=filenames, shape=(1, H, W)
            ).to(self.device)  # [B, 1, H, W]
            
            batch_size = images.size(0)  # ✓ Get batch size
            
            # ✓ Sample random timesteps for each image (validation uses random t like training)
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
            
            # ✓ Add noise to masked regions only
            # noisy_images: clean outside mask, noisy inside mask
            # noise: actual noise values inside mask, 0 outside mask
            noisy_images, noise = self.noise_scheduler.add_noise(images, t, masks)
            
            # ✓ Model predicts noise (output is 0 outside mask due to forward() masking)
            predicted_noise = model_to_eval(noisy_images, t, masks)
            
            # ✓ Compute loss between predicted and target noise
            # Both are 0 outside mask, so only masked region contributes to loss
            loss = self.loss_fn(predicted_noise, noise, masks)
            
            total_loss += loss.item()  # ✓ Accumulate loss across batches
        
        # ✓ Calculate average loss over all validation batches
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss  # ✓ Return validation loss for logging/monitoring

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    def train(self):
        """Full training loop with early stopping and checkpointing."""
        best_val_loss = float('inf')
        epochs_no_improve = 0
        full_val_epoch = 25
        all_psnr = []
        all_ssim = []
        all_mse = []
        all_mae = []


        print("\n🚀 Starting training...")
        # print(f"EMA Status: {'ENABLED' if self.use_ema else 'DISABLED'}")
        print("-" * 60)

        for epoch in range(1, self.num_epochs + 1):
            if epoch > 10:
                break
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 50)

            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")

            val_loss = self.validate(epoch)
            print(f"Validation Loss: {val_loss:.4f}")

            self.scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.save_checkpoint(epoch, "diffusion_best_model.pt")
                print(f"✅ New best model saved (val_loss={val_loss:.4f})")
            else:
                epochs_no_improve += 1

            # if epochs_no_improve >= self.patience:
            #     print(f"\n⏹️ Early stopping after {epoch} epochs (no improvement for {self.patience})")
            #     break

            if (epoch-1) % full_val_epoch == 0:
                metrics = run_validation(self.model, self.val_loader, self.noise_scheduler, self.val_mask_generator, self.device)
                all_psnr.append(metrics['psnr'])
                all_ssim.append(metrics['ssim'])
                all_mse.append(metrics['mse'])
                all_mae.append(metrics['mae'])

        self.save_checkpoint(epoch, "diffusion_final_model.pt")
        print("💾 Final model saved after all epochs.")
        print(f"\n🏁 Training complete — Best validation loss: {best_val_loss:.4f}")
        return all_psnr, all_ssim, all_mse, all_mae

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------
    def save_checkpoint(self, epoch, filename):
        """Save model and EMA state (if available)."""
        checkpoint_dir = self.config.logging.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        if self.use_ema and self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

        path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")

def run_validation(model, test_loader, noise_scheduler, mask_generator, device, save_dir='results/diffusion'):
    """
    Run comprehensive evaluation on test set.
    """
    model.eval()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize metrics calculator
    metrics_calc = InpaintingMetrics(device=device)
    
    # Collect metrics
    all_psnr = []
    all_ssim = []
    all_mse = []
    all_mae = []
    
    print("\nStarting evaluation...")
    out_dir = "./runs/eval_debug"
    os.makedirs(out_dir, exist_ok=True)

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        if batch_idx > 2:
            break
        images = batch['image'].to(device)
        filenames = batch['filename']
        
        B, C, H, W = images.shape
        
        # Generate deterministic masks
        masks = mask_generator.generate_for_filenames(
            filenames=filenames,
            shape=(1, H, W)
        ).to(device)
        
        # Add full noise to masked regions (start from complete noise)
        t = torch.full((B,), noise_scheduler.num_timesteps - 1, device=device)

        noisy_images, _ = noise_scheduler.add_noise(images, t, masks)

        # Denoise using DDPM sampling
        inpainted = sample_ddpm(model, noise_scheduler, noisy_images, masks, num_timesteps=1000)
        
        # Compute all metrics on full images
        psnr_val = metrics_calc.psnr(inpainted, images)
        ssim_val = metrics_calc.ssim(inpainted, images)
        
        # Compute MSE and MAE
        mse_val = torch.mean((inpainted - images) ** 2).item()
        mae_val = torch.mean(torch.abs(inpainted - images)).item()
        
        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)
        all_mse.append(mse_val)
        all_mae.append(mae_val)
        
        # Save first batch for visualization
        if batch_idx == 0:
            # Create masked input: show original with black mask region
            masked_input = images * (1 - masks)  # Zero out masked region
            
            comparison = torch.cat([
                images[:8],           # Row 1: Original
                masked_input[:8],     # Row 2: Input with black mask
                inpainted[:8]         # Row 3: Inpainted result
            ], dim=0)
            
            vutils.save_image(
                comparison,
                os.path.join(save_dir, 'samples.png'),
                nrow=8,
                normalize=True,
                value_range=(-1, 1)
            )
            print(f"\n✅ Saved sample images to {save_dir}/samples.png")
    
    # Compute summary statistics
    return {
        'psnr': np.mean(all_psnr),
        'ssim': np.mean(all_ssim),
        'mse': np.mean(all_mse),
        'mae': np.mean(all_mae)
    }