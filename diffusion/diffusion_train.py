"""
Diffusion Model Training Entry Point

This module serves as the main entry point for training the diffusion-based image inpainting
model in isolation, without running the full evaluation pipeline. It orchestrates the complete
training workflow by initializing all necessary components (model, datasets, schedulers, loss
functions), configuring the trainer, and executing the training loop with validation.

The script sets up both training and validation data loaders with CelebA dataset splits, creates
random mask generators for training (to ensure model generalization across diverse mask patterns)
and deterministic mask generators for validation (to enable consistent metric tracking across
epochs). After training completes, it saves comprehensive results including per-epoch metrics
in CSV format and best validation scores in a summary text file. For full end-to-end training
plus test set evaluation, use pipeline.py instead.
"""

import os
import sys

import torch
from torch.utils.data import DataLoader

# Ensure the project root (the directory containing diffusion/, data/, etc.) is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # e.g. .../in_paint_structure/diffusion
project_root = os.path.dirname(project_root)               # go one level up → .../in_paint_structure
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from data.celeba_dataset import CelebADataset
from diffusion_loss import DiffusionLoss
from diffusion_trainer import DiffusionTrainer
from masking.mask_generator import MaskGenerator
from noise_scheduler_config import NoiseConfig
from scripts.set_seed import set_seed
from unet_diffusion import NoiseScheduler, UNetDiffusion


def main():
    set_seed()
    config = Config()
    noise_scheduler = NoiseConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create mask generator
    train_mask_generator = MaskGenerator.for_train(config.mask)
    val_mask_generator = MaskGenerator.for_eval(config.mask)
    

    model = UNetDiffusion(
        input_channels=config.unet.input_channels,
        hidden_dims=config.unet.hidden_dims,
        use_attention=config.unet.use_attention,
        use_skip_connections=config.unet.use_skip_connections,
    )

    train_dataset = CelebADataset(
        root_dir=config.data.data_path,
        split='train',
        image_size=config.data.image_size,
        download=True
    )
    val_dataset = CelebADataset(
        root_dir=config.data.data_path,
        split='val',
        image_size=config.data.image_size,
        download=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create noise scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=noise_scheduler.num_timesteps,
        beta_start=noise_scheduler.beta_start,
        beta_end=noise_scheduler.beta_end,
        schedule_type=noise_scheduler.schedule_type
    )

    # Create loss function
    loss_fn = DiffusionLoss()

    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        noise_scheduler=noise_scheduler,
        config=config,
        device=device,
        train_mask_generator=train_mask_generator,  # Random for training
        val_mask_generator=val_mask_generator # Deterministic for Validation
    )

    # Start training
    print("Starting training...")
    all_psnr, all_ssim, all_mse, all_mae, dataframe = trainer.train()
    print("Training completed!")
    print("\n" + "="*60)


    # Go up one directory from diffusion/ to project root, then create runs/diffusion/
    current_dir = os.path.dirname(__file__)
    log_dir = os.path.join(os.path.dirname(current_dir), "runs", "diffusion")
    os.makedirs(log_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, "diffusion_data.csv")
    dataframe.to_csv(csv_path, index=False)
    print(f"DataFrame saved to {csv_path}")


    # Find maximum values
    max_psnr = max(all_psnr)
    max_ssim = max(all_ssim)
    min_mse = min(all_mse)   # Note: Lower MSE is better
    min_mae = min(all_mae)   # Note: Lower MAE is better

    # Write to text file
    txt_path = os.path.join(log_dir, "best_diffusion_metrics.txt")
    with open(txt_path, 'w') as f:
        f.write("Best Validation Metrics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Highest PSNR: {max_psnr:.4f}\n")
        f.write(f"Highest SSIM: {max_ssim:.4f}\n")
        f.write(f"Lowest MSE:   {min_mse:.6f}\n")
        f.write(f"Lowest MAE:   {min_mae:.6f}\n")

    print(f"✅ Best metrics saved to {txt_path}")

if __name__ == '__main__':
    main()