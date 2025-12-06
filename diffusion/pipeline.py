"""
Complete Diffusion Model Training and Evaluation Pipeline

This module serves as the main entry point for the full end-to-end workflow: training the
diffusion-based image inpainting model on the training set with validation monitoring, then
evaluating the best checkpoint on the test set. It orchestrates dataset loading for all three
splits (train/val/test), initializes the model and noise scheduler, runs the complete training
loop with checkpointing, saves training metrics and logs, reloads the best model checkpoint,
and performs comprehensive test set evaluation with DDPM sampling to generate final performance
metrics and visual comparisons. For training-only workflows without test evaluation, use
diffusion_train.py instead.
"""

import os
import sys

import torch
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from data.celeba_dataset import CelebADataset
from diffusion.diffusion_evaluate import load_model, run_evaluation
from diffusion_loss import DiffusionLoss
from diffusion_trainer import DiffusionTrainer
from masking.mask_generator import MaskGenerator
from noise_scheduler_config import NoiseConfig
from scripts.set_seed import set_seed
from unet_diffusion import NoiseScheduler, UNetDiffusion


def main():
    # Set random seed for reproducibility
    set_seed()
    # Load configuration for model and training parameters
    config = Config()
    # Load noise scheduler configuration
    noise_scheduler = NoiseConfig()
    # Set device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create mask generators: random masks for training, deterministic for validation/test
    train_mask_generator = MaskGenerator.for_train(config.mask)
    val_mask_generator = MaskGenerator.for_eval(config.mask)
    test_mask_generator = MaskGenerator.for_eval(config.mask)
    
    # Initialize U-Net diffusion model with configuration parameters
    model = UNetDiffusion(
        input_channels=config.unet.input_channels,
        hidden_dims=config.unet.hidden_dims,
        use_attention=config.unet.use_attention,
        use_skip_connections=config.unet.use_skip_connections,
    )

    # Create CelebA dataset for training split (download if needed)
    train_dataset = CelebADataset(
        root_dir=config.data.data_path,
        split='train',
        image_size=config.data.image_size,
        download=True
    )
    # Create CelebA dataset for validation split
    val_dataset = CelebADataset(
        root_dir=config.data.data_path,
        split='val',
        image_size=config.data.image_size,
        download=False
    )
    # Create CelebA dataset for test split
    test_dataset = CelebADataset(
        root_dir=config.data.data_path,
        split='test',
        image_size=config.data.image_size,
        download=False
    )
    
    # Create DataLoader for training with shuffling enabled
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    # Create DataLoader for validation without shuffling
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    # Create DataLoader for test set without shuffling
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    # Print total number of trainable parameters in the model
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize noise scheduler with beta schedule parameters
    noise_scheduler = NoiseScheduler(
        num_timesteps=noise_scheduler.num_timesteps,
        beta_start=noise_scheduler.beta_start,
        beta_end=noise_scheduler.beta_end,
        schedule_type=noise_scheduler.schedule_type
    )

    # Initialize loss function for diffusion training
    loss_fn = DiffusionLoss()

    # Create trainer with model, data loaders, and training configuration
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

    # Execute full training loop with validation
    print("Starting training...")
    all_psnr, all_ssim, all_mse, all_mae, dataframe = trainer.train()
    print("Training completed!")

    # Create directory for saving logs and metrics
    current_dir = os.path.dirname(__file__)
    log_dir = os.path.join(os.path.dirname(current_dir), "runs", "diffusion")
    os.makedirs(log_dir, exist_ok=True)

    # Save per-epoch metrics to CSV file
    csv_path = os.path.join(log_dir, "diffusion_data.csv")
    dataframe.to_csv(csv_path, index=False)
    print(f"DataFrame saved to {csv_path}")


    # Find best validation metrics across all epochs
    max_psnr = max(all_psnr)
    max_ssim = max(all_ssim)
    min_mse = min(all_mse)
    min_mae = min(all_mae)

    # Save best validation metrics to text file
    txt_path = os.path.join(log_dir, "best_diffusion_metrics.txt")
    with open(txt_path, 'w') as f:
        f.write("Best Validation Metrics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Highest PSNR: {max_psnr:.4f}\n")
        f.write(f"Highest SSIM: {max_ssim:.4f}\n")
        f.write(f"Lowest MSE:   {min_mse:.6f}\n")
        f.write(f"Lowest MAE:   {min_mae:.6f}\n")

    print(f"✅ Best metrics saved to {txt_path}")

    # Reinitialize model for test evaluation and move to device
    model = UNetDiffusion(
        input_channels=config.unet.input_channels,
        hidden_dims=config.unet.hidden_dims,
        use_attention=config.unet.use_attention,
        use_skip_connections=config.unet.use_skip_connections,
    ).to(device)

    # Load best checkpoint from training
    model = load_model(model, device)
    # Set model to evaluation mode
    model.eval()

    # Run comprehensive evaluation on test set with DDPM sampling
    run_evaluation(
        model=model,
        test_loader=test_loader,
        noise_scheduler=noise_scheduler,
        mask_generator=test_mask_generator,
        device=device
    )

if __name__ == '__main__':
    main()