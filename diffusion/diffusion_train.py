import torch
from torch.utils.data import DataLoader
import sys
import os
import numpy as np

# Ensure the project root (the directory containing diffusion/, data/, etc.) is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # e.g. .../in_paint_structure/diffusion
project_root = os.path.dirname(project_root)               # go one level up → .../in_paint_structure
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from noise_scheduler_config import NoiseConfig
from data.celeba_dataset import CelebADataset
from unet_diffusion import UNetDiffusion, NoiseScheduler
from diffusion_loss import DiffusionLoss
from diffusion_trainer import DiffusionTrainer
from masking.mask_generator import MaskGenerator
from scripts.set_seed import set_seed


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
        val_mask_generator=val_mask_generator, # Deterministic for Validation
        patience=10,
        use_ema=False
    )

    # Start training
    print("Starting training...")
    all_psnr, all_ssim, all_mse, all_mae, dataframe = trainer.train()
    print("Training completed!")
    print("\n" + "="*60)
    print("METRICS PER BATCH")
    print("="*60)
    print(f"PSNR values: {all_psnr:.2f}")
    print(f"SSIM values: {all_ssim:.2f}")
    print(f"MSE values:  {all_mse:.2f}")
    print(f"MAE values:  {all_mae:.2f}")
    print("="*60)



    # Export to logs directory (matching your training setup)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
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