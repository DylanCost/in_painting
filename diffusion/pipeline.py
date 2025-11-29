import torch
from torch.utils.data import DataLoader
import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from noise_scheduler_config import NoiseConfig
from data.celeba_dataset import CelebADataset
from unet_diffusion import UNetDiffusion, NoiseScheduler
from diffusion_loss import DiffusionLoss
from diffusion_trainer import DiffusionTrainer
from masking.mask_generator import MaskGenerator
from diffusion.diffusion_evaluate import load_model, run_evaluation



def main():
    config = Config()
    noise_scheduler = NoiseConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create mask generator
    train_mask_generator = MaskGenerator.for_train(config.mask)
    val_mask_generator = MaskGenerator.for_eval(config.mask)
    test_mask_generator = MaskGenerator.for_eval(config.mask)
    

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
    test_dataset = CelebADataset(
        root_dir=config.data.data_path,
        split='test',
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
    test_loader = DataLoader(
        test_dataset,
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
    all_psnr, all_ssim, all_mse, all_mae = trainer.train()
    print("Training completed!")
    print("\n" + "="*60)
    print("METRICS PER BATCH")
    print("="*60)
    print(f"PSNR values: {all_psnr}")
    print(f"SSIM values: {all_ssim}")
    print(f"MSE values:  {all_mse}")
    print(f"MAE values:  {all_mae}")
    print("="*60)

    model = UNetDiffusion(
        input_channels=config.unet.input_channels,
        hidden_dims=config.unet.hidden_dims,
        use_attention=config.unet.use_attention,
        use_skip_connections=config.unet.use_skip_connections,
    ).to(device)

    model = load_model(model, config, device)
    model.eval()

    results = run_evaluation(
        model=model,
        test_loader=test_loader,
        noise_scheduler=noise_scheduler,
        mask_generator=test_mask_generator,
        device=device,
        save_dir='results/diffusion'
    )

if __name__ == '__main__':
    main()