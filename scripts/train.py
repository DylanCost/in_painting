#!/usr/bin/env python
"""Main training script for VAE inpainting."""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.celeba_dataset import CelebADataset, MaskGenerator
from models.unet_vae import UNetVAE
from losses.vae_loss import VAELoss
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train VAE for inpainting')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create mask generator
    mask_generator = MaskGenerator(
        mask_type=config['mask']['type'],
        mask_ratio=config['mask']['mask_ratio'],
        min_size=config['mask']['min_size'],
        max_size=config['mask']['max_size']
    )
    
    # Create datasets
    train_dataset = CelebADataset(
        root_dir=config['data']['data_path'],
        split='train',
        image_size=config['data']['image_size'],
        mask_generator=mask_generator
    )
    
    val_dataset = CelebADataset(
        root_dir=config['data']['data_path'],
        split='val',
        image_size=config['data']['image_size'],
        mask_generator=mask_generator
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = UNetVAE(
        input_channels=config['model']['input_channels'],
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['hidden_dims'],
        use_attention=config['model']['use_attention'],
        use_skip_connections=config['model']['use_skip_connections']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    loss_fn = VAELoss(
        kl_weight=config['training']['kl_weight'],
        perceptual_weight=config['training']['perceptual_weight'],
        adversarial_weight=config['training']['adversarial_weight']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        device=device
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == '__main__':
    main()