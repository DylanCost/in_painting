#!/usr/bin/env python
"""Main training script for VAE inpainting."""

import argparse
import torch
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import default_config, pretrained_config, Config
from data.celeba_dataset import CelebADataset
from models.unet_vae import UNetVAE
from losses.vae_loss import VAELoss
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train VAE for inpainting')
    parser.add_argument('--config', type=str, choices=['default', 'pretrained'], default='default',
                        help='Configuration preset to use')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained VAE checkpoint')
    parser.add_argument('--pretrained_encoder', type=str,
                        choices=['resnet', 'vggface', 'vae', 'stylegan', 'none'],
                        default='none', help='Type of pretrained encoder')
    parser.add_argument('--freeze_encoder', type=int, default=0,
                        help='Number of encoder stages to freeze')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device to use (e.g., cpu, cuda, cuda:0)')
    args = parser.parse_args()
    
    # Load configuration
    if args.config == 'pretrained':
        config = pretrained_config.copy()
    else:
        config = default_config.copy()
    
    # Override with command line arguments
    if args.pretrained_encoder != 'none':
        config.model.pretrained_encoder = args.pretrained_encoder
        config.model.encoder_checkpoint = args.pretrained
        config.model.freeze_encoder_stages = args.freeze_encoder
    
    # Create model with pretrained options
    model = UNetVAE(
        input_channels=config.model.input_channels,
        latent_dim=config.model.latent_dim,
        hidden_dims=config.model.hidden_dims,
        image_size=config.data.image_size,
        use_attention=config.model.use_attention,
        use_skip_connections=config.model.use_skip_connections,
        pretrained_encoder=config.model.pretrained_encoder,
        encoder_checkpoint=config.model.encoder_checkpoint,
        freeze_encoder_stages=config.model.freeze_encoder_stages
    )
    
    # Load pretrained VAE weights if provided
    if args.pretrained and args.pretrained_encoder == 'vae':
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded pretrained weights from {args.pretrained}")
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create datasets (auto-download via torchvision if missing)
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
        download=False  # Already downloaded
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    loss_fn = VAELoss(
        kl_weight=config.training.kl_weight,
        perceptual_weight=config.training.perceptual_weight,
        adversarial_weight=config.training.adversarial_weight
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        mask_config=config.mask,
        device=device
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == '__main__':
    main()