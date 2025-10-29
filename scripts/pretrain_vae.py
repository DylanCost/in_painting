#!/usr/bin/env python
"""Pretrain VAE on reconstruction before inpainting."""

import argparse
import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import default_config, pretrained_config
from data.celeba_dataset import CelebADataset
from models.unet_vae import UNetVAE
from losses.vae_loss import VAELoss
from training.trainer import Trainer


class ReconstructionDataset(CelebADataset):
    """Dataset for pretraining on reconstruction (no masks)."""
    
    def __getitem__(self, idx: int):
        data = super().__getitem__(idx)
        # For pretraining, input is the same as target
        data['masked_image'] = data['image']
        data['mask'] = torch.zeros(1, self.image_size, self.image_size)
        return data


def pretrain_reconstruction(config, output_path: str):
    """Pretrain VAE on reconstruction task."""
    
    # Modify config for pretraining
    config.training.epochs = 50
    config.training.kl_weight = 0.0001  # Lower KL weight initially
    
    # Create dataset without masks
    train_dataset = ReconstructionDataset(
        root_dir=config.data.data_path,
        split='train',
        image_size=config.data.image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    
    # Create model
    model = UNetVAE(
        input_channels=config.model.input_channels,
        latent_dim=config.model.latent_dim,
        hidden_dims=config.model.hidden_dims,
        use_attention=config.model.use_attention
    )
    
    # Train on reconstruction
    trainer = Trainer(model, train_loader, None, VAELoss(), config)
    trainer.train()
    
    # Save pretrained weights
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'config': config
    }, output_path)
    
    print(f"Saved pretrained model to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, choices=['default', 'pretrained'], default='default')
    parser.add_argument('--output', type=str, default='weights/pretrained_vae.pt')
    args = parser.parse_args()
    
    # Load config
    if args.config == 'pretrained':
        config = pretrained_config.copy()
    else:
        config = default_config.copy()
    
    pretrain_reconstruction(config, args.output)