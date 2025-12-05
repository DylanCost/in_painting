"""
The purpose of this file is to create sets of images that have the exact same masks
so that there can be good qualitative analysis for the paper. This file is unnecessary
for normal runs of the code.
"""

import torch
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, get_triptych_mask_specs
from noise_scheduler_config import NoiseConfig
from data.celeba_dataset import CelebADataset
from unet_diffusion import UNetDiffusion, NoiseScheduler
from masking.mask_generator import MaskGenerator
from diffusion_evaluate import load_model, sample_ddpm


def create_manual_masks(batch_size, height, width, device='cpu'):
    """Create masks using the predefined triptych mask specifications."""
    # Get mask specs for indices 0-7
    mask_specs = get_triptych_mask_specs(indices=list(range(batch_size)))
    
    # Create empty mask tensor (batch_size, 1, height, width)
    masks = torch.zeros((batch_size, 1, height, width), device=device)
    
    # Apply each mask specification
    for i, spec in enumerate(mask_specs):
        # Set masked region to 1 (remember: 1 inside mask, 0 outside)
        masks[i, 0, spec.top:spec.bottom, spec.left:spec.right] = 1.0
    
    return masks


def main():
    # Load config
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")
    
    # Create test dataset
    test_dataset = CelebADataset(
        root_dir=config.data.data_path,
        split='test',
        image_size=config.data.image_size,
        download=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = UNetDiffusion(
        input_channels=config.unet.input_channels,
        hidden_dims=config.unet.hidden_dims,
        use_attention=config.unet.use_attention,
        use_skip_connections=config.unet.use_skip_connections,
    ).to(device)

    model = load_model(model, config, device)
    
    model.eval()
    
    # Create noise scheduler
    noise_config = NoiseConfig()
    noise_scheduler = NoiseScheduler(
        num_timesteps=noise_config.num_timesteps,
        beta_start=noise_config.beta_start,
        beta_end=noise_config.beta_end,
        schedule_type=noise_config.schedule_type
    ).to(device)

    # Grab first batch and display first 8 images
    first_batch = next(iter(test_loader))
    first_batch_images = first_batch['image'].to(device)
    images = first_batch_images[:8]  # Get first 8 images
    
    batch_size, channels, height, width = images.shape
    masks = create_manual_masks(batch_size, height, width, device=images.device).to(device)

    masked_images = images * (1 - masks)

    t = torch.full((batch_size,), noise_scheduler.num_timesteps - 1, device=device)
    noisy_images, _ = noise_scheduler.add_noise(images, t, masks)
    inpainted = sample_ddpm(model, noise_scheduler, noisy_images, masks, num_timesteps=1000)


    # Concatenate: original on top, masked on bottom
    comparison1 = torch.cat([
        images[:4],           # Row 1: Original (8 images)
        masked_images[:4], # Row 2: Masked images (8 images)
        inpainted[:4]
    ], dim=0)
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the comparison image
    vutils.save_image(
        comparison1,
        os.path.join(save_dir, 'manual_masks1.png'),
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )

    # Concatenate: original on top, masked on bottom
    comparison2 = torch.cat([
        images[4:],           # Row 1: Original (8 images)
        masked_images[4:], # Row 2: Masked images (8 images)
        inpainted[4:]
    ], dim=0)
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the comparison image
    vutils.save_image(
        comparison2,
        os.path.join(save_dir, 'manual_masks2.png'),
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )
    
    print(f"\n✅ Saved sample images to {save_dir}/manual_masks.png")
    print(f"Successfully applied {batch_size} manual masks")


if __name__ == '__main__':
    main()