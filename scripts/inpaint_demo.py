#!/usr/bin/env python
"""Interactive demo for image inpainting."""

import argparse
import torch
import yaml
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_vae import UNetVAE
from data.celeba_dataset import MaskGenerator


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = UNetVAE(
        input_channels=config['model']['input_channels'],
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['hidden_dims'],
        use_attention=config['model']['use_attention'],
        use_skip_connections=config['model']['use_skip_connections']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def inpaint_image(
    model: torch.nn.Module,
    image_path: str,
    mask_generator: MaskGenerator,
    device: str = 'cuda'
):
    """Inpaint a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate mask
    mask = mask_generator.generate((1, 256, 256))
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(device)
    
    # Create masked image
    masked_image = image_tensor * (1 - mask_tensor)
    
    # Inpaint
    with torch.no_grad():
        outputs = model(masked_image, mask_tensor)
        inpainted = outputs['reconstruction']
    
    # Combine inpainted region with original
    final_image = image_tensor * (1 - mask_tensor) + inpainted * mask_tensor
    
    return {
        'original': image_tensor.cpu(),
        'mask': mask_tensor.cpu(),
        'masked': masked_image.cpu(),
        'inpainted': inpainted.cpu(),
        'final': final_image.cpu()
    }


def visualize_results(results: dict):
    """Visualize inpainting results."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    titles = ['Original', 'Mask', 'Masked', 'Inpainted', 'Final']
    images = [results['original'], results['mask'], results['masked'], 
              results['inpainted'], results['final']]
    
    for ax, title, img in zip(axes, titles, images):
        # Denormalize and convert to numpy
        if title != 'Mask':
            img = (img.squeeze(0).permute(1, 2, 0) + 1) / 2
        else:
            img = img.squeeze()
        
        ax.imshow(img.numpy() if title != 'Mask' else img.numpy(), cmap='gray' if title == 'Mask' else None)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Interactive inpainting demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--mask_type', type=str, default='random',
                        choices=['random', 'center', 'irregular'],
                        help='Type of mask to use')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, config = load_model(args.checkpoint, args.device)
    
    # Create mask generator
    mask_generator = MaskGenerator(mask_type=args.mask_type)
    
    # Inpaint image
    print("Inpainting image...")
    results = inpaint_image(model, args.image, mask_generator, args.device)
    
    # Visualize
    visualize_results(results)


if __name__ == '__main__':
    main()