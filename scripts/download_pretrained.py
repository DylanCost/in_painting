#!/usr/bin/env python
"""Download pretrained model weights."""

import os
import torch
import requests
from tqdm import tqdm


PRETRAINED_URLS = {
    'vggface_resnet50': 'https://github.com/1adrianb/face-alignment/releases/download/v0.1.0/vggface2.pth',
    'celeba_vae_reconstruction': 'https://example.com/celeba_vae_pretrained.pt',  # Your hosted weights
    'stylegan_encoder': 'https://github.com/omertov/encoder4editing/releases/download/v0.1.0/e4e_ffhq_encode.pt'
}


def download_file(url: str, dest_path: str):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))


def download_pretrained_weights():
    """Download all pretrained weights."""
    weights_dir = './weights/pretrained'
    
    for name, url in PRETRAINED_URLS.items():
        dest_path = os.path.join(weights_dir, f'{name}.pth')
        
        if os.path.exists(dest_path):
            print(f"{name} already downloaded")
            continue
        
        print(f"Downloading {name}...")
        download_file(url, dest_path)
        print(f"Saved to {dest_path}")


if __name__ == '__main__':
    download_pretrained_weights()