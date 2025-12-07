# CelebA Inpainting

This repository contains multiple approaches to image inpainting on the CelebA dataset, including:
- a flow-matching pipeline,
- a diffusion-based pipeline,
- and a convolutional VAE baseline for masked-face reconstruction.

## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

See [`requirements.txt`](requirements.txt) for the full dependency list.

Key dependencies include:

- PyTorch
- torchvision
- numpy
- Pillow
- matplotlib
- tqdm

## Common Modules

The following modules are shared between models:

- [`celeba_dataset.py`](data/celeba_dataset.py) – CelebA dataset class.
- [`metrics.py`](evaluation/metrics.py) – Common evaluation metrics (PSNR, SSIM, etc.).
- [`common_config.py`](config/common_config.py) – Common configuration for all models.
- [`mask_generator.py`](masking/mask_generator.py) – Mask generation logic.
- [`vae_loss.py`](losses/vae_loss.py) – VAE loss implementation.

## Flowmatching Pipeline

The flow-matching implementation is contained under [`flowmatching/`](flowmatching). The flow-matching pipeline can be run via:

```bash
python -m flowmatching.pipeline --epochs 100 --batch_size 64 --num_eval_samples 1024 --num_example_images 8
```

This outputs runs stored in the following directory structure:

```text
├─ runs/
│  └─ {model}/
│     └─ {timestamp}/
│        ├─ eval_results.json
│        ├─ config_snapshot.json
│        ├─ training.log
│        ├─ learning_curves.png
│        ├─ history.csv (optional)
│        ├─ checkpoints/
│        │  ├─ best.ckpt
│        │  └─ last.ckpt
│        └─ examples/
│           ├─ triptych_0001.png
│           └─ examples_grid.png
```

## Diffusion Pipeline

The diffusion implementation is under [`diffusion/`](diffusion), and you can run the pipeline with:

```bash
python diffusion/pipeline.py
```

Outputs for diffusion are as follows:

```text
├─ runs/
│  └─ diffusion/
│     ├─ best_diffusion_metrics.txt
│     ├─ best_test_metrics.txt
│     ├─ diffusion_data.csv
│     ├─ diffusion_training_log.txt
│     ├─ diffusion_testing_log.txt
│     ├─ checkpoints/
│     │  ├─ diffusion_best_model.pt
│     │  └─ diffusion_final_model.pt
│     └─ examples/
│        ├─ samples.png
│        └─ samples_epoch_X
```

### Metrics and Output Files for Diffusion

**`best_diffusion_metrics.txt`**
Contains the highest validation PSNR, SSIM, and MAE achieved during training.

**`best_test_metrics.txt`**
Records the batch number and metrics for the best-performing batch during test set evaluation.

**`diffusion_data.csv`**
Epoch-by-epoch validation metrics (PSNR, SSIM, MAE) in CSV format for DataFrame analysis.

**`diffusion_training_log.txt`**
Per-epoch training log containing training loss, validation statistics (PSNR, SSIM, MSE, MAE), and the highest PSNR achieved so far.

**`diffusion_testing_log.txt`**
Per-batch statistics during test evaluation, with average metrics across all batches at the end of the file.

**`checkpoints/diffusion_best_model.pt`**
Model checkpoint from the epoch with the highest validation PSNR.

**`checkpoints/diffusion_final_model.pt`**
Model checkpoint from the final training epoch.

**`examples/samples/`**
Qualitative visualization of the first 8 images from the first test batch.

**`examples/samples_epoch_X/`**
Training progress visualization showing the same 8 images at epoch 1 and every 20 epochs thereafter.

## VAE for CelebA Inpainting

The VAE implementation provides a variational autoencoder baseline for image inpainting on CelebA. This model learns to reconstruct masked regions of celebrity face images by encoding images into a latent space and decoding them back with the missing regions filled in.

### Overview

This VAE-based inpainting model uses a convolutional encoder–decoder architecture to:

1. Encode masked input images into a probabilistic latent representation.
2. Sample from the learned latent distribution.
3. Decode the latent vectors to reconstruct complete images with inpainted regions.

The model is trained using a combination of reconstruction loss and KL divergence to ensure meaningful latent representations.

### Model Architecture

The VAE model architecture is defined in the [`models/`](models) directory:

- [`unet_vae.py`](models/unet_vae.py) – U-Net based VAE implementation with encoder-decoder architecture, skip connections, and optional self-attention mechanisms
- [`vae_loss.py`](models/vae_loss.py) – VAE loss function combining reconstruction loss and KL divergence

The main training script is located at [`scripts/train.py`](scripts/train.py), which provides a command-line interface for training the VAE with various configuration options including pretrained encoders and transfer learning capabilities.

### Training

You can train the VAE either via a script or an interactive notebook:

- Scripted training entry point: see [`pretrain_vae.py`](scripts/pretrain_vae.py) for an example of how to pretrain the VAE.
- Interactive training and experimentation: use the Jupyter notebook:

```bash
jupyter notebook VAE_from_scratch.ipynb
```

(See [`VAE_from_scratch.ipynb`](VAE_from_scratch.ipynb).)

### Outputs

VAE training outputs follow the same general layout as other models and are written under:

```text
runs/vae/{timestamp}/
├── config_snapshot.json    # Training configuration
├── training.log            # Training logs
├── learning_curves.png     # Loss curves visualization
├── history.csv             # Training metrics history
├── checkpoints/
│   ├── best.ckpt          # Best model checkpoint
│   └── last.ckpt          # Latest model checkpoint
└── examples/
    ├── triptych_0001.png  # Input/Output/Ground truth comparisons
    └── examples_grid.png  # Grid of example results
```

### Inference

Example pseudo-code for running inpainting with a trained VAE:

```python
from models.vae import VAE
from masking.mask_generator import MaskGenerator
import torch

# Load model
model = VAE(latent_dim=256)
model.load_state_dict(torch.load('runs/vae/{timestamp}/checkpoints/best.ckpt'))
model.eval()

# Generate mask
mask_gen = MaskGenerator(image_size=128)
mask = mask_gen.generate_random_mask()

# Run inpainting
with torch.no_grad():
    masked_image = image * (1 - mask)
    reconstructed = model(masked_image)
    inpainted = image * (1 - mask) + reconstructed * mask
```

### License

See the repository license for usage terms.

### Acknowledgments

- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) by MMLAB, CUHK.
- PyTorch VAE implementations community.
