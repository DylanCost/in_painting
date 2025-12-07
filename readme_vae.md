# VAE for CelebA Inpainting

A Variational Autoencoder (VAE) implementation for image inpainting on the CelebA dataset. This model learns to reconstruct masked regions of celebrity face images by encoding images into a latent space and decoding them back with the missing regions filled in.

## Overview

This VAE-based inpainting model uses a convolutional encoder-decoder architecture to:
1. Encode masked input images into a probabilistic latent representation
2. Sample from the learned latent distribution
3. Decode the latent vectors to reconstruct complete images with inpainted regions

The model is trained using a combination of reconstruction loss and KL divergence to ensure meaningful latent representations.

## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch
- torchvision
- numpy
- Pillow
- matplotlib
- tqdm
- wandb (optional, for experiment tracking)

## Project Structure

```
├── config/                  # Configuration files
├── data/
│   └── celeba_dataset.py   # CelebA dataset loader
├── evaluation/
│   └── metrics.py          # Evaluation metrics (PSNR, SSIM, etc.)
├── losses/                  # Loss function implementations
├── masking/
│   └── mask_generator.py   # Mask generation utilities
├── models/
│   └── vae.py              # VAE model architecture
├── training/               # Training utilities
├── scripts/                # Training and evaluation scripts
├── runs/                   # Output directory for experiments
│   └── vae/
│       └── {timestamp}/
│           ├── config_snapshot.json
│           ├── training.log
│           ├── learning_curves.png
│           ├── checkpoints/
│           │   ├── best.ckpt
│           │   └── last.ckpt
│           └── examples/
├── VAE_from_scratch.ipynb  # Interactive notebook for VAE training
└── requirements.txt
```

## Dataset

This project uses the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (CelebFaces Attributes Dataset), which contains over 200,000 celebrity face images with various attributes.

### Dataset Setup

1. Download the CelebA dataset from the [official source](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. Extract the images to your data directory
3. Update the dataset path in your configuration file

## Model Architecture

The VAE consists of two main components:

### Encoder
- Series of convolutional layers with batch normalization and ReLU activation
- Downsamples input images progressively
- Outputs mean (μ) and log-variance (log σ²) for the latent distribution

### Decoder
- Takes samples from the latent distribution
- Series of transposed convolutional layers
- Upsamples back to the original image resolution
- Final sigmoid activation for pixel values in [0, 1]

### Latent Space
- Implements the reparameterization trick: z = μ + σ × ε, where ε ~ N(0, I)
- Enables backpropagation through the sampling operation

## Training

### Using the Training Script

```bash
python -m training.train_vae --epochs 100 --batch_size 64 --latent_dim 32 --lr 1e-4
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--epochs` | Number of training epochs | 100 |
| `--batch_size` | Training batch size | 64 |
| `--latent_dim` | Dimension of latent space | 32 |
| `--lr` | Learning rate | 1e-4 |
| `--beta` | KL divergence weight (β-VAE) | 1.0 |
| `--image_size` | Input image resolution | 128 |
| `--num_eval_samples` | Samples for evaluation | 1024 |
| `--num_example_images` | Example images to save | 8 |

### Using the Notebook

For interactive training and experimentation, use the provided Jupyter notebook:

```bash
jupyter notebook VAE_from_scratch.ipynb
```

## Loss Function

The VAE is trained with the Evidence Lower Bound (ELBO) loss:

```
L = L_reconstruction + β × L_KL
```

Where:
- **L_reconstruction**: Measures how well the model reconstructs the original image (MSE or BCE)
- **L_KL**: KL divergence between the learned latent distribution and a standard normal prior
- **β**: Weighting factor for the KL term (β=1 for standard VAE, β>1 for β-VAE)

## Masking

The model supports various mask types for training:

- **Random rectangles**: Rectangular regions of random size and position
- **Center masks**: Fixed central region masking
- **Free-form masks**: Irregular brush stroke-like masks
- **Custom masks**: User-defined mask patterns

Masks are generated using the `masking/mask_generator.py` module.

## Evaluation

The model is evaluated using standard image quality metrics:

| Metric | Description |
|--------|-------------|
| **PSNR** | Peak Signal-to-Noise Ratio |
| **SSIM** | Structural Similarity Index |
| **MSE** | Mean Squared Error |
| **LPIPS** | Learned Perceptual Image Patch Similarity |

### Running Evaluation

```bash
python -m evaluation.evaluate_vae --checkpoint runs/vae/{timestamp}/checkpoints/best.ckpt --num_samples 1024
```

## Output Structure

Training outputs are saved in the following directory structure:

```
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

## Inference

To run inference on new images:

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

## Common Modules

This VAE implementation shares the following modules with other models in the repository:

- `data/celeba_dataset.py` - CelebA dataset loading and preprocessing
- `evaluation/metrics.py` - Standard evaluation metrics
- `masking/mask_generator.py` - Mask generation utilities
- `losses/` - Loss function implementations

## Hyperparameter Tuning

Key hyperparameters to tune for best results:

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| `latent_dim` | 32-512 | Higher values capture more detail but may overfit |
| `beta` | 0.1-4.0 | Controls reconstruction vs. regularization trade-off |
| `lr` | 1e-5 to 1e-3 | Use learning rate scheduling for best results |
| `batch_size` | 32-128 | Larger batches provide more stable gradients |

## Tips for Training

1. **Start with a lower β value** (e.g., 0.1) and gradually increase to 1.0 for better reconstruction quality
2. **Use learning rate warmup** for the first few epochs
3. **Monitor reconstruction loss separately** from KL divergence to diagnose training issues
4. **Validate on held-out masks** different from training masks to test generalization

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Blurry reconstructions | Decrease β, increase latent dimension, or add perceptual loss |
| KL collapse | Use KL annealing, free bits, or cyclical scheduling |
| Training instability | Reduce learning rate, use gradient clipping |
| Poor mask filling | Ensure masks are applied consistently during training |

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{celeba_vae_inpainting,
  title={VAE for CelebA Inpainting},
  author={DylanCost},
  year={2024},
  url={https://github.com/DylanCost/in_painting}
}
```

## License

See the repository license for usage terms.

## Acknowledgments

- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) by MMLAB, CUHK
- PyTorch VAE implementations community
