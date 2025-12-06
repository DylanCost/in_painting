"""
Diffusion Model Evaluation and Reverse Sampling for Image Inpainting

This module implements the reverse diffusion process (DDPM sampling) for generating inpainted
images and provides comprehensive evaluation functionality for trained diffusion models. The core
sampling function performs iterative denoising from complete noise back to clean images, while
the evaluation pipeline computes quantitative metrics and generates visual comparisons on test data.

The reverse diffusion process follows the DDPM sampling equations, starting from x_T (pure noise
in masked regions) and iteratively denoising through timesteps T-1, T-2, ..., 1, 0 using the
model's learned noise predictions. At each step, the algorithm computes the posterior mean using
the ε-parameterization and adds calibrated Gaussian noise (except at t=0). Known pixels outside
the mask are preserved throughout sampling to ensure seamless blending between inpainted and
original regions.

Key functions include: (1) sample_ddpm - performs the complete reverse diffusion process given
a noised input, returning the final denoised reconstruction; (2) run_evaluation - orchestrates
comprehensive testing on the full test set, computing PSNR, SSIM, MSE, and MAE metrics while
saving visualizations and detailed logs; (3) load_model - loads trained checkpoints from the
standard checkpoint directory; (4) evaluate - main entry point that sets up datasets, model,
and scheduler, then runs full evaluation with deterministic masks for reproducibility.
"""

# diffusion_evaluate.py
import os
import sys

import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.celeba_dataset import CelebADataset
from evaluation.metrics import InpaintingMetrics
from masking.mask_generator import MaskGenerator
from noise_scheduler_config import NoiseConfig
from scripts.set_seed import set_seed
from unet_diffusion import NoiseScheduler, UNetDiffusion


@torch.no_grad()
def sample_ddpm(model, scheduler, x_t, mask, num_timesteps=None):
    """
    Perform DDPM reverse sampling for inpainting.
    Starts from pre-noised masked input (x_t).

    Args:
        model: trained UNet predicting noise ε_θ(x_t, t)
        scheduler: NoiseScheduler instance
        x_t: pre-noised masked image [B, 3, H, W]
        mask: binary mask [B, 1, H, W] (1 = region to inpaint)
        num_timesteps: optional override for number of reverse steps

    Returns:
        x_0_pred: final denoised reconstruction [B, 3, H, W]
    """
    device = x_t.device
    B = x_t.size(0)
    T = scheduler.num_timesteps if num_timesteps is None else num_timesteps

    # make a copy to avoid overwriting the input tensor
    mask_3c = mask.repeat(1, x_t.size(1), 1, 1)
    x_t = x_t.clone()

    # Keeps clean areas; masked areas are 0
    x_orig_unmasked = x_t * (1 - mask_3c)

    for t_step in reversed(range(T)):
        t = torch.full((B,), t_step, device=device, dtype=torch.long)

        # predict noise ε
        eps_pred = model(x_t, t, mask)

        if t_step > 0:
            z = torch.randn_like(x_t)
        else:
            z = torch.zeros_like(x_t)

        # gather scalar coefficients
        alpha_t = scheduler.alphas[t]              # [B]
        alpha_bar_t = scheduler.alpha_bars[t]      # [B]

        beta_t = scheduler.betas[t]
        sigma_t = torch.sqrt(beta_t)

        # mean using ε-parameterization (Eq. 12)
        # Coefficient 1: 1/√(α_t)
        coef1 = 1.0 / torch.sqrt(alpha_t)  # [B]
        coef1 = coef1[:, None, None, None]  # [B, 1, 1, 1]

        # Coefficient 2: (1 - α_t) / √(1 - ᾱ_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)  # [B]
        coef2 = coef2[:, None, None, None]  # [B, 1, 1, 1]

        # Noise term
        noise_term = coef2 * eps_pred  # [B, 1, 1, 1] * [B, 3, H, W] = [B, 3, H, W]

        # Subtract from current x_t
        inner = x_t - noise_term  # [B, 3, H, W] - [B, 3, H, W] = [B, 3, H, W]

        # Final mean
        x_mean = coef1 * inner  # [B, 1, 1, 1] * [B, 3, H, W] = [B, 3, H, W]
        x_t_minus_1 = x_mean + sigma_t[:, None, None, None] * z

        x_t = x_t_minus_1 * mask_3c + x_orig_unmasked
    return x_t


def run_evaluation(model, test_loader, noise_scheduler, mask_generator, device):
    """
    Run comprehensive evaluation of diffusion model inpainting performance on test set.
    
    Performs full DDPM sampling (1000 timesteps) starting from complete noise in masked
    regions, then computes reconstruction quality metrics comparing inpainted results to
    ground truth images. Saves visualizations, per-batch logs, and summary statistics.
    
    Args:
        model: Trained diffusion model (UNet) in eval mode.
        test_loader: DataLoader yielding batches with 'image' and 'filename' keys.
        noise_scheduler: NoiseScheduler defining the diffusion process (forward/reverse).
        mask_generator: MaskGenerator producing deterministic masks from filenames.
        device: torch.device for computation ('cuda' or 'cpu').
        save_dir: Optional save directory (unused; defaults to ./results/).
    
    Returns:
        dict: Summary statistics with keys:
            - 'psnr': Average Peak Signal-to-Noise Ratio (higher is better)
            - 'ssim': Average Structural Similarity Index (higher is better, range [0,1])
            - 'mse': Average Mean Squared Error (lower is better)
            - 'mae': Average Mean Absolute Error (lower is better)
    
    Side Effects:
        Creates/writes to:
        - runs/diffusion/examples/samples.png: Visual comparison grid (original | masked input | inpainted)
        - runs/diffusion/examples/best_test_metrics.txt: Best per-batch scores across test set
        - runs/diffusion/diffusion_testing_log.txt: Per-batch metrics for all test samples
    
    Notes:
        - All metrics are computed over the full image (not masked region only)
        - First batch (up to 8 images) is saved for visual inspection
        - Masks are deterministically generated based on input filenames
        - Uses complete denoising (T=999 → T=0) for highest quality results
    """

    model.eval()
    
    # Go up one directory from diffusion/ to project root, then create runs/diffusion/
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "diffusion")
    save_dir = os.path.join(base_dir, "examples")
    log_dir = base_dir
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "diffusion_testing_log.txt")
    
    # Create/overwrite log file
    with open(log_file, 'w') as f:
        f.write("Testing Log\n")
        f.write("=" * 60 + "\n\n")
    
    # Initialize metrics calculator
    metrics_calc = InpaintingMetrics(device=device)
    
    # Collect metrics
    all_psnr = []
    all_ssim = []
    all_mse = []
    all_mae = []
    
    print("\nStarting evaluation...")

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        images = batch['image'].to(device)
        filenames = batch['filename']
        
        B, C, H, W = images.shape
        
        # Generate deterministic masks
        masks = mask_generator.generate_for_filenames(
            filenames=filenames,
            shape=(1, H, W)
        ).to(device)
        
        # Add full noise to masked regions (start from complete noise)
        t = torch.full((B,), noise_scheduler.num_timesteps - 1, device=device)

        noisy_images, _ = noise_scheduler.add_noise(images, t, masks)

        # Denoise using DDPM sampling
        inpainted = sample_ddpm(model, noise_scheduler, noisy_images, masks, num_timesteps=1000)
        
        # Compute all metrics on full images
        psnr_val = metrics_calc.psnr(inpainted, images, masks)
        ssim_val = metrics_calc.ssim(inpainted, images, masks)
        mae_val = metrics_calc.compute_mae(inpainted, images, masks)
        
        # Compute MSE and MAE
        mse_val = torch.mean((inpainted - images) ** 2).item()
        
        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)
        all_mse.append(mse_val)
        all_mae.append(mae_val)

        # Log batch metrics
        with open(log_file, 'a') as f:
            f.write(f"Batch {batch_idx + 1}\n")
            f.write(f"PSNR: {psnr_val:.4f}\n")
            f.write(f"SSIM: {ssim_val:.4f}\n")
            f.write(f"MSE: {mse_val:.6f}\n")
            f.write(f"MAE: {mae_val:.6f}\n")
            f.write("-" * 60 + "\n\n")
        
        # Save first batch for visualization
        if batch_idx == 0:
            # Create masked input: show original with black mask region
            masked_input = images * (1 - masks)  # Zero out masked region
            
            comparison = torch.cat([
                images[:8],           # Row 1: Original
                masked_input[:8],     # Row 2: Input with black mask
                inpainted[:8]         # Row 3: Inpainted result
            ], dim=0)
            
            vutils.save_image(
                comparison,
                os.path.join(save_dir, 'samples.png'),
                nrow=8,
                normalize=True,
                value_range=(-1, 1)
            )
            print(f"\n✅ Saved sample images to {save_dir}/samples.png")
    
    summary = {
        'psnr': np.mean(all_psnr),
        'ssim': np.mean(all_ssim),
        'mse': np.mean(all_mse),
        'mae': np.mean(all_mae)
    }
    
    # Log summary statistics
    with open(log_file, 'a') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Average PSNR: {summary['psnr']:.4f}\n")
        f.write(f"Average SSIM: {summary['ssim']:.4f}\n")
        f.write(f"Average MSE: {summary['mse']:.6f}\n")
        f.write(f"Average MAE: {summary['mae']:.6f}\n")
        f.write("=" * 60 + "\n")
    
    # Compute best metrics
    best_psnr = max(all_psnr) if all_psnr else float('nan')
    best_ssim = max(all_ssim) if all_ssim else float('nan')
    best_mse  = min(all_mse)  if all_mse  else float('nan')
    best_mae  = min(all_mae)  if all_mae  else float('nan')

    # Save best metrics
    best_file = os.path.join(base_dir, "best_test_metrics.txt") 
    with open(best_file, 'w') as f:
        f.write("BEST EVALUATION METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best PSNR: {best_psnr:.4f}\n")
        f.write(f"Best SSIM: {best_ssim:.4f}\n")
        f.write(f"Lowest MSE: {best_mse:.6f}\n")
        f.write(f"Lowest MAE: {best_mae:.6f}\n")
        f.write("=" * 60 + "\n")

    print(f"\n🏆 Best evaluation metrics saved to {best_file}")

    return summary

def load_model(model, device):
    """
    Load the best model from runs/diffusion/checkpoints/diffusion_best_model.pt
    """
    # Go up one directory from diffusion/ to project root, then look in runs/diffusion/checkpoints
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(
        os.path.dirname(current_dir), 
        'runs', 
        'diffusion', 
        'checkpoints', 
        'diffusion_best_model.pt'
    )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model

def evaluate():
    """Evaluate trained diffusion model on test set."""
    set_seed()
    # Load config
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

    model = load_model(model, device)
    
    model.eval()
    
    # Create noise scheduler
    noise_config = NoiseConfig()
    noise_scheduler = NoiseScheduler(
        num_timesteps=noise_config.num_timesteps,
        beta_start=noise_config.beta_start,
        beta_end=noise_config.beta_end,
        schedule_type=noise_config.schedule_type
    ).to(device)
    
    # Create deterministic mask generator
    mask_generator = MaskGenerator.for_eval(config.mask)
    
    # Run evaluation
    results = run_evaluation(
        model=model,
        test_loader=test_loader,
        noise_scheduler=noise_scheduler,
        mask_generator=mask_generator,
        device=device
    )
    
    return results


if __name__ == '__main__':
    results = evaluate()
    print(f"PSNR values: {results['psnr']}")
    print(f"SSIM values: {results['ssim']}")
    print(f"MSE values:  {results['mse']}")
    print(f"MAE values:  {results['mae']}")