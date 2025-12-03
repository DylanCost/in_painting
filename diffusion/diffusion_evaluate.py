# diffusion_evaluate.py
import torch
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from noise_scheduler_config import NoiseConfig
from data.celeba_dataset import CelebADataset
from unet_diffusion import UNetDiffusion, NoiseScheduler
from masking.mask_generator import MaskGenerator
from evaluation.metrics import InpaintingMetrics
from scripts.set_seed import set_seed


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

    save_dir = "./runs/eval_debug"
    os.makedirs(save_dir, exist_ok=True)

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


def run_evaluation(model, test_loader, noise_scheduler, mask_generator, device, save_dir=None):
    """
    Run comprehensive evaluation on test set.
    """

    model.eval()
    
    # Create save directory: ./results in CURRENT directory
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)

    # Create logs directory and log file
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
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
    #out_dir = "./runs/eval_debug"
    # os.makedirs(out_dir, exist_ok=True)

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        if batch_idx > 2:
            break
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
        ssim_val = metrics_calc.ssim(inpainted, images)
        
        # Compute MSE and MAE
        mse_val = torch.mean((inpainted - images) ** 2).item()
        mae_val = torch.mean(torch.abs(inpainted - images)).item()
        
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
    
    # Compute summary statistics
    # Compute summary statistics
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
    best_file = os.path.join(log_dir, "best_evaluation_metrics.txt")
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

def load_model(model, config, device):
    # Look in current directory's weights folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, 'weights', 'diffusion_final_model.pt')
    
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
    
    # print(f"Test dataset size: {len(test_dataset)}")
    
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
    
    # Create deterministic mask generator
    mask_generator = MaskGenerator.for_eval(config.mask)
    
    # Run evaluation
    results = run_evaluation(
        model=model,
        test_loader=test_loader,
        noise_scheduler=noise_scheduler,
        mask_generator=mask_generator,
        device=device,
        save_dir='results/diffusion'
    )
    
    return results


if __name__ == '__main__':
    results = evaluate()
    print(f"PSNR values: {results['psnr']}")
    print(f"SSIM values: {results['ssim']}")
    print(f"MSE values:  {results['mse']}")
    print(f"MAE values:  {results['mae']}")