import torch
import numpy as np
from typing import Optional, Tuple
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3
import lpips


class InpaintingMetrics:
    """Metrics for evaluating inpainting quality."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # LPIPS for perceptual distance
        self.lpips = None
        
        # Inception model for FID
        # self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        # self.inception.eval()
    
    def psnr(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""

        me = ((pred-target) ** 2) * mask
        total_mask_pixels = mask.sum() * pred.shape[1]
        mse = me.sum() / total_mask_pixels
    
        maximum = 2
        psnr = 10 * torch.log10(maximum**2 / mse).item()
        return psnr
    
    def ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Calculate Structural Similarity Index.
        
        Args:
            pred: Predicted tensor in [B, C, H, W] format.
            target: Target tensor in [B, C, H, W] format.
            mask: Optional binary mask in [B, 1, H, W] or [B, H, W] format.
                  If provided, SSIM is averaged only over masked pixels.
        """
        from skimage.metrics import structural_similarity
        
        # Convert to NumPy with channel-last format: [B, H, W, C]
        pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
        target_np = target.detach().cpu().numpy().transpose(0, 2, 3, 1)
        
        mask_np = None
        if mask is not None:
            # Bring mask to CPU NumPy, ensure shape [B, H, W], and boolean
            mask_np = mask.detach().cpu().numpy()
            # Accept [B, 1, H, W] or [B, H, W]
            if mask_np.ndim == 4:
                # Collapse channel dimension
                mask_np = mask_np[:, 0]
            mask_np = mask_np.astype(bool)
        
        ssim_values = []
        for i, (p, t) in enumerate(zip(pred_np, target_np)):
            # full=True returns (scalar_ssim, ssim_map[H, W])
            _, ssim_map = structural_similarity(
                p,
                t,
                channel_axis=-1,
                data_range=2.0,
                full=True,
            )
            
            if mask_np is not None:
                m = mask_np[i]
                valid = m.sum()
                if valid > 0:
                    ssim_val = float(ssim_map[m].mean())
                else:
                    # Fallback: no masked pixels for this sample; use global mean
                    ssim_val = float(ssim_map.mean())
            else:
                ssim_val = float(ssim_map.mean())
            
            ssim_values.append(ssim_val)
        
        return float(np.mean(ssim_values))
    
    def compute_mae(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute Mean Absolute Error (MAE).

        MAE measures the average absolute difference between prediction and target.
        Lower values indicate better reconstruction quality.

        Args:
            pred: Predicted image, shape [B, C, H, W] or [C, H, W]
            target: Ground truth image, same shape as pred
            mask: Optional binary mask (1 = compute metric, 0 = ignore),
                shape [B, 1, H, W] or [1, H, W]. If None, computes on full image.

        Returns:
            MAE value as a float.
        """
        # Ensure tensors are on the same device
        if pred.device != target.device:
            target = target.to(pred.device)

        abs_error = torch.abs(pred - target)

        if mask is not None:
            if mask.device != pred.device:
                mask = mask.to(pred.device)

            # Expand mask to match image channels if needed
            if mask.shape[1] == 1 and pred.shape[1] > 1:
                mask = mask.expand_as(pred)

            mae = (abs_error * mask).sum() / (mask.sum() + 1e-8)
        else:
            mae = abs_error.mean()

        return mae.item()
    
    def lpips_distance(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate LPIPS perceptual distance."""
        if self.lpips is None:
            self.lpips = lpips.LPIPS(net='alex').to(self.device)
        with torch.no_grad():
            distance = self.lpips(pred, target)
        return distance.mean().item()
    
    def calculate_fid(self, real_features: np.ndarray, fake_features: np.ndarray) -> float:
        """Calculate Fréchet Inception Distance."""
        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        diff = mu_real - mu_fake
        
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
        return fid
    
    def extract_inception_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from Inception model for FID calculation."""
        with torch.no_grad():
            # Resize to inception input size
            images = adaptive_avg_pool2d(images, (299, 299))
            
            # Get features
            features = self.inception(images)
            
        return features.cpu().numpy()