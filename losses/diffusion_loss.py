import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionLoss(nn.Module):
    """
    Loss function for diffusion model training.
    Computes MSE loss only on masked regions.
    Batch Normalization
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, predicted_noise, target_noise, mask):
        """
        Compute MSE normalized by masked region size.
        """
        # Compute MSE over entire image (most errors are 0 outside mask)
        mse = F.mse_loss(predicted_noise, target_noise, reduction='sum')
        
        # Count number of pixels in mask (across all channels), gaussian noise on 3 channels
        num_masked_pixels = mask.sum() * predicted_noise.shape[1]  # multiply by 3 for RGB
        
        # Normalize by masked area
        loss = mse / (num_masked_pixels + 1e-8)
        
        return loss

class DiffusionLossPerImage(nn.Module):
    """
    Loss function for diffusion model training.
    Computes MSE loss only on masked regions with per-image normalization.

    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, predicted_noise, target_noise, mask):
        """
        Args:
            predicted_noise: Model prediction [B, 3, H, W]
            target_noise: Ground truth noise [B, 3, H, W]
            mask: Binary mask [B, 1, H, W] (1=inpaint, 0=keep)
            
        Returns:
            loss: Scalar loss value
        """
        # MSE loss per pixel [B, 3, H, W]
        loss_per_pixel = (predicted_noise - target_noise) ** 2
        
        # Apply mask [B, 3, H, W]
        masked_loss = loss_per_pixel * mask
        
        # Sum over spatial and channel dimensions for each image [B]
        loss_per_image = masked_loss.sum(dim=(1, 2, 3))
        
        # Count masked pixels for each image [B]
        pixels_per_image = mask.sum(dim=(1, 2, 3)) * predicted_noise.shape[1]  # multiply by 3 for RGB
        
        # Normalize each image's loss by its number of masked pixels [B]
        normalized_loss_per_image = loss_per_image / (pixels_per_image + 1e-8)
        
        # Average across batch to get final scalar loss
        loss = normalized_loss_per_image.mean()
        
        return loss