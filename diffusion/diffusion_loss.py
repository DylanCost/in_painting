"""
This class holds the logic for the Diffusion models Loss function, calculated only on the differences
within the masked portion of the image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionLoss(nn.Module):
    """
    Loss function for diffusion model training.
    Computes MSE loss only on masked regions.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, predicted_noise, target_noise, mask):
        """
        Compute MSE normalized by masked region size.
        """
        # Ensure mask is broadcastable: (B, 1, H, W)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        mask = mask.float()

        # Apply mask to squared error
        se = ((predicted_noise - target_noise) ** 2) * mask

        # Count masked pixel *channels
        num = mask.sum() * predicted_noise.shape[1]

        if num == 0:
            return torch.tensor(0.0, device=predicted_noise.device)

        # Mean only over masked region
        loss = se.sum() / (num + 1e-8)

        return loss
