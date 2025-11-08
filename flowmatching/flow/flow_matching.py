"""Flow matching formulation for conditional image inpainting.

This module implements the core flow matching algorithm with linear interpolation
paths and velocity field computation for image inpainting tasks.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class FlowMatching:
    """Flow matching formulation for conditional image inpainting.
    
    Implements the flow matching framework with linear interpolation paths.
    The flow matching objective trains a model to predict velocity fields
    that transport noisy masked regions to complete images while conditioning
    on unmasked regions.
    
    Mathematical Framework:
        - Linear path: x_t = (1-t)x₀ + tx₁
        - Velocity: v_t = dx_t/dt = x₁ - x₀
        - Loss: L = E[||m ⊙ (v_pred - v_gt)||²]
    
    where:
        - x₀: Unmasked pixels from image + noise in masked regions
        - x₁: Target complete image (ground truth)
        - m: Binary mask (1 = masked/unknown, 0 = observed)
        - t: Time parameter in [0, 1]
    
    The key insight: We learn to denoise MASKED regions (m=1) while
    conditioning on UNMASKED regions (m=0) which remain as original pixels.
    
    Args:
        None
    
    Example:
        >>> fm = FlowMatching()
        >>> # Sample batch
        >>> images = torch.randn(4, 3, 128, 128)
        >>> masks = torch.randint(0, 2, (4, 1, 128, 128)).float()
        >>>
        >>> # Prepare training batch (creates x₀ with noise in masked regions)
        >>> x_t, t, v_gt, x0 = fm.prepare_training_batch(images, masks)
        >>>
        >>> # x0 has: unmasked pixels from image, masked pixels from noise
        >>> # x_t interpolates between x0 and complete image
        >>> # v_gt is the velocity field to learn
    """
    
    def __init__(self):
        """Initialize FlowMatching."""
        pass
    
    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Linearly interpolate between x0 and x1 at time t.
        
        Implements the linear interpolation path:
            x_t = (1 - t) * x₀ + t * x₁
        
        Args:
            x0: Starting point (noise in masked regions, image in unmasked), shape [B, C, H, W]
            x1: End point (complete image), shape [B, C, H, W]
            t: Time values in [0, 1], shape [B] or [B, 1]
        
        Returns:
            Interpolated tensor x_t of shape [B, C, H, W]
        
        Note:
            - At t=0: x_t = x₀ (noise in masked regions, original in unmasked)
            - At t=1: x_t = x₁ (complete image)
            - For t in (0,1): x_t is a linear combination
        """
        # Ensure t has the right shape for broadcasting
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        elif t.dim() == 2:
            t = t.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # Linear interpolation
        x_t = (1 - t) * x0 + t * x1
        
        return x_t
    
    def compute_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> torch.Tensor:
        """Compute the ground truth velocity field.
        
        For linear interpolation paths, the velocity is constant:
            v_t = dx_t/dt = x₁ - x₀
        
        Args:
            x0: Starting point (noise in masked regions, image in unmasked), shape [B, C, H, W]
            x1: End point (complete image), shape [B, C, H, W]
        
        Returns:
            Velocity field of shape [B, C, H, W]
        
        Note:
            The velocity represents the direction and magnitude of change
            needed to transform x₀ into x₁. In masked regions, this is
            denoising; in unmasked regions, this is zero (no change needed).
        """
        return x1 - x0
    
    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device = None
    ) -> torch.Tensor:
        """Sample random timesteps uniformly from [0, 1].
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on (default: None, uses CPU)
        
        Returns:
            Random timesteps of shape [B]
        """
        if device is None:
            device = torch.device('cpu')
        
        return torch.rand(batch_size, device=device)
    
    def compute_loss(
        self,
        v_pred: torch.Tensor,
        v_gt: torch.Tensor,
        mask: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Compute the flow matching loss on masked regions only.
        
        Loss function:
            L = E[||m ⊙ (v_pred - v_gt)||²]
        
        where ⊙ denotes element-wise multiplication. The loss is computed
        only on the masked regions to focus learning on inpainting.
        
        Args:
            v_pred: Predicted velocity field, shape [B, C, H, W]
            v_gt: Ground truth velocity field, shape [B, C, H, W]
            mask: Binary mask (1 = masked, 0 = observed), shape [B, 1, H, W]
            reduction: Loss reduction method ('mean', 'sum', or 'none')
        
        Returns:
            Scalar loss value if reduction is 'mean' or 'sum',
            otherwise tensor of shape [B, C, H, W]
        
        Note:
            The mask ensures we only compute loss on regions that need
            to be inpainted, not on the observed pixels.
        """
        # Compute squared error
        squared_error = (v_pred - v_gt) ** 2
        
        # Apply mask (only compute loss on masked regions)
        masked_error = mask * squared_error
        
        # Apply reduction
        if reduction == 'mean':
            # Average over all elements, accounting for mask
            # This gives equal weight to each masked pixel
            return masked_error.sum() / (mask.sum() + 1e-8)
        elif reduction == 'sum':
            return masked_error.sum()
        elif reduction == 'none':
            return masked_error
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def prepare_training_batch(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare a training batch for flow matching.
        
        This is a convenience function that:
        1. Creates x₀ by replacing masked pixels with noise, keeping unmasked pixels
        2. Samples timesteps if not provided
        3. Interpolates to get x_t
        4. Computes ground truth velocity
        
        Args:
            images: Complete images (x₁), shape [B, C, H, W]
            masks: Binary masks (1 = masked, 0 = observed), shape [B, 1, H, W]
            t: Optional timesteps, shape [B]. If None, samples uniformly.
        
        Returns:
            Tuple of (x_t, t, v_gt, x0):
                - x_t: Interpolated images, shape [B, C, H, W]
                - t: Timesteps, shape [B]
                - v_gt: Ground truth velocity, shape [B, C, H, W]
                - x0: Starting point with noise in masked regions, shape [B, C, H, W]
        
        Example:
            >>> fm = FlowMatching()
            >>> images = torch.randn(4, 3, 128, 128)
            >>> masks = torch.randint(0, 2, (4, 1, 128, 128)).float()
            >>> x_t, t, v_gt, x0 = fm.prepare_training_batch(images, masks)
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Sample timesteps if not provided
        if t is None:
            t = self.sample_timesteps(batch_size, device)
        
        # Create x₀: unmasked pixels from image, masked pixels from noise
        # This is the correct formulation for inpainting flow matching
        noise = torch.randn_like(images)
        x0 = (1 - masks) * images + masks * noise
        
        # Interpolate to get x_t: x_t = (1-t)*x₀ + t*x₁
        x_t = self.interpolate(x0, images, t)
        
        # Compute ground truth velocity: v = x₁ - x₀
        v_gt = self.compute_velocity(x0, images)
        
        return x_t, t, v_gt, x0