"""Evaluation metrics for image inpainting.

This module implements PSNR, SSIM, and other metrics for evaluating
inpainting quality, with support for computing metrics only on masked regions.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from skimage.metrics import structural_similarity as ssim


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_val: float = 1.0
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR).
    
    PSNR measures the ratio between the maximum possible signal power
    and the power of corrupting noise. Higher values indicate better quality.
    
    Formula:
        PSNR = 10 * log10(MAX^2 / MSE)
    
    where MAX is the maximum possible pixel value and MSE is the mean
    squared error.
    
    Args:
        pred: Predicted image, shape [B, C, H, W] or [C, H, W]
        target: Ground truth image, same shape as pred
        mask: Optional binary mask (1 = compute metric, 0 = ignore),
              shape [B, 1, H, W] or [1, H, W]. If None, computes on full image.
        max_val: Maximum possible pixel value (default: 1.0 for normalized images)
    
    Returns:
        PSNR value in dB. Returns inf if MSE is 0 (perfect reconstruction).
    
    Example:
        >>> pred = torch.randn(4, 3, 128, 128)
        >>> target = torch.randn(4, 3, 128, 128)
        >>> mask = torch.ones(4, 1, 128, 128)
        >>> psnr = compute_psnr(pred, target, mask)
        >>> print(f"PSNR: {psnr:.2f} dB")
    
    Note:
        - For images normalized to [-1, 1], use max_val=2.0
        - For images in [0, 1], use max_val=1.0
        - For images in [0, 255], use max_val=255.0
    """
    # Ensure tensors are on the same device
    if pred.device != target.device:
        target = target.to(pred.device)
    
    # Compute squared error
    mse = (pred - target) ** 2
    
    # Apply mask if provided
    if mask is not None:
        if mask.device != pred.device:
            mask = mask.to(pred.device)
        
        # Expand mask to match image channels if needed
        if mask.shape[1] == 1 and pred.shape[1] > 1:
            mask = mask.expand_as(pred)
        
        # Compute MSE only on masked regions
        mse = (mse * mask).sum() / (mask.sum() + 1e-8)
    else:
        # Compute MSE on full image
        mse = mse.mean()
    
    # Convert to scalar
    mse = mse.item()
    
    # Handle perfect reconstruction
    if mse == 0:
        return float('inf')
    
    # Compute PSNR
    psnr = 10 * np.log10(max_val ** 2 / mse)
    
    return psnr


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
    channel_axis: int = 1
) -> float:
    """Compute Structural Similarity Index (SSIM).
    
    SSIM measures the structural similarity between two images, considering
    luminance, contrast, and structure. Values range from -1 to 1, with 1
    indicating perfect similarity.
    
    Args:
        pred: Predicted image, shape [B, C, H, W] or [C, H, W]
        target: Ground truth image, same shape as pred
        mask: Optional binary mask (1 = compute metric, 0 = ignore),
              shape [B, 1, H, W] or [1, H, W]. If None, computes on full image.
        data_range: Range of the data (max - min). For normalized images in [0, 1],
                   use 1.0. For [-1, 1], use 2.0.
        channel_axis: Axis of the channel dimension (default: 1 for PyTorch format)
    
    Returns:
        SSIM value between -1 and 1. Higher is better.
    
    Example:
        >>> pred = torch.randn(4, 3, 128, 128)
        >>> target = torch.randn(4, 3, 128, 128)
        >>> mask = torch.ones(4, 1, 128, 128)
        >>> ssim_val = compute_ssim(pred, target, mask)
        >>> print(f"SSIM: {ssim_val:.4f}")
    
    Note:
        - This function uses scikit-image's SSIM implementation
        - For masked computation, we apply the mask before computing SSIM
        - SSIM is computed per-channel and then averaged
    """
    # Convert to numpy and move to CPU
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Handle batch dimension
    if pred_np.ndim == 4:
        # Average over batch
        ssim_values = []
        for i in range(pred_np.shape[0]):
            pred_i = pred_np[i]
            target_i = target_np[i]
            
            # Apply mask if provided
            if mask is not None:
                mask_np = mask[i].detach().cpu().numpy()
                # Expand mask to match channels
                if mask_np.shape[0] == 1:
                    mask_np = np.repeat(mask_np, pred_i.shape[0], axis=0)
                
                # Apply mask (set non-masked regions to 0)
                pred_i = pred_i * mask_np
                target_i = target_i * mask_np
            
            # Compute SSIM
            ssim_val = ssim(
                target_i,
                pred_i,
                data_range=data_range,
                channel_axis=0  # Channel is first dimension after batch
            )
            ssim_values.append(ssim_val)
        
        return float(np.mean(ssim_values))
    else:
        # Single image
        if mask is not None:
            mask_np = mask.detach().cpu().numpy()
            if mask_np.shape[0] == 1:
                mask_np = np.repeat(mask_np, pred_np.shape[0], axis=0)
            pred_np = pred_np * mask_np
            target_np = target_np * mask_np
        
        ssim_val = ssim(
            target_np,
            pred_np,
            data_range=data_range,
            channel_axis=0
        )
        return float(ssim_val)


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_val: float = 1.0,
    data_range: float = 1.0
) -> Dict[str, float]:
    """Compute multiple evaluation metrics at once.
    
    Convenience function that computes PSNR and SSIM together.
    
    Args:
        pred: Predicted image, shape [B, C, H, W] or [C, H, W]
        target: Ground truth image, same shape as pred
        mask: Optional binary mask (1 = compute metric, 0 = ignore)
        max_val: Maximum pixel value for PSNR computation
        data_range: Data range for SSIM computation
    
    Returns:
        Dictionary with keys:
            - 'psnr': PSNR value in dB
            - 'ssim': SSIM value between -1 and 1
    
    Example:
        >>> pred = torch.randn(4, 3, 128, 128)
        >>> target = torch.randn(4, 3, 128, 128)
        >>> mask = torch.ones(4, 1, 128, 128)
        >>> metrics = compute_metrics(pred, target, mask)
        >>> print(f"PSNR: {metrics['psnr']:.2f} dB")
        >>> print(f"SSIM: {metrics['ssim']:.4f}")
    """
    psnr = compute_psnr(pred, target, mask, max_val)
    ssim_val = compute_ssim(pred, target, mask, data_range)
    
    return {
        'psnr': psnr,
        'ssim': ssim_val
    }


def denormalize_image(
    image: torch.Tensor,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> torch.Tensor:
    """Denormalize an image from [-1, 1] to [0, 1].
    
    Reverses the normalization applied during preprocessing.
    
    Args:
        image: Normalized image, shape [B, C, H, W] or [C, H, W]
        mean: Mean values used for normalization
        std: Std values used for normalization
    
    Returns:
        Denormalized image in [0, 1] range
    
    Example:
        >>> normalized = torch.randn(4, 3, 128, 128)
        >>> denormalized = denormalize_image(normalized)
        >>> assert denormalized.min() >= 0 and denormalized.max() <= 1
    """
    mean = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std = torch.tensor(std, device=image.device).view(-1, 1, 1)
    
    # Handle batch dimension
    if image.ndim == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    # Denormalize: x = x * std + mean
    denormalized = image * std + mean
    
    # Clamp to [0, 1]
    denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized