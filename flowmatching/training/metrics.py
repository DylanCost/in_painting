"""Evaluation metrics for image inpainting.

This module implements PSNR, SSIM, and other metrics for evaluating
inpainting quality, with support for computing metrics only on masked regions.
"""

import torch
from typing import Dict, Optional, Tuple

from evaluation.metrics import InpaintingMetrics


_METRICS = InpaintingMetrics(device="cpu")


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_val: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) using the shared metrics object."""
    return float(_METRICS.psnr(pred, target, mask))


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
    channel_axis: int = 1,
) -> float:
    """Compute SSIM via the shared InpaintingMetrics instance.

    The ``channel_axis`` parameter is kept for API compatibility but ignored since
    InpaintingMetrics.ssim always expects channel-first tensors.
    """

    return float(_METRICS.ssim(pred, target, mask))


def compute_mae(
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


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_val: float = 1.0,
    data_range: float = 1.0,
    include_mae: bool = True,
) -> Dict[str, float]:
    """Compute multiple evaluation metrics at once.

    Convenience function that computes PSNR, SSIM, and optionally MAE together.

    Args:
        pred: Predicted image, shape [B, C, H, W] or [C, H, W]
        target: Ground truth image, same shape as pred
        mask: Optional binary mask (1 = compute metric, 0 = ignore)
        max_val: Maximum pixel value for PSNR computation
        data_range: Data range for SSIM computation
        include_mae: If True, also compute MAE and include it in the result

    Returns:
        Dictionary with keys:
            - 'psnr': PSNR value in dB
            - 'ssim': SSIM value between -1 and 1
            - 'mae': Mean absolute error (only if include_mae is True)

    Example:
        >>> pred = torch.randn(4, 3, 128, 128)
        >>> target = torch.randn(4, 3, 128, 128)
        >>> mask = torch.ones(4, 1, 128, 128)
        >>> metrics = compute_metrics(pred, target, mask)
        >>> print(f"PSNR: {metrics['psnr']:.2f} dB")
        >>> print(f"SSIM: {metrics['ssim']:.4f}")
        >>> print(f"MAE: {metrics['mae']:.6f}")
    """
    psnr = compute_psnr(pred, target, mask, max_val)
    ssim_val = compute_ssim(pred, target, mask, data_range)

    result = {"psnr": psnr, "ssim": ssim_val}

    if include_mae:
        result["mae"] = compute_mae(pred, target, mask)

    return result


def denormalize_image(
    image: torch.Tensor,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
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
