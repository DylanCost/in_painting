"""ODE sampler for flow matching inference.

This module implements ODE solvers for sampling from flow matching models,
enabling image inpainting at inference time.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from tqdm import tqdm


class ODESampler:
    """ODE sampler for flow matching inference using Euler method.
    
    Solves the ODE defined by the learned velocity field to generate
    inpainted images. The sampler integrates from t=0 (masked image)
    to t=1 (complete image) using the Euler method.
    
    ODE Formulation:
        dx_t/dt = v_θ(x_t, t, x₀, m)
    
    Euler Integration:
        x_{t+dt} = x_t + v_θ(x_t, t, x₀, m) * dt
    
    Args:
        model: The trained flow matching model (U-Net)
        num_steps: Number of integration steps (default: 100)
        preserve_observed: Whether to preserve observed pixels at each step (default: True)
        device: Device to run sampling on (default: None, uses model's device)
        show_progress: Whether to show progress bar (default: False)
    
    Example:
        >>> model = UNet(...)
        >>> sampler = ODESampler(model, num_steps=100)
        >>> 
        >>> # Prepare inputs
        >>> masked_image = torch.randn(1, 3, 128, 128)
        >>> mask = torch.randint(0, 2, (1, 1, 128, 128)).float()
        >>> 
        >>> # Sample
        >>> inpainted = sampler.sample(masked_image, mask)
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 100,
        preserve_observed: bool = True,
        device: Optional[torch.device] = None,
        show_progress: bool = False
    ):
        """Initialize ODE sampler.
        
        Args:
            model: The trained flow matching model
            num_steps: Number of integration steps
            preserve_observed: Whether to preserve observed pixels
            device: Device to run on (if None, uses model's device)
            show_progress: Whether to show progress bar
        """
        self.model = model
        self.num_steps = num_steps
        self.preserve_observed = preserve_observed
        self.show_progress = show_progress
        
        # Determine device
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
        
        # Compute step size
        self.dt = 1.0 / num_steps
    
    @torch.no_grad()
    def sample(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """Sample from the flow matching model to inpaint an image.
        
        Integrates the ODE from t=0 to t=1 using the Euler method:
            1. Start with x_0 where masked regions are sampled from N(0,1) and unmasked regions are from the input image
            2. For each step: x_{t+dt} = x_t + v_θ(x_t, t) * dt
            3. Optionally preserve observed pixels at each step
            4. Return final x_1 (inpainted image)
        
        Args:
            image: Image with observed regions, shape [B, 3, H, W]
            mask: Binary mask (1 = masked, 0 = observed), shape [B, 1, H, W]
            return_trajectory: If True, return all intermediate states (default: False)
        
        Returns:
            If return_trajectory is False:
                Inpainted image of shape [B, 3, H, W]
            If return_trajectory is True:
                List of tensors, one for each timestep (including initial and final)
        
        Note:
            The model is automatically set to eval mode during sampling.
        """
        # Set model to eval mode
        self.model.eval()
        
        # Move inputs to device
        # Initialize x_t: masked regions from N(0,1), unmasked from image
        x_t = image.to(self.device)
        x_t = (1 - mask.to(self.device)) * x_t + mask.to(self.device) * torch.randn_like(x_t)
        mask = mask.to(self.device)
        batch_size = x_t.shape[0]
        
        # Store trajectory if requested
        trajectory = [x_t.clone()] if return_trajectory else None
        
        # Create progress bar if requested
        iterator = range(self.num_steps)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Sampling", leave=False)
        
        # Euler integration
        for i in iterator:
            # Current time
            t = torch.full((batch_size,), i * self.dt, device=self.device)
            
            # Concatenate image and mask for model input
            x_input = torch.cat([x_t, mask], dim=1)  # [B, 4, H, W]
            
            # Predict velocity
            v_t = self.model(x_input, t)  # [B, 3, H, W]
            
            # Euler step
            x_t = x_t + v_t * self.dt
            
            # Preserve observed pixels (optional but recommended)
            if self.preserve_observed:
                x_t = (1 - mask) * image + mask * x_t
            
            # Store trajectory if requested
            if return_trajectory:
                trajectory.append(x_t.clone())
        
        if return_trajectory:
            return trajectory
        else:
            return x_t
    
    @torch.no_grad()
    def sample_with_callback(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        callback: Callable[[int, torch.Tensor], None]
    ) -> torch.Tensor:
        """Sample with a callback function called at each step.
        
        Useful for visualization or logging during sampling.
        
        Args:
            image: Image with observed regions, shape [B, 3, H, W]
            mask: Binary mask (1 = masked, 0 = observed), shape [B, 1, H, W]
            callback: Function called at each step with (step_idx, x_t)
        
        Returns:
            Inpainted image of shape [B, 3, H, W]
        """
        # Set model to eval mode
        self.model.eval()
        
        # Move inputs to device
        # Initialize x_t: masked regions from N(0,1), unmasked from image
        x_t = image.to(self.device)
        x_t = (1 - mask.to(self.device)) * x_t + mask.to(self.device) * torch.randn_like(x_t)
        mask = mask.to(self.device)
        batch_size = x_t.shape[0]
        
        # Initial callback
        callback(0, x_t)
        
        # Euler integration
        for i in range(self.num_steps):
            # Current time
            t = torch.full((batch_size,), i * self.dt, device=self.device)
            
            # Concatenate image and mask for model input
            x_input = torch.cat([x_t, mask], dim=1)
            
            # Predict velocity
            v_t = self.model(x_input, t)
            
            # Euler step
            x_t = x_t + v_t * self.dt
            
            # Preserve observed pixels
            if self.preserve_observed:
                x_t = (1 - mask) * image + mask * x_t
            
            # Callback
            callback(i + 1, x_t)
        
        return x_t
    
    def set_num_steps(self, num_steps: int):
        """Update the number of integration steps.
        
        Args:
            num_steps: New number of steps
        """
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps


class HeunSampler(ODESampler):
    """Heun's method (2nd order) ODE sampler for flow matching.
    
    Heun's method is a second-order Runge-Kutta method that provides
    better accuracy than Euler method with only one additional function
    evaluation per step.
    
    Algorithm:
        1. Predictor: x_pred = x_t + v_θ(x_t, t) * dt
        2. Corrector: x_{t+dt} = x_t + (v_θ(x_t, t) + v_θ(x_pred, t+dt)) * dt / 2
    
    Args:
        Same as ODESampler
    
    Note:
        Heun's method requires 2x the function evaluations of Euler method,
        but can achieve similar accuracy with fewer steps.
    """
    
    @torch.no_grad()
    def sample(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """Sample using Heun's method.
        
        Args:
            image: Image with observed regions, shape [B, 3, H, W]
            mask: Binary mask (1 = masked, 0 = observed), shape [B, 1, H, W]
            return_trajectory: If True, return all intermediate states
        
        Returns:
            Inpainted image or trajectory
        """
        # Set model to eval mode
        self.model.eval()
        
        # Move inputs to device
        # Initialize x_t: masked regions from N(0,1), unmasked from image
        x_t = image.to(self.device)
        x_t = (1 - mask.to(self.device)) * x_t + mask.to(self.device) * torch.randn_like(x_t)
        mask = mask.to(self.device)
        batch_size = x_t.shape[0]
        
        # Store trajectory if requested
        trajectory = [x_t.clone()] if return_trajectory else None
        
        # Create progress bar if requested
        iterator = range(self.num_steps)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Sampling (Heun)", leave=False)
        
        # Heun's method integration
        for i in iterator:
            # Current time
            t = torch.full((batch_size,), i * self.dt, device=self.device)
            t_next = torch.full((batch_size,), (i + 1) * self.dt, device=self.device)
            
            # Predictor step (Euler)
            x_input = torch.cat([x_t, mask], dim=1)
            v_t = self.model(x_input, t)
            x_pred = x_t + v_t * self.dt
            
            # Preserve observed pixels in predictor
            if self.preserve_observed:
                x_pred = (1 - mask) * image + mask * x_pred
            
            # Corrector step
            x_pred_input = torch.cat([x_pred, mask], dim=1)
            v_pred = self.model(x_pred_input, t_next)
            
            # Average of velocities
            x_t = x_t + (v_t + v_pred) * self.dt / 2
            
            # Preserve observed pixels
            if self.preserve_observed:
                x_t = (1 - mask) * image + mask * x_t
            
            # Store trajectory if requested
            if return_trajectory:
                trajectory.append(x_t.clone())
        
        if return_trajectory:
            return trajectory
        else:
            return x_t