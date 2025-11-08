"""Flow matching module for image inpainting.

This module provides the core flow matching formulation and ODE sampling
for conditional image inpainting.
"""

from .flow_matching import FlowMatching
from .sampler import ODESampler, HeunSampler

__all__ = ['FlowMatching', 'ODESampler', 'HeunSampler']