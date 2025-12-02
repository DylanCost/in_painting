import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional


class VAELoss(nn.Module):
    """Combined loss for VAE inpainting."""
    
    def __init__(
        self,
        kl_weight: float = 0.001,
        perceptual_weight: float = 0.1,
        adversarial_weight: float = 0.001,
        use_perceptual: bool = True
    ):
        super().__init__()
        
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        
        # Reconstruction losses
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        # Perceptual loss
        if use_perceptual and perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
            # Move to CUDA if available
            if torch.cuda.is_available():
                self.perceptual_loss = self.perceptual_loss.cuda()
        else:
            self.perceptual_loss = None
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        discriminator_fake: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Calculate total loss and individual components."""
        
        losses = {}
        
        # Reconstruction loss (only on masked regions if mask provided)
        if mask is not None:
            rec_loss = self.l1_loss(reconstruction * mask, target * mask)
            # Also add loss on unmasked regions with lower weight
            rec_loss += 0.1 * self.l1_loss(reconstruction * (1 - mask), target * (1 - mask))
        else:
            rec_loss = self.l1_loss(reconstruction, target)
        
        losses['reconstruction'] = rec_loss
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        losses['kl'] = kl_loss * self.kl_weight
        
        # Perceptual loss
        if self.perceptual_loss is not None:
            perc_loss = self.perceptual_loss(reconstruction, target)
            losses['perceptual'] = perc_loss * self.perceptual_weight
        
        # Adversarial loss (if discriminator output provided)
        if discriminator_fake is not None:
            adv_loss = -torch.mean(torch.log(discriminator_fake + 1e-8))
            losses['adversarial'] = adv_loss * self.adversarial_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, layers=None):
        super().__init__()
        
        # Use VGG16 for feature extraction
        vgg = models.vgg16(pretrained=True).features
        vgg.eval()
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Default layers to use for perceptual loss
        if layers is None:
            # Use indices instead of names for reliability
            self.layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        else:
            self.layers = layers
            
        # Extract relevant layers
        self.vgg_layers = nn.ModuleList()
        last_idx = 0
        for layer_idx in self.layers:
            self.vgg_layers.append(nn.Sequential(*list(vgg.children())[last_idx:layer_idx+1]))
            last_idx = layer_idx + 1
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss between predicted and target images.
        
        Args:
            pred: Predicted images (B, C, H, W) in range [-1, 1]
            target: Target images (B, C, H, W) in range [-1, 1]
        
        Returns:
            Perceptual loss
        """
        # Normalize from [-1, 1] to [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        
        # Normalize for VGG (ImageNet statistics)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        loss = 0.0
        
        # Extract features at each layer and compute loss
        pred_input = pred
        target_input = target
        
        for vgg_layer in self.vgg_layers:
            pred_features = vgg_layer(pred_input)
            target_features = vgg_layer(target_input)
            loss += F.l1_loss(pred_features, target_features)
            
            # Update inputs for next layer
            pred_input = pred_features
            target_input = target_features
        
        return loss / len(self.layers)