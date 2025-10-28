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
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
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
    
    def __init__(self, feature_layers: list = None):
        super().__init__()
        
        if feature_layers is None:
            feature_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract feature layers
        self.feature_extractor = nn.ModuleDict()
        layer_idx = 0
        
        for name, layer in vgg.named_children():
            if isinstance(layer, nn.Conv2d):
                layer_idx += 1
                name = f'conv{layer_idx}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu{layer_idx}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool{layer_idx}'
            
            self.feature_extractor[name] = layer
            
            if name in feature_layers:
                break
        
        self.feature_layers = feature_layers
        
        # Normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for VGG."""
        # Denormalize from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Normalize for VGG
        return (x - self.mean) / self.std
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract VGG features."""
        features = {}
        x = self.normalize(x)
        
        for name, layer in self.feature_extractor.items():
            x = layer(x)
            if name in self.feature_layers:
                features[name] = x
        
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate perceptual loss."""
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0
        for layer in self.feature_layers:
            loss += F.l1_loss(pred_features[layer], target_features[layer])
        
        return loss / len(self.feature_layers)