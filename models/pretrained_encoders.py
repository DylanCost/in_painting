import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List, Tuple
import timm  # For additional pretrained models


class PretrainedResNetEncoder(nn.Module):
    """ResNet encoder pretrained on ImageNet or face datasets."""
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        pretrained: str = 'imagenet',  # 'imagenet', 'vggface2', or path to weights
        frozen_stages: int = 2,
        output_channels: List[int] = [64, 128, 256, 512, 512]
    ):
        super().__init__()
        
        # Load pretrained ResNet
        if pretrained == 'imagenet':
            backbone = models.resnet50(pretrained=True)
        elif pretrained == 'vggface2':
            # Use VGGFace2 pretrained weights
            from facenet_pytorch import InceptionResnetV1
            backbone = InceptionResnetV1(pretrained='vggface2')
        else:
            backbone = models.resnet50(pretrained=False)
            if pretrained:
                backbone.load_state_dict(torch.load(pretrained))
        
        # Extract layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Freeze early layers
        self._freeze_stages(frozen_stages)
        
        # Adaptation layers to match expected channels
        self.adapters = nn.ModuleList()
        resnet_channels = [64, 256, 512, 1024, 2048]
        
        for resnet_ch, out_ch in zip(resnet_channels, output_channels):
            self.adapters.append(
                nn.Conv2d(resnet_ch, out_ch, 1, bias=False)
            )
    
    def _freeze_stages(self, frozen_stages: int):
        """Freeze early stages for transfer learning."""
        if frozen_stages >= 1:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
        
        for i in range(1, frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_connections = []
        
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_connections.append(self.adapters[0](x))
        
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        skip_connections.append(self.adapters[1](x))
        
        x = self.layer2(x)
        skip_connections.append(self.adapters[2](x))
        
        x = self.layer3(x)
        skip_connections.append(self.adapters[3](x))
        
        x = self.layer4(x)
        final_features = self.adapters[4](x)
        skip_connections.append(final_features)
        
        return final_features, skip_connections[:-1]


class PretrainedVAEEncoder(nn.Module):
    """Load encoder from a pretrained VAE checkpoint."""
    
    def __init__(
        self,
        checkpoint_path: str,
        frozen: bool = False
    ):
        super().__init__()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        # Recreate encoder architecture
        from models.unet_vae import UNetEncoder
        self.encoder = UNetEncoder(
            input_channels=config['model'].get('input_channels', 3),
            hidden_dims=config['model'].get('hidden_dims', [64, 128, 256, 512, 512]),
            use_attention=config['model'].get('use_attention', True)
        )
        
        # Load weights
        if 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            # Extract encoder weights from full model
            full_state = checkpoint['model_state_dict']
            encoder_state = {
                k.replace('encoder.', ''): v 
                for k, v in full_state.items() 
                if k.startswith('encoder.')
            }
            self.encoder.load_state_dict(encoder_state)
        
        # Freeze if specified
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self.encoder(x)


class PretrainedStyleGANEncoder(nn.Module):
    """Use StyleGAN encoder for better face priors."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        frozen_layers: int = 4
    ):
        super().__init__()
        
        # This would load e4e or pSp encoder
        # Placeholder for StyleGAN encoder implementation
        self.encoder = self._load_stylegan_encoder(model_path)
        self._freeze_layers(frozen_layers)
    
    def _load_stylegan_encoder(self, model_path):
        # Implementation would load e4e/pSp models
        # These are good for face inversion
        pass
    
    def _freeze_layers(self, n_layers):
        # Freeze early layers
        pass