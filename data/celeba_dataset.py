import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from typing import Tuple, Optional, Dict, Any


class CelebADataset(Dataset):
    """CelebA Dataset for inpainting with automatic mask generation."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 256,
        mask_generator: Optional['MaskGenerator'] = None,
        transform: Optional[transforms.Compose] = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.mask_generator = mask_generator or MaskGenerator()
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        # Load image paths
        self.image_paths = self._load_image_paths()
        
    def _load_image_paths(self) -> list:
        """Load image paths based on split."""
        img_dir = os.path.join(self.root_dir, 'img_align_celeba')
        if not os.path.exists(img_dir):
            raise ValueError(f"CelebA dataset not found at {img_dir}")
            
        all_images = sorted(os.listdir(img_dir))
        
        # Simple split: 80% train, 10% val, 10% test
        total = len(all_images)
        if self.split == 'train':
            return all_images[:int(0.8 * total)]
        elif self.split == 'val':
            return all_images[int(0.8 * total):int(0.9 * total)]
        else:  # test
            return all_images[int(0.9 * total):]
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        img_path = os.path.join(self.root_dir, 'img_align_celeba', self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Generate mask
        mask = self.mask_generator.generate(
            (1, self.image_size, self.image_size)
        )
        mask_tensor = torch.from_numpy(mask).float()
        
        # Create masked image
        masked_image = image_tensor * (1 - mask_tensor)
        
        return {
            'image': image_tensor,
            'masked_image': masked_image,
            'mask': mask_tensor,
            'idx': idx
        }


class MaskGenerator:
    """Generate various types of masks for inpainting."""
    
    def __init__(
        self,
        mask_type: str = 'random',
        mask_ratio: float = 0.4,
        min_size: int = 32,
        max_size: int = 128
    ):
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.min_size = min_size
        self.max_size = max_size
    
    def generate(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate mask based on configured type."""
        if self.mask_type == 'random':
            return self.random_bbox_mask(shape)
        elif self.mask_type == 'center':
            return self.center_mask(shape)
        elif self.mask_type == 'irregular':
            return self.irregular_mask(shape)
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")
    
    def random_bbox_mask(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate random bounding box mask."""
        _, h, w = shape
        mask = np.zeros((1, h, w), dtype=np.float32)
        
        # Random box size
        box_h = np.random.randint(self.min_size, min(self.max_size, h))
        box_w = np.random.randint(self.min_size, min(self.max_size, w))
        
        # Random position
        y = np.random.randint(0, h - box_h)
        x = np.random.randint(0, w - box_w)
        
        mask[:, y:y+box_h, x:x+box_w] = 1.0
        return mask
    
    def center_mask(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate center mask."""
        _, h, w = shape
        mask = np.zeros((1, h, w), dtype=np.float32)
        
        size = int(min(h, w) * self.mask_ratio)
        y = (h - size) // 2
        x = (w - size) // 2
        
        mask[:, y:y+size, x:x+size] = 1.0
        return mask
    
    def irregular_mask(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate irregular free-form mask."""
        _, h, w = shape
        mask = np.zeros((1, h, w), dtype=np.float32)
        
        # Number of strokes
        num_strokes = np.random.randint(1, 5)
        
        for _ in range(num_strokes):
            # Random starting point
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            
            # Random walk
            num_points = np.random.randint(5, 15)
            for _ in range(num_points):
                # Random direction and length
                angle = np.random.uniform(0, 2 * np.pi)
                length = np.random.randint(10, 30)
                
                # Calculate end point
                y_end = int(y + length * np.sin(angle))
                x_end = int(x + length * np.cos(angle))
                
                # Clip to image boundaries
                y_end = np.clip(y_end, 0, h - 1)
                x_end = np.clip(x_end, 0, w - 1)
                
                # Draw thick line
                thickness = np.random.randint(5, 15)
                mask[:, max(0, y-thickness):min(h, y+thickness),
                     max(0, x-thickness):min(w, x+thickness)] = 1.0
                
                # Move to end point
                y, x = y_end, x_end
        
        return mask