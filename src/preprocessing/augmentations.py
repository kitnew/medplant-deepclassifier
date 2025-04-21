import torch
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
from enum import Enum
import numpy as np

def verify_image(path: str | Path) -> bool:
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False

# Custom transform to add Gaussian noise
class AddGaussianNoise:
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

# Training augmentations as per requirements:
# - horizontal/vertical flip
# - rotation (±90°)
# - color jitter
# - random crop
# - Gaussian noise
train_transform = v2.Compose([
    # First resize to slightly larger size for random crop
    v2.Resize((240, 240), antialias=True),
    # Random crop to target size
    v2.RandomCrop((224, 224)),
    # Flips
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    # Rotation (±90°)
    v2.RandomRotation(degrees=[-90, 90]),
    # Color jitter (brightness, contrast, saturation, hue)
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # Convert to tensor
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # Add Gaussian noise
    #AddGaussianNoise(mean=0.0, std=0.02)
    v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Test transforms - only resize to match input size
test_transform = v2.Compose([
    v2.Resize((224, 224), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])