import torch
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
from enum import Enum

def verify_image(path: str | Path) -> bool:
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False

class Mode(Enum):
    train = "train"
    validation = "validation"
    test = "test"

train_transform = v2.Compose([
    v2.Resize((224, 224), antialias=True),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomVerticalFlip(0.5),
    v2.GaussianNoise(),
    v2.ColorJitter(0.1, 0.1),
    v2.ToDtype(torch.float32, scale=True),
])

val_transform = v2.Compose([
    v2.Resize((224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])

test_transform = v2.Compose([
    v2.Resize((224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])

class Augmentation(Enum):
    train: v2.Compose = train_transform
    validation: v2.Compose = val_transform
    test: v2.Compose = test_transform