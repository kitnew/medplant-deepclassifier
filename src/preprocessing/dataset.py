from torchvision.datasets import ImageFolder
from dataclasses import dataclass
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import Config
from .augmentations import train_transform, test_transform, verify_image

config = Config.from_yaml("/home/kitne/University/2lvl/NS/medplant-deepclassifier/config/default.yaml")

train_dataset = ImageFolder(config.data.processed_dir / "train", transform=train_transform, is_valid_file=verify_image)
val_dataset = ImageFolder(config.data.processed_dir / "val", transform=test_transform, is_valid_file=verify_image)
test_dataset = ImageFolder(config.data.processed_dir / "test", transform=test_transform, is_valid_file=verify_image)