import os
from pathlib import Path
from collections import defaultdict
import shutil
import sys
from pathlib import Path
from utils.config import config, Config

def prepare_data(dir: str | Path):
    """Split data into train, validation, and test sets maintaining class distribution.
    
    Args:
        dir (str | Path): Root directory path containing class folders
    """
    import random
    from sklearn.model_selection import train_test_split
    
    # Convert to Path object if string
    root_dir = Path(dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory {root_dir} does not exist")
    
    # Get configuration
    cfg = Config.from_yaml("/home/kitne/University/2lvl/NS/medplant-deepclassifier/config/default.yaml")
    
    # Extract split ratios from config
    train_ratio, val_ratio, test_ratio = cfg.data.train_split
    
    # Create train, validation, and test directories
    processed_dir = Path(cfg.data.processed_dir)
    train_dir = processed_dir / 'train'
    val_dir = processed_dir / 'val'
    test_dir = processed_dir / 'test'
    
    # Create directories if they don't exist
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(cfg.data.seed)
    
    # Process each class directory
    for class_dir in root_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        # Get all image files
        image_files = [
            f for f in class_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ]
        
        # First split: separate test set
        train_val_files, test_files = train_test_split(
            image_files,
            test_size=test_ratio,
            random_state=cfg.data.seed
        )
        
        # Second split: separate train and validation sets
        # Adjust validation ratio relative to the remaining data
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_ratio_adjusted,
            random_state=cfg.data.seed
        )
        
        # Create class directories in train, validation, and test
        train_class_dir = train_dir / class_dir.name
        val_class_dir = val_dir / class_dir.name
        test_class_dir = test_dir / class_dir.name
        
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        test_class_dir.mkdir(exist_ok=True)
        
        # Copy files to respective directories
        for file in train_files:
            shutil.copy2(file, train_class_dir / file.name)
        
        for file in val_files:
            shutil.copy2(file, val_class_dir / file.name)
        
        for file in test_files:
            shutil.copy2(file, test_class_dir / file.name)
        
        print(f"Processed {class_dir.name}: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test images")

if __name__ == "__main__":
    prepare_data(config.data.raw_dir)
    