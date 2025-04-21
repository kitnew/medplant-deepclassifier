from dataclasses import dataclass
from pathlib import Path
import pyyaml
from typing import Optional

@dataclass
class DataConfig:
    dataset_name: str
    dataset_url: str
    raw_dir: Path
    processed_dir: Path
    train_split: float
    seed: int

@dataclass
class Config:
    data: DataConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        data_config = DataConfig(
            dataset_name=config_dict['data']['dataset_name'],
            dataset_url=config_dict['data']['dataset_url'],
            raw_dir=Path(config_dict['data']['raw_dir']),
            processed_dir=Path(config_dict['data']['processed_dir']),
            train_split=float(config_dict['data']['train_split']),
            seed=int(config_dict['data']['seed'])
        )
        
        return cls(data=data_config)

config = Config.from_yaml("../../config/default.yaml")



