from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Optional, List, Tuple

@dataclass
class DataConfig:
    dataset_name: str
    dataset_url: str
    raw_dir: Path
    processed_dir: Path
    train_split: float
    seed: int

@dataclass
class ModelConfig:
    input_size: List[int]  # [height, width, channels]
    num_classes: int
    feature_dim: int

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    dropout_rate: float
    cross_validation_folds: int

@dataclass
class BCOConfig:
    use_bco: bool
    population_size: int
    max_iterations: int
    binary_method: str

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    bco: BCOConfig

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
        
        model_config = ModelConfig(
            input_size=config_dict['model']['input_size'],
            num_classes=config_dict['model']['num_classes'],
            feature_dim=config_dict['model']['feature_dim']
        )
        
        training_config = TrainingConfig(
            batch_size=config_dict['training']['batch_size'],
            epochs=config_dict['training']['epochs'],
            learning_rate=config_dict['training']['learning_rate'],
            optimizer=config_dict['training']['optimizer'],
            weight_decay=config_dict['training']['weight_decay'],
            dropout_rate=config_dict['training']['dropout_rate'],
            cross_validation_folds=config_dict['training']['cross_validation_folds']
        )
        
        bco_config = BCOConfig(
            use_bco=config_dict['bco']['use_bco'],
            population_size=config_dict['bco']['population_size'],
            max_iterations=config_dict['bco']['max_iterations'],
            binary_method=config_dict['bco']['binary_method']
        )
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            bco=bco_config
        )

config = Config.from_yaml("/home/kitne/University/2lvl/NS/medplant-deepclassifier/config/default.yaml")
