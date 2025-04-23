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
    train_split: List[float]
    seed: int

@dataclass
class ModelConfig:
    input_size: List[int]
    num_classes: int
    feature_dim: int

@dataclass
class StreamConfig:
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    warmup_factor: float
    eta_min: float

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    cross_validation_folds: int
    residual: StreamConfig
    invresidual: StreamConfig

@dataclass
class BCOConfig:
    population_size: int
    max_iterations: int
    threshold: float

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
            train_split=config_dict['data']['train_split'],
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
            cross_validation_folds=config_dict['training']['cross_validation_folds'],
            residual=StreamConfig(
                learning_rate=float(config_dict['training']['residual']['learning_rate']),
                weight_decay=float(config_dict['training']['residual']['weight_decay']),
                warmup_epochs=int(config_dict['training']['residual']['warmup_epochs']),
                warmup_factor=float(config_dict['training']['residual']['warmup_factor']),
                eta_min=float(config_dict['training']['residual']['eta_min'])
            ),
            invresidual=StreamConfig(
                learning_rate=float(config_dict['training']['invresidual']['learning_rate']),
                weight_decay=float(config_dict['training']['invresidual']['weight_decay']),
                warmup_epochs=int(config_dict['training']['invresidual']['warmup_epochs']),
                warmup_factor=float(config_dict['training']['invresidual']['warmup_factor']),
                eta_min=float(config_dict['training']['invresidual']['eta_min'])
            )
        )
        
        bco_config = BCOConfig(
            population_size=int(config_dict['bco']['population_size']),
            max_iterations=int(config_dict['bco']['max_iterations']),
            threshold=float(config_dict['bco']['threshold'])
        )
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            bco=bco_config
        )

config = Config.from_yaml("/home/kitne/University/2lvl/NS/medplant-deepclassifier/config/default.yaml")
