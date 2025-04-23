
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import logging
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.res_branch import ResidualStream
from models.invres_branch import InvertedResidualStream
from models.fusion import SerialBasedFeatureFusion
from optimization.bco import FeatureSelectionChOA
from models.pipeline import StreamsTrainingPipeline, TrainingPipeline
from preprocessing.dataset import train_dataset, test_dataset, val_dataset
from utils.config import Config
from sklearn.neighbors import KNeighborsClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
config = Config.from_yaml("/home/kitne/University/2lvl/NS/medplant-deepclassifier/config/default.yaml")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    res_model = ResidualStream(30, 1024, feature_mode=True).to('cuda')
    invres_model = InvertedResidualStream(30, 1024, feature_mode=True).to('cuda')
    fusion = SerialBasedFeatureFusion(1024, 30).to('cuda')
    optimization = FeatureSelectionChOA(2048, 30, 50, 0.5, device)
    classifier = KNeighborsClassifier()

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4)

    streampipeline = StreamsTrainingPipeline(
        residual_branch=res_model,
        invresidual_branch=invres_model,
        train_dataset=train_loader,
        val_dataset=val_loader,
        test_dataset=test_loader,
        config=config
    )

    streampipeline.load(res_model, f'ResidualStream_Best.pth', 'eval')
    streampipeline.load(invres_model, f'InvertedResidualStream_Best.pth', 'eval')
    #streampipeline.cross_validation_train(res_model)
    #streampipeline.cross_validation_train(invres_model)

    trainingpipeline = TrainingPipeline(
        residual_branch=res_model.to(device),
        invresidual_branch=invres_model.to(device),
        fusion=fusion.to(device),
        optimization=optimization,
        classifier=classifier,
        train_dataset=train_loader,
        val_dataset=val_loader,
        test_dataset=test_loader,
        config=config
    )

    trainingpipeline.load()

    trainingpipeline.evaluate()