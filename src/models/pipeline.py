import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional, Union, Any
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler
import torch.optim as optim
import pickle

from models.res_branch import ResidualStream
from models.invres_branch import InvertedResidualStream
from models.fusion import SerialBasedFeatureFusion
from optimization.bco import FeatureSelectionChOA

from utils.config import Config
from training.utils import WarmupCosineAnnealingLR

class StreamsTrainingPipeline:
    def __init__(self,
        residual_branch: ResidualStream,
        invresidual_branch: InvertedResidualStream,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        config: Config
    ) -> None:
        
        self.residual_branch = residual_branch
        self.invresidual_branch = invresidual_branch
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = config.training.epochs

        self.criterion = nn.CrossEntropyLoss()

        self.res_optimizer = torch.optim.Adam(
            self.residual_branch.parameters(),
            lr=config.training.residual.learning_rate,
            weight_decay=config.training.residual.weight_decay
        )
        self.invres_optimizer = torch.optim.Adam(
            self.invresidual_branch.parameters(),
            lr=config.training.invresidual.learning_rate,
            weight_decay=config.training.invresidual.weight_decay
        )

        self.res_scheduler = WarmupCosineAnnealingLR(
            self.res_optimizer,
            warmup_epochs=config.training.residual.warmup_epochs,
            warmup_factor=config.training.residual.warmup_factor,
            T_max=self.epochs,
            eta_min=config.training.residual.eta_min
        )

        self.invres_scheduler = WarmupCosineAnnealingLR(
            self.invres_optimizer,
            warmup_epochs=config.training.invresidual.warmup_epochs,
            warmup_factor=config.training.invresidual.warmup_factor,
            T_max=self.epochs,
            eta_min=config.training.invresidual.eta_min
        )

        self.scaler = GradScaler('cuda')

    def train_epoch(self, model: Union[ResidualStream, InvertedResidualStream]):
        model.train()
        running_loss = 0.0
        total_samples = 0
        
        for x_batch, y_batch in tqdm(self.train_dataset, desc="Training"):
            batch_size = x_batch.size(0)
            total_samples += batch_size
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            
            optimizer = self.res_optimizer if isinstance(model, ResidualStream) else self.invres_optimizer
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(x_batch)
                batch_loss = self.criterion(logits, y_batch)
            
            self.scaler.scale(batch_loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            running_loss += batch_loss.item() * batch_size
        
        return running_loss / total_samples

    def evaluate(self, model: Union[ResidualStream, InvertedResidualStream], dataset: Dataset):
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        
        with torch.no_grad():
            for x, y in tqdm(dataset, desc="Evaluating"):
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                loss = self.criterion(logits, y)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                running_loss += loss.item() * y.size(0)
        
        accuracy = correct / total
        avg_loss = running_loss / total
        return accuracy, avg_loss

    def validate(self, model: Union[ResidualStream, InvertedResidualStream]):
        return self.evaluate(model, self.val_dataset)

    def test(self, model: Union[ResidualStream, InvertedResidualStream]):
        return self.evaluate(model, self.test_dataset)

    def train(self, model: Union[ResidualStream, InvertedResidualStream], start_epoch=0, best_acc=0):
        scheduler = self.res_scheduler if isinstance(model, ResidualStream) else self.invres_scheduler
        
        for epoch in range(start_epoch, self.epochs):
            loss = self.train_epoch(model)
            acc = self.validate(model)
            
            # Print training results
            print(f"Epoch [{epoch+1}/{self.epochs}]")
            print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
            
            # Get current learning rates
            lr = scheduler.get_last_lr()[0]
            print(f"Learning rate: {lr:.8f}")
            
            # Step scheduler (cosine annealing doesn't need metrics)
            scheduler.step()
            
            # Save if we got better accuracy
            if acc > best_acc:
                best_acc = acc
                print(f"New best accuracy: {best_acc:.4f}")
                self.save(model, f"{model.__class__.__name__}_{epoch}_{best_acc:.4f}.pth", epoch, loss, acc, best_acc)
        
    def save(self, model: Union[ResidualStream, InvertedResidualStream], filename: str, epoch: int, loss: float, acc: float, best_acc: float):
        optimizer = self.res_optimizer if isinstance(model, ResidualStream) else self.invres_optimizer
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'acc': acc,
            'best_acc': best_acc
        }, filename)
        
    def load(self, model: Union[ResidualStream, InvertedResidualStream], filename: str, mode: str = 'eval'):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        if mode == 'eval':
            model.eval()
        elif mode == 'train':
            model.train()
        return {
            'model': model,
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss'],
            'acc': checkpoint['acc'],
            'best_acc': checkpoint['best_acc']
        }

class TrainingPipeline:
    def __init__(self,
        residual_branch: ResidualStream,
        invresidual_branch: InvertedResidualStream,
        fusion: SerialBasedFeatureFusion,
        optimization: FeatureSelectionChOA,
        classifier: Any,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        config: Config
    ) -> None:

        self.residual_branch = residual_branch
        self.invresidual_branch = invresidual_branch
        self.fusion = fusion
        self.optimization = optimization
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        self.residual_branch.eval()
        self.invresidual_branch.eval()
        self.fusion.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch_imgs, batch_lbls in self.train_dataset:
                f_res = self.residual_branch(batch_imgs)     # [B, 1024]
                f_inv = self.invresidual_branch(batch_imgs)     # [B, 1024]
                f_fused = self.fusion(f_res, f_inv)    # [B, 2048]
                all_features.append(f_fused)
                all_labels.append(batch_lbls)

        # Concatenate all feature batches
        X = torch.cat(all_features, dim=0).cpu().numpy()
        y = torch.cat(all_labels, dim=0).cpu().numpy()

        # Optimize
        best_mask = self.optimization.optimize(X, y, self.classifier)

        optimized_features = X[:, best_mask]

        # Train the classifier
        clf = self.classifier
        clf.fit(optimized_features, y)

        self.classifier = clf

        self.save()

    def save(self):
        torch.save(self.residual_branch, "residual_branch.pth")
        torch.save(self.invresidual_branch, "invresidual_branch.pth")
        torch.save(self.fusion, "fusion.pth")
        with open("classifier.pkl", "wb") as f:
            pickle.dump(self.classifier, f)

    def load(self):
        self.residual_branch = torch.load("residual_branch.pth")
        self.invresidual_branch = torch.load("invresidual_branch.pth")
        self.fusion = torch.load("fusion.pth")
        with open("classifier.pkl", "rb") as f:
            self.classifier = pickle.load(f)