import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from typing import Dict, Tuple, List, Optional, Union, Any
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler
import torch.optim as optim
import pickle
import os
import logging
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        train_dataset: DataLoader,
        val_dataset: DataLoader,
        test_dataset: DataLoader,
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
            acc, _ = self.test(model)
            
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
        
        return best_acc
        
    def save(self, model: Union[ResidualStream, InvertedResidualStream], filename: str, epoch: int, loss: float, acc: float, best_acc: float):
        optimizer = self.res_optimizer if isinstance(model, ResidualStream) else self.invres_optimizer
        scheduler = self.res_scheduler if isinstance(model, ResidualStream) else self.invres_scheduler
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_last_lr': scheduler.get_last_lr()[0],
            'scheduler_current_epoch': scheduler.current_epoch,
            'scaler_state_dict': self.scaler.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'loss': loss,
            'acc': acc,
            'best_acc': best_acc
        }, filename)
        
    def load(self, model: Union[ResidualStream, InvertedResidualStream], filename: str, mode: str = 'eval'):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = self.res_optimizer if isinstance(model, ResidualStream) else self.invres_optimizer
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = self.res_scheduler if isinstance(model, ResidualStream) else self.invres_scheduler
        scheduler.set_last_lr(checkpoint['scheduler_last_lr'])
        #scheduler.set_current_epoch(checkpoint['scheduler_current_epoch'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        if mode == 'eval':
            model.eval()
        elif mode == 'train':
            model.train()
        return {
            'model': model,
            'optimizer': optimizer,
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss'],
            'acc': checkpoint['acc'],
            'best_acc': checkpoint['best_acc']
        }
        
    def cross_validation_train(self, model, num_folds=None):
        """Perform cross-validation training on the given model and dataset.
        
        Args:
            model: A pretrained model (ResidualStream or InvertedResidualStream)
            dataset: The dataset to use for cross-validation
            num_folds: Number of folds for cross-validation. If None, uses config value.
            
        Returns:
            Dict containing cross-validation metrics (accuracy, loss) for each fold
        """
        # Use config value if num_folds not provided
        if num_folds is None:
            num_folds = self.config.training.cross_validation_folds
            
        # Initialize KFold cross-validator
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=self.config.data.seed)
        
        # Store results for each fold
        cv_results = {
            'fold_accuracies': [],
            'fold_losses': [],
            'best_accuracy': 0.0,
            'best_fold': -1,
            'best_model_path': None
        }
        
        # Get dataset indices
        indices = list(range(len(self.train_dataset)))
        
        # Perform k-fold cross-validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            print(f"\nTraining fold {fold+1}/{num_folds}")
            
            # Create data samplers for train and validation splits
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            # Create data loaders using the samplers
            from preprocessing.dataset import train_dataset, val_dataset
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.training.batch_size,
                sampler=train_sampler
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                sampler=val_sampler
            )
            
            # Store original datasets temporarily
            original_train_dataset = self.train_dataset
            original_val_dataset = self.val_dataset
            
            # Replace with fold-specific datasets
            self.train_dataset = train_loader
            self.val_dataset = val_loader
            
            # Create a fresh copy of the model for this fold
            # We're using a pretrained model, so we'll clone it
            fold_model = type(model)(
                num_classes=self.config.model.num_classes,
                num_features=self.config.model.feature_dim,
                feature_mode=False
            )
            fold_model.load_state_dict(model.state_dict())
            fold_model = fold_model.to(self.device)
            
            # Train the model for this fold
            best_fold_acc = self.train(fold_model, self.epochs-10)
            
            # Evaluate on validation set
            fold_acc, fold_loss = self.validate(fold_model)
            
            # Save results for this fold
            cv_results['fold_accuracies'].append(fold_acc)
            cv_results['fold_losses'].append(fold_loss)
            
            # Check if this is the best fold so far
            if fold_acc > cv_results['best_accuracy']:
                cv_results['best_accuracy'] = fold_acc
                cv_results['best_fold'] = fold
                
                # Save the best model
                model_path = f"cv_best_model_fold_{fold+1}_{fold_acc:.4f}.pth"
                self.save(fold_model, model_path, self.epochs-1, fold_loss, fold_acc, fold_acc)
                cv_results['best_model_path'] = model_path
            
            # Restore original datasets
            self.train_dataset = original_train_dataset
            self.val_dataset = original_val_dataset
        
        # Calculate and print average metrics
        avg_accuracy = np.mean(cv_results['fold_accuracies'])
        avg_loss = np.mean(cv_results['fold_losses'])
        std_accuracy = np.std(cv_results['fold_accuracies'])
        
        print(f"\nCross-Validation Results:")
        print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Best Fold: {cv_results['best_fold']+1} with accuracy {cv_results['best_accuracy']:.4f}")
        print(f"Best model saved to: {cv_results['best_model_path']}")
        
        # Add average metrics to results
        cv_results['avg_accuracy'] = avg_accuracy
        cv_results['avg_loss'] = avg_loss
        cv_results['std_accuracy'] = std_accuracy
        
        return cv_results

class TrainingPipeline:
    def __init__(self,
        residual_branch: ResidualStream,
        invresidual_branch: InvertedResidualStream,
        fusion: SerialBasedFeatureFusion,
        optimization: FeatureSelectionChOA,
        classifier: Any,
        train_dataset: DataLoader,
        val_dataset: DataLoader,
        test_dataset: DataLoader,
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
            for batch_imgs, batch_lbls in tqdm(self.train_dataset, desc="Extracting features"):
                batch_imgs, batch_lbls = batch_imgs.to(self.device), batch_lbls.to(self.device)
                f_res = self.residual_branch(batch_imgs)     # [B, 1024]
                f_inv = self.invresidual_branch(batch_imgs)     # [B, 1024]
                f_fused = self.fusion(f_res, f_inv)    # [B, 2048]
                all_features.append(f_fused)
                all_labels.append(batch_lbls)

        # Concatenate all feature batches
        X = torch.cat(all_features, dim=0).cpu().numpy()
        y = torch.cat(all_labels, dim=0).cpu().numpy()

        # Optimize
        best_mask = self.optimization.optimize(X, y, self.classifier).cpu().numpy()

        optimized_features = X[:, best_mask]

        # Train the classifier
        clf = self.classifier
        clf.fit(optimized_features, y)

        self.classifier = clf

        self.save()

    def evaluate(self):
        # Same structure, no fitting
        X_eval, y_eval = [], []
        
        # Store a sample batch for Grad-CAM visualization
        sample_batch = None
        sample_labels = None

        for i, (images, labels) in enumerate(tqdm(self.val_dataset, desc="Evaluating")):
            # Store the first batch for Grad-CAM visualization
            if i == 0:
                sample_batch = images.clone()
                sample_labels = labels.clone()
                
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.no_grad():
                f_res = self.residual_branch(images)
                f_inv = self.invresidual_branch(images)
                fused = self.fusion(f_res, f_inv)

            X_eval.append(fused.cpu())
            y_eval.append(labels.cpu())

        X_eval = torch.cat(X_eval, dim=0).cpu().numpy()
        y_eval = torch.cat(y_eval, dim=0).cpu().numpy()

        # Optimize
        best_mask = self.optimization.optimize(X_eval, y_eval, self.classifier).cpu().numpy()

        optimized_features = X_eval[:, best_mask]

        # Predict and Evaluate
        predictions = self.classifier.predict(optimized_features)
        
        # Print results
        print("Evaluation Results:")
        print(f"Accuracy: {accuracy_score(y_eval, predictions):.4f}")
        print(f"Precision: {precision_score(y_eval, predictions, average='weighted'):.4f}")
        print(f"Recall: {recall_score(y_eval, predictions, average='weighted'):.4f}")
        print(f"F1-score: {f1_score(y_eval, predictions, average='weighted'):.4f}")
        print("\nClassification Report:")
        #print(classification_report(y_eval, predictions))
        
        # Create visualization directory
        vis_dir = Path(self.config.data.processed_dir).parent / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Get class names if available
        class_names = None
        if hasattr(self.test_dataset.dataset, 'classes'):
            class_names = self.test_dataset.dataset.classes
        
        # 1. Visualize evaluation metrics
        from src.visualization.evaluation_metrics import EvaluationVisualizer
        
        # Create a visualizer with a save directory
        visualizer = EvaluationVisualizer(save_dir=vis_dir)
        
        # Visualize all metrics
        visualizer.visualize_all(
            y_true=y_eval, 
            y_pred=predictions,
            class_names=class_names,
            title_prefix="Test "
        )
        
        # 2. Visualize Grad-CAM for a few sample images
        from src.visualization.grad_cam import GradCAM, get_last_conv_layer, visualize_multiple_classes
        
        # Only proceed with Grad-CAM if we have sample images
        if sample_batch is not None and len(sample_batch) > 0:
            # Create Grad-CAM directory
            gradcam_dir = vis_dir / 'gradcam'
            gradcam_dir.mkdir(parents=True, exist_ok=True)
            
            # Get the last convolutional layers from both branches
            try:
                # Get target layers for both branches
                res_target_layer = get_last_conv_layer(self.residual_branch)
                invres_target_layer = get_last_conv_layer(self.invresidual_branch)
                
                # Select a few sample images (up to 5)
                num_samples = min(5, len(sample_batch))
                
                for i in range(num_samples):
                    sample_img = sample_batch[i:i+1]  # Keep batch dimension
                    sample_label = sample_labels[i].item()
                    
                    # Class name for the sample
                    sample_class_name = class_names[sample_label] if class_names else f"Class {sample_label}"
                    
                    # Generate Grad-CAM for residual branch
                    print(f"\nGenerating Grad-CAM for sample {i+1}/{num_samples} (Class: {sample_class_name})")
                    
                    # Residual branch Grad-CAM
                    res_save_dir = gradcam_dir / f"sample_{i+1}_residual"
                    visualize_multiple_classes(
                        model=self.residual_branch,
                        img_tensor=sample_img,
                        class_indices=[sample_label],
                        class_names=[sample_class_name] if class_names else None,
                        target_layer=res_target_layer,
                        save_dir=res_save_dir,
                        device=self.device
                    )
                    
                    # Inverted Residual branch Grad-CAM
                    invres_save_dir = gradcam_dir / f"sample_{i+1}_invresidual"
                    visualize_multiple_classes(
                        model=self.invresidual_branch,
                        img_tensor=sample_img,
                        class_indices=[sample_label],
                        class_names=[sample_class_name] if class_names else None,
                        target_layer=invres_target_layer,
                        save_dir=invres_save_dir,
                        device=self.device
                    )
                    
                print(f"\nGrad-CAM visualizations saved to {gradcam_dir}")
                    
            except Exception as e:
                print(f"Error generating Grad-CAM visualizations: {str(e)}")
        
        return {
            'accuracy': accuracy_score(y_eval, predictions),
            'precision': precision_score(y_eval, predictions, average='weighted'),
            'recall': recall_score(y_eval, predictions, average='weighted'),
            'f1_score': f1_score(y_eval, predictions, average='weighted')
        }
        

    def save(self):
        torch.save(self.residual_branch.state_dict(), "residual_branch.pth")
        torch.save(self.invresidual_branch.state_dict(), "invresidual_branch.pth")
        torch.save(self.fusion.state_dict(), "fusion.pth")
        with open("optimization.pkl", "wb") as f:
            pickle.dump(self.optimization, f)
        with open("classifier.pkl", "wb") as f:
            pickle.dump(self.classifier, f)

    def load(self):
        self.residual_branch.load_state_dict(torch.load("residual_branch.pth"))
        self.invresidual_branch.load_state_dict(torch.load("invresidual_branch.pth"))
        self.fusion.load_state_dict(torch.load("fusion.pth"))
        with open("optimization.pkl", "rb") as f:
            self.optimization = pickle.load(f)
        with open("classifier.pkl", "rb") as f:
            self.classifier = pickle.load(f)