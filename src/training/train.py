
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time
import logging
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pipeline import DeepClassifierPipeline, create_bco_fitness_function
from preprocessing.dataset import train_dataset, test_dataset
from utils.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
config = Config.from_yaml("/home/kitne/University/2lvl/NS/medplant-deepclassifier/config/default.yaml")

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=None,
    device='cuda',
    apply_bco=None,
    save_dir='checkpoints'
):
    # Use config values if parameters are not provided
    num_epochs = num_epochs if num_epochs is not None else config.training.epochs
    apply_bco = apply_bco if apply_bco is not None else config.bco.use_bco
    """
    Train the model with the given parameters
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of epochs to train
        device: Device to train on
        apply_bco: Whether to apply BCO feature selection
        save_dir: Directory to save model checkpoints
    
    Returns:
        Trained model and training history
    """
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize variables
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Move model to device
    model = model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, apply_feature_mask=False)
            loss = criterion(outputs['logits'], labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs['logits'], 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate epoch statistics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs, apply_feature_mask=False)
                loss = criterion(outputs['logits'], labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs['logits'], 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate epoch statistics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update learning rate scheduler if provided
        if scheduler:
            scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log progress
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save model if it's the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path / f"best_model.pth")
            logger.info(f"Saved new best model with validation accuracy: {val_acc:.4f}")
    
    # Apply BCO feature selection after training
    if apply_bco:
        logger.info("Applying Binary Chimp Optimization for feature selection...")
        
        # Extract features from validation set
        val_features = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Extracting validation features"):
                inputs, labels = inputs.to(device), labels.to(device)
                features = model.extract_features(inputs)
                val_features.append(features)
                val_labels.append(labels)
        
        # Concatenate all features and labels
        val_features = torch.cat(val_features, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        
        # Create fitness function for BCO
        fitness_fn = create_bco_fitness_function(model, val_loader, criterion, device)
        
        # Apply BCO
        feature_mask = model.apply_bco(val_features, val_labels, fitness_fn)
        
        # Save the model with feature mask
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_mask': feature_mask,
        }, save_path / f"model_with_bco.pth")
        logger.info(f"Saved model with BCO feature selection")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path / f"final_model.pth")
    
    return model, history

def cross_validation(
    model_class,
    dataset,
    num_folds=None,
    batch_size=None,
    num_epochs=None,
    learning_rate=None,
    optimizer_type=None,
    device='cuda',
    apply_bco=None,
    **model_kwargs
):
    # Use config values if parameters are not provided
    num_folds = num_folds if num_folds is not None else config.training.cross_validation_folds
    batch_size = batch_size if batch_size is not None else config.training.batch_size
    num_epochs = num_epochs if num_epochs is not None else config.training.epochs
    learning_rate = float(learning_rate) if learning_rate is not None else config.training.learning_rate
    optimizer_type = optimizer_type if optimizer_type is not None else config.training.optimizer
    apply_bco = apply_bco if apply_bco is not None else config.bco.use_bco
    """
    Perform k-fold cross-validation
    
    Args:
        model_class: Class of the model to train
        dataset: Dataset to use for cross-validation
        num_folds: Number of folds for cross-validation
        batch_size: Batch size for training
        num_epochs: Number of epochs to train each fold
        learning_rate: Learning rate for optimizer
        optimizer_type: Type of optimizer ('adam' or 'sgd')
        device: Device to train on
        apply_bco: Whether to apply BCO feature selection
        **model_kwargs: Additional arguments for model initialization
    
    Returns:
        List of trained models and their performance metrics
    """
    # Initialize KFold
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=config.data.seed)
    
    # Initialize metrics
    fold_results = []
    
    # Get class names
    class_names = dataset.classes
    num_classes = len(class_names)
    
    # Start cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logger.info(f"FOLD {fold+1}/{num_folds}")
        logger.info("-" * 50)
        
        # Sample elements randomly from a given list of indices, no replacement
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        # Define data loaders for training and validation
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=val_subsampler)
        
        # Initialize model
        model = model_class(num_classes=num_classes, **model_kwargs)
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Initialize optimizer
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(), 
                lr=learning_rate,
                weight_decay=float(config.training.weight_decay)
            )
        else:  # SGD with momentum
            optimizer = optim.SGD(
                model.parameters(), 
                lr=learning_rate, 
                momentum=0.9,
                weight_decay=float(config.training.weight_decay)
            )
        
        # Initialize scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.05, patience=2)

        
        # Train model
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            apply_bco=apply_bco,
            save_dir=f"checkpoints/fold_{fold+1}"
        )
        
        # Evaluate model
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, apply_feature_mask=True)
                _, preds = torch.max(outputs['logits'], 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Store results
        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'history': history
        })
        
        logger.info(f"Fold {fold+1} Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info("-" * 50)
    
    # Calculate average metrics
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_precision = np.mean([r['precision'] for r in fold_results])
    avg_recall = np.mean([r['recall'] for r in fold_results])
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    
    logger.info("Cross-Validation Results:")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Average Precision: {avg_precision:.4f}")
    logger.info(f"Average Recall: {avg_recall:.4f}")
    logger.info(f"Average F1 Score: {avg_f1:.4f}")
    
    return fold_results

def main():
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.data.seed)
    np.random.seed(config.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.data.seed)
    
    # Create data loaders
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=config.training.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=config.training.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4)
    
    # Get number of classes
    num_classes = len(train_dataset.classes)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {train_dataset.classes}")
    
    # Initialize model
    model = DeepClassifierPipeline(
        num_classes=num_classes,
        feature_dim=config.model.feature_dim,
        use_bco=config.bco.use_bco,
        bco_population_size=config.bco.population_size,
        bco_max_iter=config.bco.max_iterations,
        device=device
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Choose optimizer based on config
    if config.training.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=float(config.training.learning_rate),
            weight_decay=float(config.training.weight_decay)
        )
    else:  # SGD with momentum
        optimizer = optim.SGD(
            model.parameters(), 
            lr=float(config.training.learning_rate), 
            momentum=0.9,
            weight_decay=float(config.training.weight_decay)
        )
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.05, patience=3)
    
    # Train model
    logger.info("Starting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.training.epochs,
        device=device,
        apply_bco=config.bco.use_bco,
        save_dir="checkpoints"
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs, apply_feature_mask=True)
            loss = criterion(outputs['logits'], labels)
            
            # Statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs['logits'], 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / test_total
    
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")

    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
    
    # Uncomment to run cross-validation instead of single training
    # fold_results = cross_validation(
    #     model_class=DeepClassifierPipeline,
    #     dataset=train_dataset,
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    #     # All other parameters will be taken from config
    # )