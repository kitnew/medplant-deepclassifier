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
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time
import logging
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pipeline import DeepClassifierPipeline
from models.dualcnn import DualCNN
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
):
    # Use config values if parameters are not provided
    num_epochs = num_epochs if num_epochs is not None else config.training.epochs
    
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
            output = model(inputs, labels, RandomForestClassifier())
            probabilities = F.softmax(output, dim=1)
            loss = criterion(output, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            preds = torch.argmax(probabilities, dim=1)
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
        
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
                output = model(inputs, labels, RandomForestClassifier())
                probabilities = F.softmax(output, dim=1)
                loss = criterion(output, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(probabilities, dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        
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

    return model

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
                lr=float(learning_rate),
                weight_decay=float(config.training.weight_decay)
            )
        else:  # SGD with momentum
            optimizer = optim.SGD(
                model.parameters(), 
                lr=float(learning_rate), 
                momentum=0.9,
                weight_decay=float(config.training.weight_decay)
            )
        
        # Initialize scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.05, patience=2)

        
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
        bco_population_size=config.bco.population_size,
        bco_max_iter=config.bco.max_iterations,
        bco_threshold=float(config.bco.threshold),
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
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.training.epochs,
        device=device
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
            output = model(inputs, labels, RandomForestClassifier())
            probabilities = F.softmax(output, dim=1)
            loss = criterion(output, labels)
            
            # Statistics
            test_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(probabilities, dim=1)
            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
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
    #main()
    
    # Uncomment to run cross-validation instead of single training
    # fold_results = cross_validation(
    #     model_class=DeepClassifierPipeline,
    #     dataset=train_dataset,
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    #     # All other parameters will be taken from config
    # )

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualCNN(30, 1024).to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Use Adam optimizer with lower learning rate
    optimizer1 = torch.optim.Adam(model.res_branch.parameters(), lr=0.01)
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, 
        mode='max',
        factor=0.5,
        patience=3,  # Increase patience to allow more exploration
        verbose=True
    )

    optimizer2 = torch.optim.Adam(model.invres_branch.parameters(), lr=0.01)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, 
        mode='max',
        factor=0.5,
        patience=3,  # Increase patience to allow more exploration
        verbose=True
    )

    def train_epoch():
        model.train()
        running_loss1 = 0.0
        running_loss2 = 0.0
        total_samples = 0
        
        for x_batch, y_batch in train_loader:
            batch_size = x_batch.size(0)
            total_samples += batch_size
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass
            logits1, logits2 = model(x_batch)
            
            # Calculate losses
            batch_loss1 = criterion(logits1, y_batch)
            batch_loss2 = criterion(logits2, y_batch)
            
            # Train residual branch
            optimizer1.zero_grad()
            batch_loss1.backward(retain_graph=True)  # Retain graph for second backward pass
            optimizer1.step()
            
            # Train inverted residual branch
            optimizer2.zero_grad()
            batch_loss2.backward()
            optimizer2.step()
            
            # Accumulate running loss
            running_loss1 += batch_loss1.item() * batch_size
            running_loss2 += batch_loss2.item() * batch_size

        return running_loss1 / total_samples, running_loss2 / total_samples

    def evaluate():
        model.eval()
        correct1 = 0
        correct2 = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                logits1, logits2 = model(x)
                preds1 = logits1.argmax(dim=1)
                preds2 = logits2.argmax(dim=1)
                correct1 += (preds1 == y).sum().item()
                correct2 += (preds2 == y).sum().item()

            acc1 = correct1 / len(test_loader.dataset)
            acc2 = correct2 / len(test_loader.dataset)
            return acc1, acc2

    epochs = 20
    best_acc = 0
    start_epoch = 0

    def train(start_epoch=0, best_acc=0):
        for epoch in range(start_epoch, epochs):
            loss1, loss2 = train_epoch()
            acc1, acc2 = evaluate()
            
            # Calculate combined accuracy
            combined_acc = (acc1 + acc2) / 2
            
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"Residual Block Loss: {loss1:.4f}, Accuracy: {acc1:.4f}")
            print(f"Inverted Residual Block Loss: {loss2:.4f}, Accuracy: {acc2:.4f}")
            print(f"Combined Accuracy: {combined_acc:.4f}")
            
            # Get current learning rates
            lr1 = optimizer1.param_groups[0]['lr']
            lr2 = optimizer2.param_groups[0]['lr']
            print(f"Learning rate 1: {lr1}")
            print(f"Learning rate 2: {lr2}")
            
            # Step schedulers based on accuracy
            scheduler1.step(acc1)
            scheduler2.step(acc2)

            # Save if we got better accuracy
            if acc1 > best_acc or acc2 > best_acc:
                best_acc = max(acc1, acc2)
                print(f"New best accuracy: {best_acc:.4f}")
                
                # Save model checkpoint
                # torch.save({
                #     'epoch': epoch,
                #     'model_state_dict': model.state_dict(),
                #     'optimizer1_state_dict': optimizer1.state_dict(),
                #     'optimizer2_state_dict': optimizer2.state_dict(),
                #     'loss1': loss1,
                #     'loss2': loss2,
                #     'acc1': acc1,
                #     'acc2': acc2,
                #     'best_acc': best_acc
                # }, f'checkpoint_epoch_{epoch+1}_acc_{best_acc:.4f}.pt')

    model.train()
    train(start_epoch, best_acc)