
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

from models.pipeline import DeepClassifierPipeline
from preprocessing.dataset import test_dataset
from utils.config import Config

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

def load_model(model_path, num_classes, device='cuda'):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to the model checkpoint
        num_classes: Number of classes
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Initialize model
    model = DeepClassifierPipeline(
        num_classes=num_classes,
        feature_dim=1024,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load feature mask if available
    if 'feature_mask' in checkpoint:
        model.feature_mask = checkpoint['feature_mask']
        logger.info(f"Loaded feature mask with {int(model.feature_mask.sum().item())} selected features")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, test_loader, criterion, device='cuda'):
    """
    Evaluate model on test set
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs, apply_feature_mask=True)
            loss = criterion(outputs['logits'], labels)
            
            # Get probabilities
            probs = outputs['probabilities']
            
            # Statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs['logits'], 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / test_total
    
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Calculate per-class metrics
    class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Get classification report
    report = classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes, output_dict=True)
    
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'confusion_matrix': cm,
        'report': report,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }

def visualize_results(results, class_names, save_dir='results'):
    """
    Visualize evaluation results
    
    Args:
        results: Dictionary with evaluation metrics
        class_names: List of class names
        save_dir: Directory to save visualizations
    """
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = results['confusion_matrix']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrix.png')
    
    # 2. Per-class metrics
    plt.figure(figsize=(15, 6))
    
    metrics = {
        'Precision': results['class_precision'],
        'Recall': results['class_recall'],
        'F1 Score': results['class_f1']
    }
    
    x = np.arange(len(class_names))
    width = 0.25
    multiplier = 0
    
    for metric_name, metric_values in metrics.items():
        offset = width * multiplier
        plt.bar(x + offset, metric_values, width, label=metric_name)
        multiplier += 1
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Per-class Metrics')
    plt.xticks(x + width, class_names, rotation=45, ha='right')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path / 'per_class_metrics.png')
    
    # 3. Feature importance if available
    if hasattr(model, 'feature_mask') and model.feature_mask is not None:
        feature_importance = model.get_feature_importance().cpu().numpy()
        selected_features = np.sum(feature_importance)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.title(f'Feature Importance (Selected: {int(selected_features)}/{len(feature_importance)})')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance (1=Selected, 0=Not Selected)')
        plt.savefig(save_path / 'feature_importance.png')
    
    # 4. ROC curves and AUC for each class (if binary or multiclass)
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    
    # One-vs-Rest ROC curves
    plt.figure(figsize=(12, 10))
    
    n_classes = len(class_names)
    
    # Binarize the labels for one-vs-rest ROC
    y_test = np.zeros((len(results['labels']), n_classes))
    for i in range(len(results['labels'])):
        y_test[i, results['labels'][i]] = 1
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], results['probabilities'][:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], 
            tpr[i], 
            color=color, 
            lw=2,
            label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.savefig(save_path / 'roc_curves.png')
    
    # Save metrics to text file
    with open(save_path / 'metrics.txt', 'w') as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n\n")
        
        f.write("Classification Report:\n")
        for class_name, metrics in results['report'].items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            f.write(f"Class: {class_name}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1-score']:.4f}\n")
            f.write(f"  Support: {metrics['support']}\n\n")

def main():
    """Main evaluation function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.data.seed)
    np.random.seed(config.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.data.seed)
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Get number of classes
    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")
    
    # Load model
    model_path = "checkpoints/model_with_bco.pth"  # Change this to the path of your model
    if not os.path.exists(model_path):
        model_path = "checkpoints/best_model.pth"
        if not os.path.exists(model_path):
            model_path = "checkpoints/final_model.pth"
    
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path, num_classes, device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = evaluate_model(model, test_loader, criterion, device)
    
    # Log results
    logger.info(f"Test Loss: {results['loss']:.4f}")
    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Test Precision: {results['precision']:.4f}")
    logger.info(f"Test Recall: {results['recall']:.4f}")
    logger.info(f"Test F1 Score: {results['f1']:.4f}")
    
    # Visualize results
    logger.info("Visualizing results...")
    visualize_results(results, class_names, save_dir="results")
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()