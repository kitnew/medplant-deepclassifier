"""
Visualization module for evaluation metrics.
This module provides functions to visualize various evaluation metrics
such as accuracy, precision, recall, F1-score, and confusion matrix.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class EvaluationVisualizer:
    """Class for visualizing evaluation metrics."""
    
    def __init__(self, save_dir=None):
        """
        Initialize the EvaluationVisualizer.
        
        Args:
            save_dir (str or Path, optional): Directory to save visualizations.
                If None, visualizations will not be saved.
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir and not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory for saving visualizations: {self.save_dir}")
    
    def plot_metrics(self, y_true, y_pred, class_names=None, title_prefix=""):
        """
        Plot and optionally save evaluation metrics.
        
        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.
            class_names (list, optional): List of class names.
            title_prefix (str, optional): Prefix for plot titles.
        
        Returns:
            dict: Dictionary containing the evaluation metrics.
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{title_prefix}Evaluation Metrics", fontsize=16)
        
        # Plot bar charts for each metric
        metrics_list = list(metrics.items())
        
        for i, ax in enumerate(axes.flatten()[:4]):
            if i < len(metrics_list):
                metric_name, metric_value = metrics_list[i]
                ax.bar([metric_name], [metric_value], color='steelblue')
                ax.set_ylim(0, 1)
                ax.set_title(f"{metric_name.capitalize()}: {metric_value:.4f}")
                ax.set_ylabel('Score')
                # Add the value on top of the bar
                #ax.text(0, metric_value + 0.02, f"{metric_value:.4f}", 
                #        ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        # Save the figure if save_dir is provided
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.save_dir / f"{title_prefix}metrics_{timestamp}.png"
            plt.savefig(filename)
            logger.info(f"Saved metrics visualization to {filename}")
        
        plt.show()
        plt.close()
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, title_prefix=""):
        """
        Plot and optionally save confusion matrix.
        
        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.
            class_names (list, optional): List of class names.
            title_prefix (str, optional): Prefix for plot title.
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot confusion matrix
        if class_names is not None and len(class_names) == cm.shape[0]:
            # If we have class names and they match the number of classes
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
        else:
            # Otherwise use numerical indices
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title(f"{title_prefix}Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Make sure labels are visible
        plt.tight_layout()
        
        # Save the figure if save_dir is provided
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.save_dir / f"{title_prefix}confusion_matrix_{timestamp}.png"
            plt.savefig(filename)
            logger.info(f"Saved confusion matrix to {filename}")
        
        plt.show()
        plt.close()
    
    def visualize_all(self, y_true, y_pred, class_names=None, title_prefix=""):
        """
        Visualize all evaluation metrics including confusion matrix.
        
        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.
            class_names (list, optional): List of class names.
            title_prefix (str, optional): Prefix for plot titles.
            
        Returns:
            dict: Dictionary containing the evaluation metrics.
        """
        metrics = self.plot_metrics(y_true, y_pred, class_names, title_prefix)
        self.plot_confusion_matrix(y_true, y_pred, class_names, title_prefix)
        
        # Log metrics
        logger.info(f"{title_prefix}Evaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-score: {metrics['f1_score']:.4f}")
        
        return metrics
