"""Grad-CAM implementation for CNN-based models.

This module implements Gradient-weighted Class Activation Mapping (Grad-CAM),
a technique to produce visual explanations for CNN decisions.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
           via Gradient-based Localization", https://arxiv.org/abs/1610.02391
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import logging
from typing import List, Tuple, Union, Optional, Dict, Any

logger = logging.getLogger(__name__)

class GradCAM:
    """Class for generating Grad-CAM visualizations for CNN models."""
    
    def __init__(self, model, target_layer, device='cuda'):
        """
        Initialize GradCAM.
        
        Args:
            model: PyTorch model
            target_layer: Layer to use for Grad-CAM
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        
        # Put model in evaluation mode
        self.model.eval()
        
        # Register hooks
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to the target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """Remove all hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def __call__(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM for the specified class index.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            class_idx: Class index for which to generate Grad-CAM
                      If None, uses the class with the highest score
        
        Returns:
            cam: Grad-CAM heatmap
            output: Model output
        """
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)
        
        # If class_idx is None, use the class with the highest score
        if class_idx is None:
            class_idx = torch.argmax(output).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights
        gradients = self.gradients.data
        activations = self.activations.data
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()[0, 0], output
    
    def visualize(self, img_tensor, class_idx=None, class_name=None, save_path=None, alpha=0.5):
        """
        Generate and visualize Grad-CAM for the given image.
        
        Args:
            img_tensor: Input image tensor [1, C, H, W]
            class_idx: Class index for which to generate Grad-CAM
            class_name: Class name for the title
            save_path: Path to save the visualization
            alpha: Weight for heatmap overlay
        
        Returns:
            fig: Matplotlib figure
        """
        # Generate Grad-CAM
        cam, output = self.__call__(img_tensor, class_idx)
        
        # If class_idx is None, use the predicted class
        if class_idx is None:
            class_idx = torch.argmax(output).item()
        
        # Convert tensor to numpy for visualization
        img = img_tensor.cpu().numpy()[0].transpose(1, 2, 0)
        
        # Normalize image for visualization
        img = (img - img.min()) / (img.max() - img.min())
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Overlay heatmap on image
        overlay = alpha * heatmap + (1 - alpha) * img
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot heatmap
        axes[1].imshow(heatmap)
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Plot overlay
        axes[2].imshow(overlay)
        title = f'Overlay (Class: {class_name if class_name else class_idx})'
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            save_path = Path(save_path)
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Saved Grad-CAM visualization to {save_path}")
        
        return fig


def get_last_conv_layer(model):
    """
    Utility function to find the last convolutional layer in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        last_conv_layer: Last convolutional layer
    """
    # Find all convolutional layers
    conv_layers = []
    
    def find_conv_layers(module, prefix=''):
        for name, m in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(m, torch.nn.Conv2d):
                conv_layers.append((full_name, m))
            else:
                find_conv_layers(m, full_name)
    
    find_conv_layers(model)
    
    if not conv_layers:
        raise ValueError("No convolutional layers found in the model")
    
    # Return the last convolutional layer
    return conv_layers[-1][1]


def visualize_multiple_classes(model, img_tensor, class_indices=None, class_names=None, 
                              target_layer=None, save_dir=None, device='cuda'):
    """
    Generate Grad-CAM visualizations for multiple classes.
    
    Args:
        model: PyTorch model
        img_tensor: Input image tensor [1, C, H, W]
        class_indices: List of class indices to visualize
        class_names: List of class names corresponding to class_indices
        target_layer: Layer to use for Grad-CAM (if None, uses the last conv layer)
        save_dir: Directory to save visualizations
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        figs: List of matplotlib figures
    """
    # If target_layer is not specified, find the last conv layer
    if target_layer is None:
        target_layer = get_last_conv_layer(model)
    
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer, device)
    
    # If class_indices is None, use all classes
    if class_indices is None:
        # Forward pass to get number of classes
        with torch.no_grad():
            output = model(img_tensor.to(device))
        num_classes = output.shape[1]
        class_indices = list(range(num_classes))
    
    # Generate figures
    figs = []
    for i, class_idx in enumerate(class_indices):
        class_name = class_names[i] if class_names else None
        
        # Generate save path if save_dir is provided
        save_path = None
        if save_dir:
            save_dir = Path(save_dir)
            class_label = class_name if class_name else f"class_{class_idx}"
            save_path = save_dir / f"gradcam_{class_label}.png"
        
        # Generate visualization
        fig = grad_cam.visualize(img_tensor, class_idx, class_name, save_path)
        figs.append(fig)
    
    # Remove hooks
    grad_cam.remove_hooks()
    
    return figs