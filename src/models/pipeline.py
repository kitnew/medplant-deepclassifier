import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Union, Any

from models.res_branch import ResidualStream
from models.invres_branch import InvertedResidualStream
from models.fusion import SerialBasedFeatureFusion
from models.classifier import MLPClassifier
from optimization.bco import FeatureSelectionChOA


class DeepClassifierPipeline(nn.Module):
    """
    Complete pipeline for medical plant image classification combining:
    1. Data preprocessing with augmentations
    2. Parallel feature extraction streams (Residual and Inverted Residual)
    3. Serial feature fusion with entropy-based selection
    4. Binary Chimp Optimization for feature selection
    5. MLP classifier for final prediction
    """
    def __init__(
        self, 
        num_classes: int,
        feature_dim: int = 1024,
        use_bco: bool = True,
        bco_population_size: int = 30,
        bco_max_iter: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        super(DeepClassifierPipeline, self).__init__()
        
        # 1. Feature extraction branches
        self.res_branch = ResidualStream(num_classes)
        self.invres_branch = InvertedResidualStream(num_classes)
        
        # 2. Feature fusion module
        self.fusion = SerialBasedFeatureFusion(
            input_dims=(1024, 1024),  # Both branches output 1024-dim features
            fused_dim=feature_dim,
            num_classes=num_classes
        )
        
        # 3. Final classifier
        self.classifier = MLPClassifier(in_dim=feature_dim, num_classes=num_classes)
        
        # 4. BCO settings
        self.use_bco = use_bco
        self.bco_population_size = bco_population_size
        self.bco_max_iter = bco_max_iter
        self.device = device
        
        # 5. Feature selection mask (will be set during training)
        self.feature_mask = None
        self.feature_dim = feature_dim
        
    def forward(self, x: torch.Tensor, apply_feature_mask: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete pipeline
        
        Args:
            x: Input tensor of shape [batch_size, 3, 224, 224]
            apply_feature_mask: Whether to apply the feature selection mask
                               (should be False during training before BCO)
        
        Returns:
            Dictionary containing:
                - 'res_features': Features from residual stream
                - 'invres_features': Features from inverted residual stream
                - 'fused_features': Features after fusion
                - 'masked_features': Features after applying BCO mask (if available)
                - 'logits': Final classification logits
                - 'probabilities': Softmax probabilities
        """
        # Extract features from both branches
        res_features = self.res_branch(x)  # [batch_size, 1024]
        invres_features = self.invres_branch(x)  # [batch_size, 1024]
        
        # Fuse features
        logits1, logits2, fused1, fused2 = self.fusion(res_features, invres_features)
        
        # Use the second fusion result (which should be more refined)
        fused_features = fused2
        
        # Apply feature mask if available and requested
        if self.feature_mask is not None and apply_feature_mask:
            masked_features = fused_features * self.feature_mask.to(fused_features.device)
        else:
            masked_features = fused_features
        
        # Final classification
        logits = self.classifier(masked_features)
        probabilities = F.softmax(logits, dim=1)
        
        return {
            'res_features': res_features,
            'invres_features': invres_features,
            'fused_features': fused_features,
            'masked_features': masked_features,
            'logits': logits,
            'probabilities': probabilities
        }
    
    def apply_bco(
        self, 
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        fitness_fn: callable
    ) -> torch.Tensor:
        """
        Apply Binary Chimp Optimization for feature selection
        
        Args:
            X_val: Validation features of shape [num_samples, feature_dim]
            y_val: Validation labels of shape [num_samples]
            fitness_fn: Function to evaluate feature subsets
                       Should accept (X, y, mask) and return a scalar (lower is better)
        
        Returns:
            Binary mask indicating selected features
        """
        # Initialize BCO optimizer
        bco = FeatureSelectionChOA(
            X=X_val,
            y=y_val,
            fitness_fn=fitness_fn,
            pop_size=self.bco_population_size,
            max_iter=self.bco_max_iter,
            device=self.device,
            verbose=True
        )
        
        # Run optimization
        best_mask, _, _ = bco.optimize()
        
        # Save the mask for future use
        self.feature_mask = best_mask
        
        return best_mask
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract fused features without classification
        Useful for feature extraction during BCO optimization
        
        Args:
            x: Input tensor of shape [batch_size, 3, 224, 224]
            
        Returns:
            Fused features of shape [batch_size, feature_dim]
        """
        res_features = self.res_branch(x)
        invres_features = self.invres_branch(x)
        _, _, _, fused_features = self.fusion(res_features, invres_features)
        return fused_features

    def get_feature_importance(self) -> torch.Tensor:
        """
        Get feature importance scores based on the BCO mask
        
        Returns:
            Feature importance scores (1 for selected, 0 for not selected)
        """
        if self.feature_mask is None:
            return torch.ones(self.feature_dim, device=self.device)
        return self.feature_mask


def create_bco_fitness_function(model, val_loader, criterion, device):
    """
    Create a fitness function for BCO optimization
    
    Args:
        model: The classifier model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run evaluation on
    
    Returns:
        Fitness function that accepts (X, y, mask) and returns validation error
    """
    def fitness_fn(X, y, mask):
        # Apply mask to features
        X_masked = X * mask.to(X.device)
        
        # Forward pass through classifier only
        with torch.no_grad():
            outputs = model.classifier(X_masked)
            loss = criterion(outputs, y)
            
            # Calculate error rate
            _, predicted = torch.max(outputs, 1)
            error_rate = 1.0 - (predicted == y).float().mean().item()
        
        return error_rate
    
    return fitness_fn