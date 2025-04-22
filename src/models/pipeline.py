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
        bco_population_size: int = 50,
        bco_max_iter: int = 50,
        bco_threshold: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        super(DeepClassifierPipeline, self).__init__()
        
        # 1. Feature extraction branches
        self.res_branch = ResidualStream(feature_dim)
        self.invres_branch = InvertedResidualStream(feature_dim)
        
        # 2. Feature fusion module
        self.fusion = SerialBasedFeatureFusion(
            feature_dim=feature_dim,
            bins=num_classes
        )
        
        # 3. Final classifier
        self.classifier = MLPClassifier(in_dim=feature_dim, num_classes=num_classes)
        
        # 4. BCO settings
        self.bco_population_size = bco_population_size
        self.bco_max_iter = bco_max_iter
        self.bco_threshold = bco_threshold
        self.device = device
        
        # 5. Feature selection mask (will be set during training)
        self.feature_mask = None
        self.feature_dim = feature_dim
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor, clf: Any) -> torch.Tensor:
        """
        Forward pass through the complete pipeline
        
        Args:
            x: Input tensor of shape [batch_size, 3, 224, 224]
            all_feats: All features of the dataset
            all_labels: All labels of the dataset
            clf: Classifier to be used for feature selection
                               (should be False during training before BCO)
        
        Returns:
            Dictionary containing:
                - 'res_features': Features from residual stream
                - 'invres_features': Features from inverted residual stream
                - 'fused_features': Features after fusion
                - 'feature_mask': Binary mask indicating selected features
                - 'selected_features': Features after applying BCO mask (if available)
                - 'logits': Final classification logits
                - 'probabilities': Softmax probabilities
        """

        # Extract features from both branches
        res_features = self.res_branch(x)  # [batch_size, 1024]
        invres_features = self.invres_branch(x)  # [batch_size, 1024]
        
        # Fuse features
        fused_features = self.fusion(res_features, invres_features)

        feature_mask = self.apply_bco(fused_features, self.bco_population_size, self.bco_max_iter, self.bco_threshold, self.device, fused_features, labels, clf)
        
        selected_features = fused_features[:, feature_mask]
        
        # Final classification
        logits = self.classifier(selected_features)
        
        return logits
    
    def apply_bco(
        self, 
        num_features: int,
        pop_size: int,
        max_iter: int,
        threshold: float,
        device: torch.device,
        feats: torch.Tensor,
        labels: torch.Tensor,
        clf: Any
    ) -> torch.Tensor:

        # Initialize BCO optimizer
        bco = FeatureSelectionChOA(
            num_features=num_features.size(1),
            pop_size=pop_size,
            max_iter=max_iter,
            threshold=threshold,
            device=device
        )
        
        mask = bco.optimize(feats, labels, clf)
        selected_idx = mask.nonzero(as_tuple=True)[0]
        
        return selected_idx
    
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