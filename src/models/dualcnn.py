import torch
import torch.nn as nn
from typing import Tuple
from torch.nn.utils.parametrizations import weight_norm

from models.res_branch import ResidualStream
from models.invres_branch import InvertedResidualStream
#from models.res_branch2 import ResidualStream
#from models.invres_branch2 import InvertedResidualStream

class DualCNN(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 1024) -> None:
        super(DualCNN, self).__init__()

        self.res_branch = ResidualStream(num_classes)
        self.invres_branch = InvertedResidualStream(num_classes)
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.tensor([0.4, 0.6]), requires_grad=True)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Apply classification layers
        res_logits = self.res_branch(x)
        invres_logits = self.invres_branch(x)
        
        # Normalize ensemble weights to sum to 1
        weights = torch.softmax(self.ensemble_weights, dim=0)
        
        # Weighted ensemble of logits
        ensemble_logits = weights[0] * res_logits + weights[1] * invres_logits
        
        return res_logits, invres_logits, ensemble_logits
        