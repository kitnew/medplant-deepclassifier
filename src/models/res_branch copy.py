import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from typing import List, Optional, Tuple

class ParallelBlock(nn.Module):
    """
    Параллельный блок: две ветки Conv→ReLU→Conv→BN→ReLU, выходы суммируются.
    Добавлены skip-connections и улучшенный downsample.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        # Projection shortcut for identity mapping when dimensions change
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
            
        # Standard projection for feature combination
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Ветка 1 - improved with BN after each conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Ветка 2 (идентичная)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply downsample if needed
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Project for feature combination
        proj_x = self.proj(x)
        
        # Process through branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        
        # Add identity and apply activation
        out1 = self.relu(out1 + proj_x)
        out2 = self.relu(out2 + proj_x)
        
        # Combine branch outputs and add skip connection
        out = out1 + out2 + identity
        return self.relu(out)


class ResidualStream(nn.Module):
    """
    Улучшенный первый поток архитектуры (Residual Stream).
    Добавлены skip-connections, дополнительный параллельный блок и улучшенный downsampling.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        
        # Initial stem block for better feature extraction
        self.stem = nn.Sequential(
            # First conv with downsampling
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Second conv to enrich features
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Store intermediate outputs for skip connections
        self.skip_connections = []
        
        # First stage - Parallel Block 1 (16→32) with downsampling
        self.parallel1 = ParallelBlock(in_channels=16, out_channels=32, stride=2)
        
        # Downsampling conv between stages
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Second stage - Parallel Block 2 (64→64)
        self.parallel2 = ParallelBlock(in_channels=64, out_channels=64, stride=1)
        
        # Downsampling conv between stages
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Third stage - New Parallel Block 3 (128→128)
        self.parallel3 = ParallelBlock(in_channels=128, out_channels=128, stride=1)
        
        # Final conv to increase channels
        self.conv_final = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global Average Pooling + improved classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        
        # Improved classifier with multiple layers
        self.classifier = nn.Sequential(
            weight_norm(nn.Linear(256, 128)),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            weight_norm(nn.Linear(128, 64)),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            weight_norm(nn.Linear(64, num_classes))
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial stem block
        x = self.stem(x)
        stem_features = x  # Save for potential skip connection
        
        # First parallel block with downsampling
        x = self.parallel1(x)
        stage1_features = x  # Save for skip connection
        
        # Downsampling and second parallel block
        x = self.down1(x)
        x = self.parallel2(x)
        stage2_features = x  # Save for skip connection
        
        # Downsampling and third parallel block
        x = self.down2(x)
        x = self.parallel3(x)
        
        # Final convolution to increase channels
        x = self.conv_final(x)
        
        # Global pooling and classification
        x = self.gap(x)              # [B,256,1,1]
        x = self.flat(x)             # [B,256]
        x = self.classifier(x)       # Apply classifier
        
        return x

