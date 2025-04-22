import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from typing import List, Optional, Tuple

class InvertedParallelBlock(nn.Module):
    """
    Улучшенный параллельный блок для Inverted Residual Stream с MobileNetV2-подобной структурой:
    - Expansion layer (1x1 conv to increase channels)
    - Depthwise conv
    - Projection layer (1x1 conv to reduce channels)
    - Две параллельные ветки с общим skip-connection
    """
    def __init__(self, in_channels, out_channels, expansion_factor=6, stride=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = (in_channels == out_channels and stride == 1)
        
        # Расширенное количество каналов для inverted residual concept
        expanded_channels = in_channels * expansion_factor
        
        # Общий expansion layer для обеих веток
        self.expand = nn.Sequential(
            # 1x1 pointwise conv для увеличения размерности (expansion)
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True)
        ) if expansion_factor != 1 else nn.Identity()
        
        # Ветка 1 - depthwise + pointwise (проекция)
        self.branch1 = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, 
                      stride=stride, padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            # Pointwise conv для проекции обратно (сжатие)
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Ветка 2 - depthwise + pointwise (проекция) - аналогично ветке 1
        self.branch2 = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, 
                      stride=stride, padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            # Pointwise conv для проекции обратно (сжатие)
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut connection если размерности совпадают
        self.shortcut = None
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Сохраняем входные данные для skip connection
        identity = x
        
        # Expansion phase
        x = self.expand(x)
        
        # Parallel branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        
        # Combine branches
        out = out1 + out2
        
        # Add skip connection if dimensions match
        if self.use_residual:
            out = out + identity
        elif self.shortcut is not None:
            out = out + self.shortcut(identity)
            
        return out

class InvertedResidualStream(nn.Module):
    """
    Улучшенный Inverted Residual Stream с MobileNetV2-подобной архитектурой,
    агрессивным увеличением каналов и параллельными ветками.
    """
    def __init__(self, num_classes: int):
        super().__init__()

        # Начальный stem block
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            # Depthwise separable conv
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1, bias=False),  # Projection
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )
        
        # Агрессивное увеличение каналов с inverted residual blocks
        # Первый блок: t=1 (no expansion for first block)
        self.block1 = InvertedParallelBlock(16, 24, expansion_factor=1, stride=2)  # 16→24, /2
        
        # Второй блок: t=6 (standard expansion factor)
        self.block2 = InvertedParallelBlock(24, 32, expansion_factor=6, stride=2)  # 24→32, /2
        
        # Третий блок: t=6
        self.block3 = InvertedParallelBlock(32, 64, expansion_factor=6, stride=2)  # 32→64, /2
        
        # Четвертый блок: t=6
        self.block4 = InvertedParallelBlock(64, 96, expansion_factor=6, stride=1)  # 64→96
        
        # Пятый блок: t=6
        self.block5 = InvertedParallelBlock(96, 160, expansion_factor=6, stride=2)  # 96→160, /2
        
        # Шестой блок: t=6
        self.block6 = InvertedParallelBlock(160, 320, expansion_factor=6, stride=1)  # 160→320

        # Финальный conv point-wise для увеличения каналов
        self.conv_final = nn.Sequential(
            nn.Conv2d(320, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True)
        )

        # Global pooling и классификатор
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        
        # Улучшенный классификатор
        self.classifier = nn.Sequential(
            weight_norm(nn.Linear(512, 256)),
            nn.BatchNorm1d(256),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.4),
            weight_norm(nn.Linear(256, 128)),
            nn.BatchNorm1d(128),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.4),
            weight_norm(nn.Linear(128, num_classes))
        )

        # Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem block
        x = self.stem(x)
        
        # Inverted residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        # Final convolution
        x = self.conv_final(x)
        
        # Global pooling and classification
        x = self.gap(x)
        x = self.flat(x)
        x = self.classifier(x)
        
        return x