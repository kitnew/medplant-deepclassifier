import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

class ParallelBlock(nn.Module):
    """
    Параллельный блок: две ветки Conv→ReLU→Conv→BN→ReLU, выходы суммируются.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, is_second_block=False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Ветка 1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,  out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # Ветка 2 (идентичная)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,  out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.proj(x)
        out1 = self.relu(self.branch1(x) + identity)
        out2 = self.relu(self.branch2(x) + identity)
        return out1 + out2


class ResidualStream(nn.Module):
    """
    Первый поток предложенной архитектуры (Residual Stream).
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # 1) Input → Conv(3→16, k=3, s=2) → BN → ReLU
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU(inplace=True)

        # 2) Parallel Block 1 (16→16)
        self.parallel1 = ParallelBlock(in_channels=16, out_channels=16, stride=1)

        # 3) Conv(16→32, k=3, s=1) → BN → ReLU
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(32)

        # 4) Parallel Block 2 (32→32)
        self.parallel2 = ParallelBlock(in_channels=32, out_channels=32, stride=1)

        # 5) Conv(32→64, k=3, s=1) → (далее GlobalAvgPool → FC)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Global Average Pooling + классификатор
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc = weight_norm(nn.Linear(64, 32))
        self.bn_fc = nn.BatchNorm1d(32)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.4)
        self.classification = weight_norm(nn.Linear(32, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # начальная свёртка
        x = self.relu(self.bn1(self.conv1(x)))

        # первый параллельный блок
        x = self.parallel1(x)

        # вторая свёртка
        x = self.relu(self.bn2(self.conv2(x)))

        # второй параллельный блок
        x = self.parallel2(x)

        # финальная свёртка → GAP → FC
        x = self.conv3(x)
        x = self.gap(x)              # [B,64,1,1]
        x = self.flat(x)              # [B,64]
        x = self.fc(x)               # [B,32]
        x = self.bn_fc(x)            # Apply batch normalization
        x = self.dropout(x)         # Apply dropout for regularization
        x = self.classification(x)  # Apply classification
        return x

