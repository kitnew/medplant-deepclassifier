from re import I
import torch
import torch.nn as nn

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InvertedResidualBlock, self).__init__()

        # Две параллельные свёрточные ветви
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.end = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Параллельные свёртки
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        
        # Суммирование выходов
        y = y1 + y2
        y = self.end(y)
        return y

class InvertedResidualStream(nn.Module):
    def __init__(self, num_classes=1000):
        super(InvertedResidualStream, self).__init__()

        # Начальный блок: BatchNorm + Conv3x3 stride=2
        # Вход: 224x224x3 -> Выход: 112x112x16
        self.initial = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        )

        # Первый блок: две параллельные свертки + BN + ReLU
        # Вход: 112x112x16 -> Выход: 112x112x16
        self.stage1 = InvertedResidualBlock(16, 32)

        # Второй блок: две параллельные свертки + BN + ReLU
        # Вход: 112x112x32 -> Выход: 112x112x32
        self.stage2 = InvertedResidualBlock(32, 64)

        # Третий блок: ReLU -> Conv3x3 -> ReLU -> Conv3x3 -> BN
        # Вход: 112x112x32 -> Выход: 112x112x64
        self.end = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Добавляем слой классификации (не указан в спецификации, но необходим)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Начальный блок
        x = self.initial(x)  # 224x224x3 -> 112x112x16

        # Первый блок
        x = self.stage1(x)   # 112x112x16 -> 112x112x16
        
        # Второй блок
        x = self.stage2(x)   # 112x112x32 -> 112x112x32

        # Третий блок
        x = self.end(x)      # 112x112x32 -> 112x112x64
        
        # Классификатор
        x = self.classifier(x)  # 112x112x64 -> num_classes
        
        return x