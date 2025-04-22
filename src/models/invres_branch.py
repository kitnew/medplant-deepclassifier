import torch
import torch.nn as nn

class InvertedParallelBlock(nn.Module):
    """
    Параллельный блок для Inverted Residual Stream:
    две ветки, каждая из которых — Conv→ReLU→DepthwiseConv→BN→ReLU, выходы суммируются.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Ветка 1
        self.branch1 = nn.Sequential(
            # обычная свёртка
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU6(inplace=True),
            # depthwise (групповая) свёртка
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                      groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        # Ветка 2 — такая же
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                      groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        if self.in_channels == self.out_channels:
            identity = x
        out1 = self.branch1(x)
        out2 = self.branch2(x)

        if self.in_channels == self.out_channels:
            out1 += identity
            out2 += identity

        return out1 + out2

class InvertedResidualStream(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.bn0   = nn.BatchNorm2d(16)
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu  = nn.ReLU(inplace=True)

        self.block1 = InvertedParallelBlock(16, 16, stride=1)   # 16→16
        self.block2 = InvertedParallelBlock(16, 32, stride=1)   # 16→32
        self.block3 = InvertedParallelBlock(32, 64, stride=1)   # 32→64

        self.conv_final = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_final   = nn.BatchNorm2d(64)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(64, num_classes)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.relu(self.bn_final(self.conv_final(x)))

        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)