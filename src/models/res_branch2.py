import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, is_second_block=False) -> None:
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.is_second_block = is_second_block
        
        # First parallel branch
        self.branch1 = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False
                )
        
        # Second parallel branch
        self.branch2 = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False
                )
        
        # Batch normalization for second block (only one as per specification)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # Process through parallel branches
        # branch 1
        out1 = self.branch1(x)
        if not self.is_second_block:
            out1 = self.bn(out1)
        out1 = self.relu(out1)

        out2 = self.branch2(x)
        if not self.is_second_block:
            out2 = self.bn(out2)
        out2 = self.relu(out2)
        
        # Combine outputs from parallel branches
        out = out1 + out2
        
        # For second block, use only one batch normalization as per specification
        if self.is_second_block:
            out = self.bn(out)
            
        # Apply downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add identity connection
        out += identity
        out = self.relu(out)

        return out

class ResidualStream(nn.Module):
    def __init__(self, num_classes=1000) -> None:
        super(ResidualStream, self).__init__()

        # Initial layer: 224x224x3 -> 112x112x16
        self.initial = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),  # Conv 1
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Расширяем архитектуру до 22 сверточных слоев и ~45 слоев всего
        # Первый набор блоков (4 блока с 16->16 каналами)
        self.stage1_blocks = nn.ModuleList()
        for i in range(4): # Conv 2-9
            # Каждый блок содержит 2 параллельных сверточных слоя
            # Всего 8 сверточных слоев в этой стадии
            self.stage1_blocks.append(ResidualBlock(16, 16, is_second_block=False))
        
        # Второй набор блоков (3 блока с 16->32 каналами)
        # Первый блок с изменением размерности
        self.stage2_block1 = ResidualBlock(16, 32, stride=1, # Conv 10-11
                     downsample=nn.Sequential(
                         nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
                         nn.BatchNorm2d(32)
                     ),
                     is_second_block=True)
        
        # Остальные блоки второй стадии (2 блока)
        self.stage2_blocks = nn.ModuleList()
        for i in range(2): # Conv 13-17
            # Каждый блок содержит 2 параллельных сверточных слоя
            # Всего 4 сверточных слоя в этой части
            self.stage2_blocks.append(ResidualBlock(32, 32, is_second_block=True))
        
        # Третий набор блоков (2 блока с 32->64 каналами)
        # Первый блок с изменением размерности
        self.stage3_block1 = ResidualBlock(32, 64, stride=1, # Conv 18-20
                     downsample=nn.Sequential(
                         nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False), 
                         nn.BatchNorm2d(64)
                     ),
                     is_second_block=True)
        
        # Второй блок третьей стадии
        self.stage3_block2 = ResidualBlock(64, 64, is_second_block=True)  #Conv 21-22
        
        # Группированная свертка (grouped convolution)
        # Упомянуто в спецификации: "extra connections are created between the grouped convolution layer"
        self.grouped_conv = nn.Conv2d(64, 512, kernel_size=3, stride=2, padding=1, groups=32, bias=False)  # Conv 23-24
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        # Финальные слои с дополнительными соединениями
        # "extra connections are created between the grouped convolution layer and the fully connected, 
        # softmax, global average pooling, and classification layers"
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, 1024)
        self.classifier = nn.Linear(1024, num_classes)
        # Проекция для прямого соединения grouped_conv -> classifier
        self.direct_classifier_projection = nn.Linear(512, num_classes)
        # Adding Softmax as per specification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Начальный слой
        x = self.initial(x)  # 224x224x3 -> 112x112x16

        # Первая стадия: 4 блока с 16->16 каналами
        for block in self.stage1_blocks:
            x = block(x)  # Сохраняем размерность 112x112x16
        
        # Сохраняем выход первой стадии
        x1 = x  # 112x112x16
        
        # Вторая стадия: переход 16->32 и еще 2 блока
        x = self.stage2_block1(x)  # 112x112x16 -> 112x112x32
        for block in self.stage2_blocks:
            x = block(x)  # Сохраняем размерность 112x112x32
        
        # Сохраняем выход второй стадии
        x2 = x  # 112x112x32
        
        # Третья стадия: переход 32->64 и еще 1 блок
        x = self.stage3_block1(x)  # 112x112x32 -> 112x112x64
        x = self.stage3_block2(x)  # Сохраняем размерность 112x112x64
        
        # Группированная свертка
        x3 = self.grouped_conv(x)  # 112x112x64 -> 56x56x512
        x3 = self.bn(x3)
        x3 = self.relu(x3)
        

        # Расширенные extra connections между grouped convolution и другими слоями
        # Соединение 1: от x2 (выход второго блока) к классификатору
        x2_pool = self.global_pool(x2)  # 112x112x32 -> 1x1x32
        x2_flat = torch.flatten(x2_pool, 1)  # 1x1x32 -> 32
        
        # Соединение 2: от x3 (выход grouped_conv) через отдельный путь
        # Создаем отдельный канал для x3 (grouped_conv output)
        x3_direct = self.global_pool(x3)  # 56x56x512 -> 1x1x512
        x3_direct_flat = torch.flatten(x3_direct, 1)  # 1x1x512 -> 512
        
        # Соединение 3: Прямой путь от grouped_conv к классификатору
        # Сохраняем копию для прямого соединения с классификатором
        x3_to_classifier = self.global_pool(x3)  # 56x56x512 -> 1x1x512
        x3_to_classifier_flat = torch.flatten(x3_to_classifier, 1)  # 1x1x512 -> 512
        
        # Основной путь для x3
        x3_main = self.global_pool(x3)  # 56x56x512 -> 1x1x512
        x3_main_flat = torch.flatten(x3_main, 1)  # 1x1x512 -> 512
        x3_main_flat = self.dropout(x3_main_flat)
        x3_main_fc = self.fc(x3_main_flat)  # 512 -> 1024
        
        # Обработка прямого соединения от x3
        x3_direct_fc = F.linear(x3_direct_flat, self.fc.weight, self.fc.bias)  # 512 -> 1024
        
        # Объединение всех путей
        # Подготовка x2_flat для сложения (паддинг до размера 1024)
        if x2_flat.size(1) < x3_main_fc.size(1):
            x2_flat = F.pad(x2_flat, (0, x3_main_fc.size(1) - x2_flat.size(1)))
        
        # Объединение всех соединений
        combined = x3_main_fc + x2_flat + x3_direct_fc
        
        # Преобразование x3_to_classifier_flat для прямого соединения с классификатором
        # Используем заранее определенный слой проекции
        x3_to_classifier_projected = self.direct_classifier_projection(x3_to_classifier_flat)
        
        # Классификатор
        output = self.classifier(combined)  # 1024 -> num_classes
        
        # Добавляем прямое соединение от grouped_conv к классификатору
        output = output + x3_to_classifier_projected
        
        # Применение Softmax как указано в спецификации
        output = self.softmax(output)
        
        return output