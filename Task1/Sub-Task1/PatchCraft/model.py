import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class PreprocessBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.Hardtanh()
        )
    def forward(self, x):
        return self.block(x)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rich = PreprocessBlock()
        self.poor = PreprocessBlock()
        self.layers = nn.Sequential(
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            nn.AvgPool2d(2, 2),
            ConvBlock(),
            ConvBlock(),
            nn.AvgPool2d(2, 2),
            ConvBlock(),
            ConvBlock(),
            nn.AvgPool2d(2,2),
            ConvBlock(),
            ConvBlock(),
            nn.AvgPool2d(2,2 ),
            nn.Flatten(),
            nn.Linear(8192, 1)
        )
    def forward(self, x_rich, x_poor):
        x_rich = self.rich(x_rich)
        x_poor = self.poor(x_poor)
        x = x_rich-x_poor
        x = self.layers(x)
        x = nn.Sigmoid()(x)
        return x


