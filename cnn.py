import torch
from torch import nn
from torchsummary import summary

class ShotDetectionNetwork(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels = hidden_units,
                kernel_size = 4,
                padding = 1,
                stride = 2
            ),

            nn.Conv2d(
                in_channels=hidden_units,
                out_channels = int(hidden_units/4),
                kernel_size = 4,
                padding = 1,
                stride = 2
            ),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels= int(hidden_units/4),
                out_channels = 1,
                kernel_size = 2,
                padding = 1,
                stride = 2
            ),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=1),
        )
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        logits = self.classifier(y)
        return logits
    


class ShotDetectionNetwork2(nn.Module):
    def __init__(self, hidden_units, num_layers):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels = hidden_units,
                kernel_size = 4,
                padding = 1,
                stride = 2
            ),

            nn.Conv2d(
                in_channels=hidden_units,
                out_channels = int(hidden_units/4),
                kernel_size = 4,
                padding = 1,
                stride = 2
            ),
            nn.MaxPool2d(kernel_size=2),
        )

        self.additional_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=int(hidden_units / 4),
                out_channels=int(hidden_units / 4),
                kernel_size=3,
                padding=1,
                stride=1
            ) for _ in range(num_layers)
        ])

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels= int(hidden_units/4),
                out_channels = 1,
                kernel_size = 2,
                padding = 1,
                stride = 2
            ),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=1),
        )
    
    def forward(self, x):
        y = self.conv1(x)
        for layer in self.additional_layers:
            y = layer(y)
        y = self.conv2(y)
        logits = self.classifier(y)
        return logits


if __name__ == "__main__":
    cnn = ShotDetectionNetwork(hidden_units=64)
    summary(cnn, (1, 128, 128))