# Defining LeNet model architecture

import torch
import torch.nn as nn

# Define the LeNet model
class LeNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # feature extraction using sequential layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        # classification using sequential layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features= 16 * 4 * 4, out_features=120),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x