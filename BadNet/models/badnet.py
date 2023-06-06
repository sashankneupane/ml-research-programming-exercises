from torch import nn

class BadNet(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=.5),
            nn.Linear(32 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, output_channels),
        )

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x