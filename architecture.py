from torch import nn
import torch.nn.functional as F


class WildLifeModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential()

        self.base = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classification_headers = nn.ModuleList(
            [
                nn.Conv2d(64, (num_classes + 1) * 4, kernel_size=3, padding=1),
                nn.Conv2d(128, (num_classes + 1) * 4, kernel_size=3, padding=1),
                nn.Conv2d(128, (num_classes + 1) * 4, kernel_size=3, padding=1),
            ]
        )

        self.regression_headers = nn.ModuleList(
            [
                nn.Conv2d(64, 4, kernel_size=3, padding=1),
                nn.Conv2d(128, 4, kernel_size=3, padding=1),
                nn.Conv2d(128, 4, kernel_size=3, padding=1),
            ]
        )

        def forward(self, x):
            detection_features = []

            x = self.base(x)
            detection_features.append(x)

            for i in range(len(self.classification_headers)):
                classification = self.classification_headers[i](x)
                regression = self.regression_headers[i](x)

                detection_features.append(classification)
                detection_features.append(regression)

                if i < len(self.classification_headers) - 1:
                    x = F.relu(x)

            return detection_features
