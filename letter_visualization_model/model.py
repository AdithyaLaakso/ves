import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SingleLetterModel(nn.Module):
    def __init__(self):
        super(SingleLetterModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 32 * 32 * 3)
        self.model = nn.Sequential(
            self.resnet,
            nn.ReLU(inplace=True),  # Add ReLU activation for non-linearity
            nn.Unflatten(1, (3, 32, 32)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
