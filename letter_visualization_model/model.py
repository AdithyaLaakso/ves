import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ClassificationModel(nn.Module):
    def __init__(self, num_classes=26):
        super(ClassificationModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
class ReconstructionModel(nn.Module):
    def __init__(self, pretrained_model: ClassificationModel):
        super(ReconstructionModel, self).__init__()
        if pretrained_model == None:
            self.resnet = models.resnet18(pretrained=True)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 32 * 32 * 3)
            self.model = nn.Sequential(
                self.resnet,
                nn.ReLU(inplace=True),  # Add ReLU activation for non-linearity
                nn.Unflatten(1, (3, 32, 32)),
                nn.Sigmoid()  # Use Sigmoid to ensure output is in [0, 1]
            )
           
        else:
            self.model = pretrained_model
            self.model.resnet.fc = nn.Linear(self.model.resnet.fc.in_features, 32 * 32 * 3)
            self.model = nn.Sequential(
                self.model.resnet,
                nn.ReLU(inplace=True),  # Add ReLU activation for non-linearity
                nn.Unflatten(1, (3, 32, 32)),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.model(x)