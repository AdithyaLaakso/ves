# Model that takes a 256x256 image and predicts a letter out of 24 options, plus a "no letter" option using pytorch.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class SingleLetterModel(nn.Module):
    def __init__(self, num_classes=25):
        super(SingleLetterModel, self).__init__()
        # Load a pre-trained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)
        # Modify the final fully connected layer to match the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
