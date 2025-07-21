# Model that takes a 256x256 image and predicts a letter out of 24 options, plus a "no letter" option using pytorch.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class SingleLetterModel(nn.Module):
    def __init__(self):
        super(SingleLetterModel, self).__init__()
        # use a pre-trained ResNet18 model and modify the final layers such that it outputs 32x32x3 images
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 32 * 32 * 3)  # Change the output layer to match the image size
        self.model = nn.Sequential(
            self.resnet,
            nn.Unflatten(1, (3, 32, 32)),  # Unflatten the output to match the image shape
            nn.Conv2d(3, 3, kernel_size=1),  # Ensure the output is still 3 channels
            nn.Sigmoid()  # Use Sigmoid to ensure outputs are in the range [0, 1]
        )
    def forward(self, x):
        return self.model(x)
