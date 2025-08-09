DATA_PATH = "./paths.json"
MAX_SIZE = 10780
IMG_PATH = 0
LABEL = 1
import json
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from PIL import Image
import torchvision.models as models
# TODO: Change sizes to 224x224
class SingleLetterDataset:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.dataset = self.load_dataset()
        self.class_to_index = self.innit_class_name_to_index()
    def load_dataset(self):
        """Load the dataset from the JSON file."""
        with open(self.data_path, "r") as f:
            all_data = json.load(f)['paths']
            if len(all_data) > MAX_SIZE:
                indices = np.random.choice(len(all_data), MAX_SIZE, replace=False)
                data = [all_data[i] for i in indices]
            else:
                data = all_data
        return data
    def get_class_name(self):
        labels = [item[LABEL] for item in self.dataset]
        return list(set(labels))
    def innit_class_name_to_index(self):
        class_names = self.get_class_names()
        return {name: index for index, name in enumerate(class_names)}
class SingleLetterDataLoader:
    def __init__(self, dataset, class_to_index, batch_size=32, shuffle=True, device="cpu"):
        self.dataset = dataset
        self.class_to_index = class_to_index
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
    def resnet_normalize(self, imgs: Tensor) -> Tensor:
        """Normalize the image tensor using ResNet normalization."""
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        imgs -= mean
        imgs /= std
        return imgs
    def get_class_name_to_index(self):
        """Create a mapping from class names to indices."""
        class_names = self.dataset.get_class_names()
        return {name: index for index, name in enumerate(class_names)}
    def label_to_index(self, label):
        if not hasattr(self, 'class_to_index'):
            self.class_to_index = self.get_class_name_to_index()
        return self.class_to_index.get(label, -1)
    def __iter__(self):
        data = self.dataset
        if self.shuffle:
            np.random.shuffle(data)
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size]
            imgs = [(np.array(Image.open(item[IMG_PATH]).convert("RGB"))/255.0) for item in batch_data]
            # Ensure images are identical in shape
            if len(imgs) == 0:
                continue
            img_shape = imgs[0].shape
            for img in imgs:
                if img.shape != img_shape:
                    raise ValueError(f"Image shape mismatch: expected {img_shape}, got {img.shape}")
            labels = [self.class_to_index[item[LABEL]] for item in batch_data]
            # reshape images to (batch_size, channels, height, width)
            imgs = np.array([np.transpose(img, (2, 0, 1)) for img in imgs])
            # Convert images to tensor
            imgs = torch.tensor(imgs, dtype=torch.float32).to(self.device)
            # Normalize images
            imgs = self.resnet_normalize(imgs).to(self.device)
            # Ensure labels are in tensor format
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
            yield imgs, labels

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

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 10
    train_percent = 0.8  # Percentage of data to use for training

    # dataset
    train_test_data = SingleLetterDataset()  # Initialize your dataset
    dataset = train_test_data.dataset  # Access the dataset attribute
    label_dict = train_test_data.class_to_index  # Access the class to index mapping

    # Model, Loss, Optimizer
    model = SingleLetterModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Split dataset into train and test sets
    train_size = int(train_percent * len(dataset))
    test_size = len(dataset) - train_size
    indices = torch.randperm(len(dataset))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]

    # Gpu settings
    gpu_avail = torch.cuda.is_available()
    device = torch.device("cuda:0" if gpu_avail else "cpu")
    model.to(device)
    print(f"Training on device: {device}")

    train_loader = SingleLetterDataLoader(train_dataset,
                                          class_to_index=label_dict,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          device=device)
    test_loader = SingleLetterDataLoader(test_dataset,
                                         class_to_index=label_dict,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         device=device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

        # Evaluation on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        avg_test_loss = test_loss / len(test_loader.dataset)
    accuracy = correct / total
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pth")
