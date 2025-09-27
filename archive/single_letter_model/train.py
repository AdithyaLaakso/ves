import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from model import SingleLetterModel  # Replace with your actual model class name
from dataset import SingleLetterDataset, SingleLetterDataLoader # Replace with your actual dataset class name

# Hyperparameters
batch_size = 32
learning_rate = 1e-5
num_epochs = 5
train_percent = 0.95  # Percentage of data to use for training

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
