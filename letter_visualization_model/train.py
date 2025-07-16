import torch
from torch.utils.data import DataLoader

from model import SingleLetterModel
from dataset import SingleLetterDataset, SingleLetterDataLoader

# Hyperparameters
batch_size = 32
learning_rate = 1e-3
num_epochs = 10
train_percent = 0.8

# dataset
train_test_data = SingleLetterDataset()
dataset = train_test_data.dataset

# Model, Loss, Optimizer
model = SingleLetterModel()
criterion = torch.nn.MSELoss()  # For image reconstruction
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Split dataset into train and test sets
train_size = int(train_percent * len(dataset))
test_size = len(dataset) - train_size
indices = torch.randperm(len(dataset))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
train_dataset = [dataset[i] for i in train_indices]
test_dataset = [dataset[i] for i in test_indices]

train_loader = SingleLetterDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = SingleLetterDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:  # targets are now images
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
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "trained_model.pth")
