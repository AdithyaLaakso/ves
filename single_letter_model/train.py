import torch
from torch.utils.data import DataLoader

from model import SingleLetterModel  # Replace with your actual model class name
from dataset import SingleLetterDataLoader # Replace with your actual dataset class name



# Hyperparameters
batch_size = 32
learning_rate = 1e-3
num_epochs = 10

# DataLoader
train_loader = SingleLetterDataLoader(batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
model = SingleLetterModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# TODO: ADD TRAINING LOGIC
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    i = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        i += 1
        print(i)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / 1000#train_loader.data_size
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "trained_model.pth")