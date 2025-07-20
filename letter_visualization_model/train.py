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

#select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"training on {device}")

# Model, Loss, Optimizer
model = SingleLetterModel()
model.to(device)
# criterion = torch.nn.MSELoss()  # For image reconstruction
# create criterion that compares output and target images with bias against white pixels
class BiasedMSELoss(torch.nn.Module):
    def __init__(self, white_threshold=0.95, bias_factor=6.0):
        super().__init__()
        self.white_threshold = white_threshold
        self.bias_factor = bias_factor

    def forward(self, output, target):
        # Assume output and target are in [0,1], shape (batch, channels, H, W)
        mse = (output - target) ** 2
        # Mask: where output is "white" but target is "dark"
        white_guess = output > self.white_threshold
        dark_target = target < (1.0 - self.white_threshold)
        bias_mask = white_guess & dark_target
        mse[bias_mask] *= self.bias_factor
        return mse.mean()

criterion = BiasedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Split dataset into train and test sets
train_size = int(train_percent * len(dataset))
test_size = len(dataset) - train_size
indices = torch.randperm(len(dataset))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
train_dataset = [dataset[i] for i in train_indices]
test_dataset = [dataset[i] for i in test_indices]

train_loader = SingleLetterDataLoader(train_dataset, batch_size=batch_size, shuffle=True, device=device)
test_loader = SingleLetterDataLoader(test_dataset, batch_size=batch_size, shuffle=False, device=device)

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
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
torch.save(model.state_dict(), "trained_image_reconstruction_model.pth")
