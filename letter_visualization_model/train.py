import os
import torch
from constants import hyperparams_list
from model import SingleLetterModel
from dataset import SingleLetterDataset, SingleLetterDataLoader

# dataset
train_test_data = SingleLetterDataset()
dataset = train_test_data.dataset

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"training on {device}")

# create criterion that compares output and target images with bias against white pixels
class BiasedMSELoss(torch.nn.Module):
    def __init__(self, bias_factor=3.0):
        super().__init__()
        self.bias_factor = bias_factor

    def forward(self, output, target):
        mse = (output - target) ** 2
        penalty_by_pixel = output * self.bias_factor + 1
        mse *= penalty_by_pixel
        reg = torch.mean(output)
        # make var penalty based on the variance of each output channel
        var_penalty = torch.var(output, dim=(2, 3)).mean()
        return mse.mean() - var_penalty * 5

def train_model(batch_size, learning_rate, num_epochs, train_percent, optimizer_class, bias_factor=6.0):
    # Split dataset into train and test sets
    train_size = int(train_percent * len(dataset))
    indices = torch.randperm(len(dataset))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]

    train_loader = SingleLetterDataLoader(train_dataset, batch_size=batch_size, shuffle=True, device=device)
    test_loader = SingleLetterDataLoader(test_dataset, batch_size=batch_size, shuffle=False, device=device)

    model = SingleLetterModel()
    model.to(device)
    criterion = BiasedMSELoss(bias_factor=bias_factor)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
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
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
        avg_test_loss = test_loss / len(test_loader.dataset)
        print(f"Test Loss: {avg_test_loss:.4f}")

    # Save the trained model
    optimizer_name = optimizer_class.__name__ if hasattr(optimizer_class, "__name__") else str(optimizer_class).split(".")[-1].split("'")[0]
    os.makedirs("trained_image_reconstruction_models", exist_ok=True)
    torch.save(model.state_dict(), f"trained_image_reconstruction_models/trained_image_reconstruction_model_{optimizer_name}.pth")
    return epoch_loss, avg_test_loss

# Example: iteratively call train_model with varying hyperparameters


for params in hyperparams_list:
    print(f"\nTraining with hyperparameters: {params}")
    train_loss, test_loss = train_model(**params)
    print(f"Final Train Loss: {train_loss:.4f}, Final Test Loss: {test_loss:.4f}")
