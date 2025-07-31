import os
import torch
from constants import hyperparams_list
from model import ClassificationModel
from dataset import SingleLetterClassificationDataLoader, SingleLetterClassificationDataset

# dataset
train_test_data = SingleLetterClassificationDataset()
dataset = train_test_data.dataset

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"training on {device}")


def train_model(batch_size, learning_rate, num_epochs, train_percent, optimizer_class):
    # Split dataset into train and test sets
    train_size = int(train_percent * len(dataset))
    indices = torch.randperm(len(dataset))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]

    train_loader = SingleLetterClassificationDataLoader(train_dataset, batch_size=batch_size, shuffle=True, device=device)
    test_loader = SingleLetterClassificationDataLoader(test_dataset, batch_size=batch_size, shuffle=False, device=device)

    model = ClassificationModel()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {100 * correct / total:.2f}%")

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
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {100 * correct / total:.2f}%")

    # Save the trained model
    optimizer_name = optimizer_class.__name__ if hasattr(optimizer_class, "__name__") else str(optimizer_class).split(".")[-1].split("'")[0]
    os.makedirs("trained_image_classification_models", exist_ok=True)
    torch.save(model.state_dict(), f"trained_image_classification_models/trained_image_classification_model_{optimizer_name}.pth")
    return epoch_loss, avg_test_loss

# Example: iteratively call train_model with varying hyperparameters


for params in hyperparams_list:
    print(f"\nTraining with hyperparameters: {params}")
    train_loss, test_loss = train_model(**params)
    print(f"Final Train Loss: {train_loss:.4f}, Final Test Loss: {test_loss:.4f}")
