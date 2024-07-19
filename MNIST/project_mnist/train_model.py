import click
import torch
import os

from models.model import MyAwesomeModel
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple
import matplotlib.pyplot as plt


def corrupt_mnist(DATA_PATH) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    train_set, train_target = [], []
   
    train_set = torch.load(f"{DATA_PATH}/train_images.pt")
    train_target = torch.load(f"{DATA_PATH}/train_target.pt")

    test_set = torch.load(f"{DATA_PATH}/test_images.pt")
    test_target = torch.load(f"{DATA_PATH}/test_target.pt")
    
    train_set = torch.utils.data.TensorDataset(train_set, train_target)
    test_set = torch.utils.data.TensorDataset(test_set, test_target)

    return train_set, test_set

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--project_name", default="", required=False, help="Name of the project directory")
@click.option("--processed_dir", default="data/processed", help="Path to processed data directory")
def train(lr, project_name: str, processed_dir: str):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
   # Construct paths using the project_name
    raw_dir = os.path.join(project_name, processed_dir)
    processed_dir = os.path.join(project_name, processed_dir)
    train_set, _ = corrupt_mnist(processed_dir)
    trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    epochs = 10
    training_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            # Ensure the images have the correct shape
            if images.dim() == 3:
                images = images.unsqueeze(1)  # Add a channel dimension if missing
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_set)
        training_losses.append(avg_loss)    
        # Print average loss per epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_set):.4f}")

    # Save the trained model checkpoint
    torch.save(model.state_dict(), "trained_model2.pth")
    print("Training completed and model saved.")
    
    # Plot and save the training loss curve
    plt.figure()
    plt.plot(range(1, epochs+1), training_losses, marker='o', linestyle='-', color='b')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/training_loss_curve.png")
    plt.show()


 



if __name__ == "__main__":
    train()