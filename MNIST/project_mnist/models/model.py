import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        #print("Shape before conv1:", x.shape)  # Check shape before conv1
        x = torch.relu(self.conv1(x))
        
        #print("Shape after conv1:", x.shape)  # Check shape after conv1
        x = torch.max_pool2d(x, 2, 2)
        #print("Shape after max_pool2d 1:", x.shape)  # Check shape after max_pool2d 1
        x = torch.relu(self.conv2(x))
        #print("Shape after conv2:", x.shape)  # Check shape after conv2
        x = torch.max_pool2d(x, 2, 2)
        #print("Shape after max_pool2d 2:", x.shape)  # Check shape after max_pool2d 2
        x = torch.relu(self.conv3(x))
        #print("Shape after conv3:", x.shape)  # Check shape after conv3
        x = torch.max_pool2d(x, 2, 2)
        #print("Shape after max_pool2d 3:", x.shape)  # Check shape after max_pool2d 3
        x = torch.flatten(x, 1)
        #print("Shape after flatten:", x.shape)  # Check shape after flatten
        
        x = self.dropout(x)
        #print("Shape after dropout:", x.shape)  # Check shape after dropout
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")