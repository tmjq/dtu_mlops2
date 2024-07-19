import click
import torch
from models.model import MyAwesomeModel
import os

def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()

# Function to load true labels
def load_true_labels(target_path: str) -> torch.Tensor:
    """Load true labels from file."""
    return torch.load(target_path)

def predict(model: torch.nn.Module, input_data: torch.Tensor) -> torch.Tensor:
    """Predict using the model."""
    # Normalize input data
    input_data = normalize(input_data)
    input_data = input_data.unsqueeze(1)  # Add channel dimension if necessary

    # Make predictions
    with torch.no_grad():
        predictions = model(input_data)
    
    return predictions

@click.command()
@click.argument('model_path', type=str)
@click.argument('input_path', type=click.Path(exists=True))

def main(model_path, input_path):
# Load model state_dict
    model_state_dict = torch.load(model_path)

    # Instantiate your model
    model = MyAwesomeModel()
    model.load_state_dict(model_state_dict)
    model.eval()

    # Load input data (assuming input_path is a test_set.pt file)
    input_data = torch.load(input_path)

    # Load true labels
    true_labels = load_true_labels("data/raw/test_target.pt")

    # Predict using the model
    predictions = predict(model, input_data)

    # Compute accuracy
    _, predicted_classes = torch.max(predictions, 1)
    correct = (predicted_classes == true_labels).sum().item()
    accuracy = correct / true_labels.size(0) * 100.0

    print(f'Accuracy on the test set: {accuracy:.2f}%')

if __name__ == "__main__":
    main()
