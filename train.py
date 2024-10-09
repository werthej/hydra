import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import hydra
import logging
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from mlp import MLP
import os
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
log = logging.getLogger(__name__)

# Initialize loggers
train_logger = logging.getLogger("train_logger")
val_logger = logging.getLogger("val_logger")

# Example function to simulate saving a model's checkpoint
def save_checkpoint(model_state, checkpoint_dir, epoch):
    # Create a filename for the checkpoint
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

    # Save the model's state dictionary
    torch.save(model_state, checkpoint_file)
    print(f"Checkpoint saved at: {checkpoint_file}")

# Function to save the loss curve as an image
def save_loss_curve(output_dir, losses):
    sns.set(style="darkgrid")

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss", color='b', linewidth=2)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Define the path for saving the image
    image_path = os.path.join(output_dir, "loss_curve.png")

    # Save the plot as a PNG image
    plt.savefig(image_path)
    print(f"Loss curve saved at: {image_path}")
    plt.close()

def get_subset(dataset, subset_size_ratio=0.1):
    """Returns a subset of the dataset based on the given ratio."""
    subset_size = int(len(dataset) * subset_size_ratio)
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    return Subset(dataset, indices)

@hydra.main(version_base=None, config_path="config", config_name="config")
def train_model(cfg: DictConfig):
    # Set up data transformation and loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize CIFAR-10
    ])

    full_trainset = torchvision.datasets.CIFAR10(root=cfg.dataset.root, train=True, download=cfg.dataset.download, transform=transform)
    full_testset = torchvision.datasets.CIFAR10(root=cfg.dataset.root, train=False, download=cfg.dataset.download, transform=transform)

    # Subset the dataset (e.g., 10% of the full dataset)
    trainset = get_subset(full_trainset, subset_size_ratio=0.1)
    testset = get_subset(full_testset, subset_size_ratio=0.1)

    trainloader = DataLoader(trainset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2)

    # Initialize the MLP model
    model = MLP(input_size=cfg.model.input_size, hidden_size=cfg.model.hidden_size, num_classes=cfg.model.num_classes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    losses = []
    for epoch in range(cfg.train.epochs):
        train_logger.info(f"Training epoch {epoch}")
        val_logger.info(f"Validating epoch {epoch}")
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
        losses.append(loss)

        # Save model checkpoint at every epoch
        model_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        save_checkpoint(model_state, checkpoint_dir, epoch + 1)

        log.info(f"Epoch [{epoch + 1}/{cfg.train.epochs}]")

    log.info("Training complete.")
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        save_loss_curve(output_dir, losses)
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the 10,000 test images: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    train_model()
