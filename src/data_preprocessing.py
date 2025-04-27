import os
import torch
import random
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import config

class LungCancerDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        # Map the class indices to meaningful names
        self.class_names = [config.CLASS_NAMES[cls] for cls in self.classes]

def get_data_transforms():
    """Define data transformations for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    return train_transform, val_transform

def create_data_loaders():
    """Create data loaders for training and validation"""
    train_transform, val_transform = get_data_transforms()

    # Create the full dataset
    full_dataset = LungCancerDataset(
        root=config.TRAIN_DATA_PATH,
        transform=train_transform
    )

    # Split the dataset into train and test sets (80% train, 20% test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Apply validation transform to test set
    test_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    return train_loader, test_loader, full_dataset.class_names

if __name__ == "__main__":
    # Test the data loaders
    train_loader, test_loader, classes = create_data_loaders()
    print(f"Classes: {classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}") 