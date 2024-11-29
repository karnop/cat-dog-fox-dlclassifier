import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from loguru import logger

# transformation pipeline for preprocessing
def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224,224)),  # Resize images to 224x224
            transforms.RandomHorizontalFlip(),  # Random horizontal flip for data augmentation
            transforms.RandomRotation(30),  # Random rotation for data augmentation
            transforms.ToTensor(),  # Convert image to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])  # Normalize using ImageNet values
        ]
    )

# function to load dataset
def load_data(data_dir, batch_size=32):
    try:
        logger.info("Entering data preprocessing and loading")
        # get the transformations
        transform = get_transforms()

        # load train. validation and test datasets
        train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

        # create Dataloader for batching and shuffling
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Data loaded with batch size {batch_size}.")
        return train_loader, val_loader, test_loader
    
    except Exception as e:
        logger.error(f"Error in loading data: {e}")
        raise

split_data_dir = "data/split"
load_data(split_data_dir)