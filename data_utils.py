"""
Data utilities for loading and preprocessing MNIST and CIFAR-10 datasets.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(
    batch_size: int = 100,
    data_dir: str = "./data",
    num_workers: int = 0,
) -> tuple:
    """
    Get MNIST data loaders with proper normalization.
    
    Images are flattened to 784-dim vectors and normalized to unit vectors.
    
    Args:
        batch_size: Batch size for training
        data_dir: Directory to store/load data
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Transform: flatten and normalize to unit vectors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten to 784
        transforms.Lambda(lambda x: x / (x.norm() + 1e-8)),  # Unit vector
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, test_loader


def get_cifar10_loaders(
    batch_size: int = 100,
    data_dir: str = "./data",
    num_workers: int = 0,
) -> tuple:
    """
    Get CIFAR-10 data loaders with proper normalization.
    
    Images are flattened to 3072-dim vectors and normalized to unit vectors.
    
    Args:
        batch_size: Batch size for training
        data_dir: Directory to store/load data
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Transform: flatten and normalize to unit vectors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten to 3072
        transforms.Lambda(lambda x: x / (x.norm() + 1e-8)),  # Unit vector
    ])
    
    train_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Test data loading
    train_loader, test_loader = get_mnist_loaders()
    print(f"MNIST train dataset size: {len(train_loader.dataset)}")
    print(f"MNIST test dataset size: {len(test_loader.dataset)}")

    print(f"CIFAR-10 train dataset size: {len(train_loader.dataset)}")
    print(f"CIFAR-10 test dataset size: {len(test_loader.dataset)}")