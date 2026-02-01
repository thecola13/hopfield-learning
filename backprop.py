"""
Backpropagation baseline for comparison with bio-inspired learning.

Standard MLP trained end-to-end with Adam optimizer.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class BackpropNetwork(nn.Module):
    """
    Standard MLP trained with backpropagation.
    
    Architecture matches the bio-network:
    - Linear: input_dim -> hidden_dim
    - ReLU activation
    - Linear: hidden_dim -> num_classes
    
    Args:
        in_features: Input dimension
        hidden_features: Number of hidden units
        num_classes: Number of output classes
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
    ):
        super().__init__()
        self.hidden = nn.Linear(in_features, hidden_features)
        self.readout = nn.Linear(hidden_features, num_classes)
        self.relu = nn.ReLU()
        
        # Initialize weights
        nn.init.kaiming_normal_(self.hidden.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_normal_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        h = self.hidden(x)
        h = self.relu(h)
        return self.readout(h)
    
    def train_supervised(
        self,
        train_loader,
        test_loader,
        epochs: int = 300,
        lr: float = 0.001,
        device: str = "cpu",
    ):
        """
        Train the network end-to-end with backpropagation.
        
        Args:
            train_loader: Training DataLoader
            test_loader: Test DataLoader
            epochs: Number of training epochs
            lr: Learning rate for Adam optimizer
            device: Device to train on
            
        Returns:
            Dictionary with training history
        """
        self.to(device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        history = {"train_loss": [], "train_error": [], "test_error": []}
        
        pbar = tqdm(range(epochs), desc="Backprop training")
        for epoch in pbar:
            # Training
            self.train()
            total_loss = 0
            for data, target in train_loader:
                data = data.view(data.size(0), -1).to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            history["train_loss"].append(avg_loss)
            
            # Evaluation on train and test
            train_error = self.evaluate(train_loader, device)
            history["train_error"].append(train_error)
            test_error = self.evaluate(test_loader, device)
            history["test_error"].append(test_error)
            
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "error": f"{test_error:.2f}%"
            })
        
        return history
    
    @torch.no_grad()
    def evaluate(self, dataloader, device: str = "cpu") -> float:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader to evaluate on
            device: Device to use
            
        Returns:
            Error rate as percentage
        """
        self.eval()
        correct = 0
        total = 0
        
        for data, target in dataloader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            
            output = self(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        
        error = 100.0 * (1 - correct / total)
        return error
    
    def get_weight_images(self, img_shape: tuple):
        """
        Get first layer weights as images.
        
        Args:
            img_shape: Original image shape, e.g., (28, 28)
            
        Returns:
            Array of shape (hidden_features, *img_shape)
        """
        import numpy as np
        weights = self.hidden.weight.data.cpu().numpy()
        return weights.reshape(weights.shape[0], *img_shape)
