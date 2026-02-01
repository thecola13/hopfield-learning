"""
Bio-inspired linear layer with ranking-based unsupervised learning.

Implements the "Fast Implementation" from Krotov & Hopfield (2019)
"Unsupervised learning by competing hidden units"
"""

import torch
import torch.nn as nn
import numpy as np


class BioLinear(nn.Module):
    """
    Bio-inspired linear layer using ranking-based competitive learning.
    
    The layer learns via unsupervised competition between hidden units:
    - Activation: I_μ = sgn(W) * |W|^(p-1) · v  (Lebesgue p-norm similarity)
    - Competition: Winner gets g=1, k-th runner-up gets g=-Δ
    - Update: ΔW ∝ g·v - g⟨W,v⟩·W (keeps weights on p-norm sphere)
    
    Args:
        in_features: Size of input (flattened image dimension)
        out_features: Number of hidden units
        p: Lebesgue norm parameter (default: 3 for MNIST, 2 for CIFAR-10)
        k: Ranking parameter for anti-Hebbian term (default: 7)
        delta: Anti-Hebbian strength (default: 0.4)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        p: float = 3.0,
        k: int = 7,
        delta: float = 0.4,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.k = k
        self.delta = delta
        
        # Initialize weights randomly and normalize using L^p norm
        weights = torch.randn(out_features, in_features)
        # Normalize each row to unit L^p norm: sum(|W_i|^p) = 1
        lp_norm = (torch.abs(weights) ** p).sum(dim=1, keepdim=True) ** (1.0 / p)
        weights = weights / lp_norm
        self.weight = nn.Parameter(weights, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes activations using p-norm similarity.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Activations of shape (batch_size, out_features)
        """
        # Compute I_μ = sgn(W) * |W|^(p-1) · v
        sig = torch.sign(self.weight)
        W_transformed = sig * torch.abs(self.weight) ** (self.p - 1)
        return torch.mm(x, W_transformed.t())
    
    @torch.no_grad()
    def unsupervised_update(self, x: torch.Tensor, lr: float) -> None:
        """
        Perform one unsupervised learning update using ranking-based competition.
        
        Pure PyTorch implementation for GPU acceleration.
        
        Args:
            x: Input batch of shape (batch_size, in_features)
            lr: Current learning rate (should decay linearly)
        """
        batch_size = x.shape[0]
        hid = self.out_features
        device = self.weight.device
        
        # Work directly with tensors on device
        synapses = self.weight.data
        inputs = x.T  # Shape: (in_features, batch_size)
        
        # 1. Compute similarity/activation using p-norm
        sig = torch.sign(synapses)
        tot_input = torch.mm(sig * torch.abs(synapses) ** (self.p - 1), inputs)
        # tot_input shape: (out_features, batch_size)
        
        # 2. Ranking-based competition
        y = torch.argsort(tot_input, dim=0)  # Sort hidden units by activation
        yl = torch.zeros(hid, batch_size, device=device)
        
        # Winner (highest activation): g = 1
        yl[y[hid - 1, :], torch.arange(batch_size, device=device)] = 1.0
        
        # k-th runner-up: g = -delta (anti-Hebbian)
        yl[y[hid - self.k, :], torch.arange(batch_size, device=device)] = -self.delta
        
        # 3. Compute weight update with L^p normalization
        # Compute the inner product ⟨W,v⟩ using the p-norm metric
        # ⟨W,v⟩ = sum_i |W_i|^(p-2) * W_i * v_i = sum_i sign(W_i) * |W_i|^(p-1) * v_i
        # This is already computed in tot_input

        # xx = sum over batch of (g * ⟨W,v⟩) for each hidden unit
        xx = torch.sum(yl * tot_input, dim=1)

        # Update rule from Eq. 3: τ_L dW/dt = g(Q) * [R^p * v - ⟨W,v⟩ * W]
        # In the fast implementation with ranking, g(Q) is absorbed into yl
        # The update is: ΔW = lr * [g * v - g * ⟨W,v⟩ * W]
        ds = torch.mm(yl, inputs.T) - (xx[:, None] * synapses)

        # Normalize by max absolute change for stability
        nc = torch.max(torch.abs(ds))
        if nc < 1e-30:
            nc = 1e-30

        # Apply update in-place
        synapses.add_(lr * (ds / nc))

        # Renormalize weights to L^p sphere after update
        # This ensures the constraint sum(|W_i|^p) = 1 is maintained
        lp_norm = (torch.abs(synapses) ** self.p).sum(dim=1, keepdim=True) ** (1.0 / self.p)
        # Avoid division by zero
        lp_norm = torch.clamp(lp_norm, min=1e-8)
        synapses.div_(lp_norm)
    
    def get_weight_images(self, img_shape: tuple) -> np.ndarray:
        """
        Reshape weights to visualizable images.
        
        Args:
            img_shape: Original image shape, e.g., (28, 28) for MNIST
            
        Returns:
            Array of shape (out_features, *img_shape)
        """
        weights = self.weight.data.cpu().numpy()
        return weights.reshape(self.out_features, *img_shape)


class BioNetwork(nn.Module):
    """
    Complete bio-inspired network with unsupervised BioLinear layer
    and supervised readout layer.
    
    Architecture:
    - BioLinear: Unsupervised feature extraction
    - ReLU^n: Non-linear activation  
    - Linear: Supervised classification
    
    Args:
        in_features: Input dimension
        hidden_features: Number of hidden units
        num_classes: Number of output classes
        p: Lebesgue norm for BioLinear
        k: Ranking parameter
        delta: Anti-Hebbian strength
        n: Exponent for ReLU^n activation (4.5 for MNIST, 10 for CIFAR-10)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        p: float = 3.0,
        k: int = 7,
        delta: float = 0.4,
        n: float = 4.5,
    ):
        super().__init__()
        self.bio_layer = BioLinear(in_features, hidden_features, p, k, delta)
        self.readout = nn.Linear(hidden_features, num_classes)
        self.n = n
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Bio layer activation
        h = self.bio_layer(x)
        # ReLU^n activation
        h = torch.relu(h) ** self.n
        # Readout layer
        return self.readout(h)
    
    def train_unsupervised(
        self,
        dataloader,
        epochs: int = 1000,
        lr_start: float = 0.04,
        lr_end: float = 0.0,
        device: str = "cpu",
    ):
        """
        Phase 1: Unsupervised training of the BioLinear layer.
        
        Args:
            dataloader: DataLoader providing input batches
            epochs: Number of training epochs
            lr_start: Initial learning rate
            lr_end: Final learning rate (linear decay)
            device: Device to train on
        """
        from tqdm import tqdm
        
        self.to(device)
        total_batches = epochs * len(dataloader)
        current_batch = 0
        
        pbar = tqdm(range(epochs), desc="Unsupervised training")
        for epoch in pbar:
            for batch_idx, (data, _) in enumerate(dataloader):
                # Linear learning rate decay
                progress = current_batch / total_batches
                lr = lr_start + (lr_end - lr_start) * progress
                
                # Flatten and normalize input
                data = data.view(data.size(0), -1).to(device)
                
                # Unsupervised update
                self.bio_layer.unsupervised_update(data, lr)
                
                current_batch += 1
            
            pbar.set_postfix({"lr": f"{lr:.5f}"})
    
    def train_supervised(
        self,
        train_loader,
        test_loader,
        epochs: int = 300,
        lr_schedule: list = None,
        device: str = "cpu",
    ):
        """
        Phase 2: Supervised training of the readout layer.
        
        Args:
            train_loader: Training DataLoader
            test_loader: Test DataLoader
            epochs: Number of training epochs
            lr_schedule: List of (epoch, lr) tuples for LR schedule
            device: Device to train on
            
        Returns:
            Dictionary with training history
        """
        from tqdm import tqdm
        
        if lr_schedule is None:
            # Default MNIST schedule: 0.001 for 100 epochs, halve every 50
            lr_schedule = [(0, 0.001), (100, 0.0005), (150, 0.00025), (200, 0.000125), (250, 0.0000625)]
        
        self.to(device)
        # Freeze bio layer
        self.bio_layer.weight.requires_grad = False
        
        optimizer = torch.optim.Adam(self.readout.parameters(), lr=lr_schedule[0][1])
        criterion = nn.CrossEntropyLoss()
        
        history = {"train_loss": [], "test_error": []}
        schedule_idx = 0
        
        pbar = tqdm(range(epochs), desc="Supervised training")
        for epoch in pbar:
            # Update learning rate
            while schedule_idx < len(lr_schedule) - 1 and epoch >= lr_schedule[schedule_idx + 1][0]:
                schedule_idx += 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_schedule[schedule_idx][1]
            
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
            
            # Evaluation
            test_error = self.evaluate(test_loader, device)
            history["test_error"].append(test_error)
            
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "error": f"{test_error:.2f}%",
                "lr": f"{lr_schedule[schedule_idx][1]:.6f}"
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
