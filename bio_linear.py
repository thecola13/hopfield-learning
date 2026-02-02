"""
Bio-inspired linear layer with ranking-based unsupervised learning.

Implements the "Fast Implementation" from Krotov & Hopfield (2019)
"Unsupervised learning by competing hidden units"

Parity notes for NumPy reference (old.py) matching:
- dtype='float64': NumPy defaults to float64; use this for trajectory matching
- normalize_init=False: old.py uses randn/sqrt(in_dim) without row normalization
- shuffle=True: old.py reshuffles entire dataset every epoch before batching
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, RandomSampler

import torch.nn.functional as F

def paper_loss(c, y, m: int):
    # y: (B,) class indices
    # build one-hot in {0,1}
    oh01 = F.one_hot(y, num_classes=c.size(1)).to(dtype=c.dtype)
    # convert to {-1,+1}: correct class +1, others -1
    t = oh01 * 2 - 1
    # L_m, averaged over batch (paper sums; averaging is equivalent up to scaling LR)
    return torch.mean(torch.sum(torch.abs(c - t) ** m, dim=1))


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
        dtype: Tensor dtype, 'float32' or 'float64' (default: 'float64' for NumPy parity)
        device: Device to create weights on (default: 'cpu')
        normalize_init: If True, L2-normalize weight rows at init (default: False for NumPy parity)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        p: float = 3.0,
        k: int = 7,
        delta: float = 0.4,
        dtype: str = 'float64',
        device: str = 'cpu',
        normalize_init: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.k = k
        self.delta = delta
        
        # Resolve dtype
        self._dtype_str = dtype
        torch_dtype = torch.float64 if dtype == 'float64' else torch.float32
        
        # MPS doesn't support float64 - fall back to float32 with warning
        if device == 'mps' and torch_dtype == torch.float64:
            import warnings
            warnings.warn(
                "MPS device doesn't support float64. Falling back to float32. "
                "For exact NumPy parity, use device='cpu' with dtype='float64'."
            )
            torch_dtype = torch.float32
            self._dtype_str = 'float32'
        
        weights = torch.rand(out_features, in_features, dtype=torch_dtype, device=device)

        if normalize_init:
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
        # Cast input to same dtype as weights for precision parity
        x = x.to(dtype=self.weight.dtype)
        
        # Compute I_μ = sgn(W) * |W|^(p-1) · v
        sig = torch.sign(self.weight)
        W_transformed = sig * torch.abs(self.weight) ** (self.p - 1)
        return torch.mm(x, W_transformed.t())
    
    @torch.no_grad()
    def unsupervised_update(self, x: torch.Tensor, lr: float) -> None:
        """
        Perform one unsupervised learning update using ranking-based competition.
        
        Args:
            x: Input batch of shape (batch_size, in_features)
            lr: Current learning rate (should decay linearly)
        """
        # Cast input to same dtype as weights
        x = x.to(dtype=self.weight.dtype)
        
        batch_size = x.shape[0]
        device = self.weight.device
        
        synapses = self.weight.data
        inputs = x.T  # Shape: (in_features, batch_size)
        
        # 1. Compute similarity/activation (Optimized for p=2)
        if self.p == 2.0:
            tot_input = torch.mm(synapses, inputs)
        else:
            sig = torch.sign(synapses)
            tot_input = torch.mm(sig * torch.abs(synapses) ** (self.p - 1), inputs)
        
        # 2. Ranking-based competition (Optimized with topk)
        # indices[0] = Winner (Rank 1)
        # indices[k-1] = K-th runner up (Rank k)
        vals, indices = torch.topk(tot_input, k=self.k, dim=0, sorted=True)
        
        winner_indices = indices[0, :]
        kth_indices = indices[self.k-1, :]
        
        # Create sparse plasticity coefficients g
        yl = torch.zeros_like(tot_input)
        yl.scatter_(0, winner_indices.unsqueeze(0), 1.0)
        yl.scatter_(0, kth_indices.unsqueeze(0), -self.delta)
        
        # 3. Compute weight update
        xx = torch.sum(yl * tot_input, dim=1)
        
        # ds = g(Q) * x - (g(Q) * y) * w
        ds = torch.mm(yl, inputs.T) - (xx[:, None] * synapses)
        
        # Normalize by max absolute change
        nc = torch.max(torch.abs(ds))
        if nc < 1e-30: nc = 1e-30

        synapses.add_(lr * (ds / nc))
    
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
        dtype: Tensor dtype, 'float32' or 'float64' (default: 'float64' for NumPy parity)
        device: Device for BioLinear weights (default: 'cpu')
        normalize_init: If True, L2-normalize BioLinear weight rows at init (default: False)
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
        dtype: str = 'float64',
        device: str = 'cpu',
        normalize_init: bool = False,
    ):
        super().__init__()
        self.bio_layer = BioLinear(
            in_features, hidden_features, p, k, delta,
            dtype=dtype, device=device, normalize_init=normalize_init
        )
        # Readout uses same dtype as bio_layer (handles MPS float64->float32 fallback)
        torch_dtype = self.bio_layer.weight.dtype
        self.readout = nn.Linear(hidden_features, num_classes).to(dtype=torch_dtype, device=device)
        self.n = n
        self._dtype_str = self.bio_layer._dtype_str
        self.beta = 1.0
        self.m = 6
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Bio layer activation
        h = self.bio_layer(x)
        # ReLU^n activation
        h = torch.relu(h) ** self.n
        # Readout layer
        logits = self.readout(h)
        c = torch.tanh(self.beta * logits)

        return c
    
    def train_unsupervised(
        self,
        dataloader_or_dataset,
        epochs: int = 300,
        lr_start: float = 0.04,
        lr_end: float = 0.0,
        device: str = "cpu",
        shuffle: bool = True,
        seed: Optional[int] = None,
        deterministic: bool = False,
        batch_size: Optional[int] = None,
        record_evolution: bool = False,
        snapshot_interval: int = 10,
    ):
        """
        Phase 1: Unsupervised training of the BioLinear layer.
        
        For parity with NumPy reference (old.py), use shuffle=True (default) to
        reshuffle the entire dataset at the start of each epoch.
        
        Args:
            dataloader_or_dataset: DataLoader or Dataset providing input batches.
                If a Dataset is passed and shuffle=True, a new DataLoader is built
                each epoch with fresh shuffling.
            epochs: Number of training epochs
            lr_start: Initial learning rate
            lr_end: Final learning rate (linear decay)
            device: Device to train on
            shuffle: If True, reshuffle data every epoch (default True for NumPy parity)
            seed: Optional random seed for reproducibility
            deterministic: If True, enable PyTorch deterministic mode (may slow training)
            batch_size: Batch size (inferred from dataloader if not provided)
            record_evolution: If True, capture weight snapshots during training
            snapshot_interval: Epochs between weight snapshots (when record_evolution=True)
            
        Returns:
            List of weight snapshots if record_evolution=True, else None
        """
        from tqdm import tqdm
        from torch.utils.data import Dataset
        
        # Optional reproducibility settings
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
        
        self.to(device)
        
        # Weight snapshots storage
        weight_snapshots = [] if record_evolution else None
        
        # Capture initial weights if recording evolution
        if record_evolution:
            initial_weights = self.bio_layer.weight.data.cpu().numpy().copy()
            weight_snapshots.append({'epoch': 0, 'weights': initial_weights})
        
        # Determine if we have a DataLoader or Dataset
        is_dataset = isinstance(dataloader_or_dataset, Dataset)
        
        if is_dataset:
            dataset = dataloader_or_dataset
            if batch_size is None:
                raise ValueError("batch_size must be provided when passing a Dataset")
            # Build dataloader with shuffle setting
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            dataloader = dataloader_or_dataset
            # Infer batch_size from dataloader if not provided
            if batch_size is None:
                batch_size = dataloader.batch_size
            # If shuffle requested, extract dataset to rebuild each epoch
            if shuffle and hasattr(dataloader, 'dataset'):
                dataset = dataloader.dataset
                is_dataset = True  # Treat as dataset to rebuild each epoch
        
        num_batches_per_epoch = len(dataloader)
        total_batches = epochs * num_batches_per_epoch
        current_batch = 0
        
        pbar = tqdm(range(epochs), desc="Unsupervised training")
        for epoch in pbar:
            # Rebuild DataLoader each epoch if shuffle is requested (parity with old.py)
            if shuffle and is_dataset:
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for batch_idx, (data, _) in enumerate(dataloader):
                # Linear learning rate decay
                progress = current_batch / total_batches
                lr = lr_start + (lr_end - lr_start) * progress
                
                # Flatten and move to device (dtype cast happens in bio_layer)
                data = data.view(data.size(0), -1).to(device)
                
                # Unsupervised update
                self.bio_layer.unsupervised_update(data, lr)
                
                current_batch += 1
            
            # Capture snapshot every N epochs if recording evolution
            if record_evolution and (epoch + 1) % snapshot_interval == 0:
                weights = self.bio_layer.weight.data.cpu().numpy().copy()
                weight_snapshots.append({'epoch': epoch + 1, 'weights': weights})
                pbar.set_postfix({"lr": f"{lr:.5f}", "epoch": epoch + 1, "snapshots": len(weight_snapshots)})
            else:
                pbar.set_postfix({"lr": f"{lr:.5f}", "epoch": epoch + 1})
        
        return weight_snapshots
    
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
        
        history = {"train_loss": [], "train_error": [], "test_error": []}
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
                c = self(data)
                loss = paper_loss(c, target, m = self.m)
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
