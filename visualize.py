"""
Visualization utilities for weight visualization and learning curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def draw_weights(
    weights: np.ndarray,
    img_shape: tuple,
    n_cols: int = 20,
    n_rows: int = 10,
    title: str = "Learned Weights",
    save_path: str = None,
    cmap: str = "gray",
) -> plt.Figure:
    """
    Visualize weight vectors as a grid of images.
    
    This function creates a grid visualization of receptive fields.
    For bio-learned weights, you should see "ghost" digits with
    negative ink halos due to the anti-Hebbian delta term.
    For backprop weights, they will appear as random noise.
    
    Args:
        weights: Weight matrix of shape (n_hidden, input_dim)
        img_shape: Original image shape, e.g., (28, 28) for MNIST
        n_cols: Number of columns in the grid
        n_rows: Number of rows in the grid
        title: Plot title
        save_path: Path to save the figure (optional)
        cmap: Colormap to use
        
    Returns:
        matplotlib Figure object
    """
    n_show = min(n_cols * n_rows, weights.shape[0])
    
    # Reshape weights to images
    weight_imgs = weights[:n_show].reshape(n_show, *img_shape)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.8, n_rows * 0.8))
    fig.suptitle(title, fontsize=14)
    
    # Flatten axes for easy iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        ax.axis("off")
        
        if idx < n_show:
            img = weight_imgs[idx]
            # Normalize for visualization
            vmax = max(abs(img.min()), abs(img.max()))
            ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved weight visualization to {save_path}")
    
    return fig


def draw_weights_color(
    weights: np.ndarray,
    img_shape: tuple = (3, 32, 32),
    n_cols: int = 20,
    n_rows: int = 10,
    title: str = "Learned Weights",
    save_path: str = None,
) -> plt.Figure:
    """
    Visualize color image weight vectors as a grid.
    
    For CIFAR-10: weights should reshape to (C, H, W) = (3, 32, 32)
    
    Args:
        weights: Weight matrix of shape (n_hidden, input_dim)
        img_shape: Shape as (C, H, W), e.g., (3, 32, 32) for CIFAR-10
        n_cols: Number of columns in the grid
        n_rows: Number of rows in the grid
        title: Plot title
        save_path: Path to save the figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    n_show = min(n_cols * n_rows, weights.shape[0])
    C, H, W = img_shape
    
    # Reshape weights to images (C, H, W) -> (H, W, C)
    weight_imgs = weights[:n_show].reshape(n_show, C, H, W)
    weight_imgs = np.transpose(weight_imgs, (0, 2, 3, 1))  # (n, H, W, C)
    
    # Normalize to [0, 1] for display
    weight_imgs = (weight_imgs - weight_imgs.min()) / (weight_imgs.max() - weight_imgs.min() + 1e-8)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.8, n_rows * 0.8))
    fig.suptitle(title, fontsize=14)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        ax.axis("off")
        
        if idx < n_show:
            ax.imshow(weight_imgs[idx])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved weight visualization to {save_path}")
    
    return fig


def plot_learning_curves(
    bio_history: dict,
    bp_history: dict = None,
    title: str = "Learning Curves",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot test error vs epochs for bio-inspired and backprop methods.
    
    Args:
        bio_history: Dict with 'test_error' list from bio training
        bp_history: Dict with 'test_error' list from backprop training (optional)
        title: Plot title
        save_path: Path to save the figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bio curve
    epochs_bio = range(1, len(bio_history["test_error"]) + 1)
    ax.plot(epochs_bio, bio_history["test_error"], label="Bio-Inspired", linewidth=2, color="blue")
    
    # Plot backprop curve if provided
    if bp_history is not None:
        epochs_bp = range(1, len(bp_history["test_error"]) + 1)
        ax.plot(epochs_bp, bp_history["test_error"], label="Backprop", linewidth=2, color="red")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Error (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to show reasonable range
    min_err = min(bio_history["test_error"])
    if bp_history:
        min_err = min(min_err, min(bp_history["test_error"]))
    ax.set_ylim(bottom=0, top=max(10, min_err * 3))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved learning curves to {save_path}")
    
    return fig
