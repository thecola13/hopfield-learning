"""
Visualization utilities for weight visualization and learning curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import io


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
    
    # Global normalization (key for consistent scale)
    all_weights = weight_imgs.reshape(-1, img_shape[0]*img_shape[1])
    vmax = np.max(np.abs(all_weights))
    vmin = -vmax  # Symmetric for weights
    
    # Create figure with space for colorbar
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.8 + 1.5, n_rows * 0.8))
    fig.suptitle(title, fontsize=14)
    
    # Flatten axes
    axes_flat = axes.flat if n_rows > 1 or n_cols > 1 else axes.reshape(1, -1)
    
    images = []  # Collect mappables for colorbar
    
    for idx in range(n_rows * n_cols):
        ax = axes_flat[idx]
        ax.axis("off")
        
        if idx < n_show:
            img = ax.imshow(
                weight_imgs[idx], cmap=cmap, vmin=vmin, vmax=vmax
            )
            images.append(img)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave 10% space on right
    
    # Add single colorbar on right (using first image as mappable representative)
    cbar = fig.colorbar(images[0], ax=axes, shrink=0.6, pad=0.02, aspect=20)
    cbar.set_label("Weight Value", rotation=270, labelpad=20)
    
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
    Plot train and test accuracy vs epochs for bio-inspired and backprop methods.
    
    Args:
        bio_history: Dict with 'train_error' and 'test_error' lists from bio training
        bp_history: Dict with 'train_error' and 'test_error' lists from backprop training (optional)
        title: Plot title
        save_path: Path to save the figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert error to accuracy
    bio_train_acc = [100 - e for e in bio_history["train_error"]] if "train_error" in bio_history else None
    bio_test_acc = [100 - e for e in bio_history["test_error"]]
    
    epochs_bio = range(1, len(bio_history["test_error"]) + 1)
    
    # Plot bio curves (train: blue, test: orange)
    if bio_train_acc:
        ax.plot(epochs_bio, bio_train_acc, label="Bio Train", linewidth=2, color="blue")
    ax.plot(epochs_bio, bio_test_acc, label="Bio Test", linewidth=2, color="orange")
    
    # Plot backprop curves if provided (train: pink, test: green)
    if bp_history is not None:
        bp_train_acc = [100 - e for e in bp_history["train_error"]] if "train_error" in bp_history else None
        bp_test_acc = [100 - e for e in bp_history["test_error"]]
        epochs_bp = range(1, len(bp_history["test_error"]) + 1)
        
        if bp_train_acc:
            ax.plot(epochs_bp, bp_train_acc, label="BP Train", linewidth=2, color="pink")
        ax.plot(epochs_bp, bp_test_acc, label="BP Test", linewidth=2, color="green")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to show reasonable range for accuracy
    max_acc = max(bio_test_acc)
    if bp_history:
        max_acc = max(max_acc, max([100 - e for e in bp_history["test_error"]]))
    ax.set_ylim(bottom=min(90, max_acc - 10), top=100)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved learning curves to {save_path}")
    
    return fig


def draw_weights_to_array(
    weights: np.ndarray,
    img_shape: tuple,
    n_cols: int = 20,
    n_rows: int = 10,
    title: str = "Learned Weights",
    cmap: str = "RdBu_r",
) -> np.ndarray:
    """
    Render weight visualization to a numpy array (for video frames).
    
    Args:
        weights: Weight matrix of shape (n_hidden, input_dim)
        img_shape: Original image shape, e.g., (28, 28) for MNIST
        n_cols: Number of columns in the grid
        n_rows: Number of rows in the grid
        title: Plot title
        cmap: Colormap to use
        
    Returns:
        numpy array of shape (H, W, 3) representing the rendered image
    """
    n_show = min(n_cols * n_rows, weights.shape[0])
    
    # Reshape weights to images
    weight_imgs = weights[:n_show].reshape(n_show, *img_shape)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.8, n_rows * 0.8), dpi=100)
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
            if vmax > 0:
                ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                ax.imshow(img, cmap=cmap)
    
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    
    # Get the RGBA buffer from the figure
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    
    # Convert from RGBA to RGB
    img_rgb = img_array[:, :, :3].copy()
    
    plt.close(fig)
    
    return img_rgb


def draw_weights_color_to_array(
    weights: np.ndarray,
    img_shape: tuple = (3, 32, 32),
    n_cols: int = 20,
    n_rows: int = 10,
    title: str = "Learned Weights",
) -> np.ndarray:
    """
    Render color weight visualization to a numpy array (for video frames).
    
    For CIFAR-10: weights should reshape to (C, H, W) = (3, 32, 32)
    
    Args:
        weights: Weight matrix of shape (n_hidden, input_dim)
        img_shape: Shape as (C, H, W), e.g., (3, 32, 32) for CIFAR-10
        n_cols: Number of columns in the grid
        n_rows: Number of rows in the grid
        title: Plot title
        
    Returns:
        numpy array of shape (H, W, 3) representing the rendered image
    """
    n_show = min(n_cols * n_rows, weights.shape[0])
    C, H, W = img_shape
    
    # Reshape weights to images (C, H, W) -> (H, W, C)
    weight_imgs = weights[:n_show].reshape(n_show, C, H, W)
    weight_imgs = np.transpose(weight_imgs, (0, 2, 3, 1))  # (n, H, W, C)
    
    # Normalize to [0, 1] for display
    weight_imgs = (weight_imgs - weight_imgs.min()) / (weight_imgs.max() - weight_imgs.min() + 1e-8)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.8, n_rows * 0.8), dpi=100)
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
    
    # Convert figure to numpy array
    fig.canvas.draw()
    
    # Get the RGBA buffer from the figure
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    
    # Convert from RGBA to RGB
    img_rgb = img_array[:, :, :3].copy()
    
    plt.close(fig)
    
    return img_rgb


def create_weight_evolution_video(
    weight_snapshots: List[dict],
    img_shape: tuple,
    save_path: str,
    n_cols: int = 20,
    n_rows: int = 10,
    fps: int = 5,
    cmap: str = "RdBu_r",
    color: bool = False,
) -> str:
    """
    Create a video showing weight evolution during unsupervised learning.
    
    Args:
        weight_snapshots: List of dicts with 'epoch', 'weights' keys
        img_shape: Original image shape, e.g., (28, 28) for MNIST or (3, 32, 32) for CIFAR-10
        save_path: Path to save the video (should end with .mp4 or .gif)
        n_cols: Number of columns in the weight grid
        n_rows: Number of rows in the weight grid
        fps: Frames per second
        cmap: Colormap to use (for grayscale images)
        color: If True, treat images as color (C, H, W) format
        
    Returns:
        Path where the video was saved
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import imageio as iio
        except ImportError:
            print("Warning: imageio not installed. Cannot create video.")
            print("Install with: pip install imageio[ffmpeg]")
            return None
    
    print(f"Creating weight evolution video with {len(weight_snapshots)} frames...")
    
    frames = []
    for snapshot in weight_snapshots:
        epoch = snapshot['epoch']
        weights = snapshot['weights']
        
        title = f"Weight Evolution - Epoch {epoch}"
        if color:
            frame = draw_weights_color_to_array(
                weights, img_shape, n_cols, n_rows, title
            )
        else:
            frame = draw_weights_to_array(
                weights, img_shape, n_cols, n_rows, title, cmap
            )
        frames.append(frame)
    
    # Ensure parent directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write video
    try:
        if save_path.endswith('.gif'):
            iio.imwrite(save_path, frames, duration=1000//fps, loop=0)
        else:
            # For mp4, try to use pyav or ffmpeg plugin
            try:
                iio.imwrite(save_path, frames, fps=fps, codec='libx264')
            except Exception:
                # Fallback: try with default settings
                iio.imwrite(save_path, frames, fps=fps)
        
        print(f"Saved weight evolution video to {save_path}")
        return save_path
    except Exception as e:
        print(f"Warning: Could not save video to {save_path}: {e}")
        return None


def save_weight_evolution_frames(
    weight_snapshots: List[dict],
    img_shape: tuple,
    save_dir: str,
    n_cols: int = 20,
    n_rows: int = 10,
    cmap: str = "RdBu_r",
    color: bool = False,
) -> List[str]:
    """
    Save individual weight evolution frames as images.
    
    Args:
        weight_snapshots: List of dicts with 'epoch', 'weights' keys
        img_shape: Original image shape, e.g., (28, 28) for MNIST or (3, 32, 32) for CIFAR-10
        save_dir: Directory to save frames
        n_cols: Number of columns in the weight grid
        n_rows: Number of rows in the weight grid
        cmap: Colormap to use (for grayscale images)
        color: If True, treat images as color (C, H, W) format
        
    Returns:
        List of paths to saved frames
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    for snapshot in weight_snapshots:
        epoch = snapshot['epoch']
        weights = snapshot['weights']
        
        save_path = save_dir / f"weights_epoch_{epoch:04d}.png"
        if color:
            draw_weights_color(
                weights, img_shape, n_cols, n_rows,
                title=f"Weight Evolution - Epoch {epoch}",
                save_path=str(save_path),
            )
        else:
            draw_weights(
                weights, img_shape, n_cols, n_rows,
                title=f"Weight Evolution - Epoch {epoch}",
                save_path=str(save_path),
                cmap=cmap,
            )
        saved_paths.append(str(save_path))
        plt.close('all')  # Clean up figures
    
    return saved_paths


def draw_random_feature_detectors(
    weights: np.ndarray,
    img_shape: tuple,
    n_detectors: int = 20,
    title: str = "Randomly Chosen Feature Detectors",
    save_path: str = None,
    cmap: str = "RdBu_r",
    seed: int = 42,
) -> plt.Figure:
    """
    Visualize randomly chosen feature detectors (weights) as in the paper.
    
    This reproduces the paper's visualization showing "Twenty randomly chosen 
    feature detectors of 2,000 are shown".
    
    Args:
        weights: Weight matrix of shape (n_hidden, input_dim)
        img_shape: Original image shape, e.g., (28, 28) for MNIST
        n_detectors: Number of random detectors to show (default: 20)
        title: Plot title
        save_path: Path to save the figure (optional)
        cmap: Colormap to use
        seed: Random seed for reproducibility
        
    Returns:
        matplotlib Figure object
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Randomly select feature detectors
    n_total = weights.shape[0]
    selected_indices = np.random.choice(n_total, size=min(n_detectors, n_total), replace=False)
    selected_weights = weights[selected_indices]
    
    # Determine grid layout (try to make it roughly square)
    n_show = len(selected_indices)
    n_cols = min(5, n_show)  # Max 5 columns
    n_rows = (n_show + n_cols - 1) // n_cols
    
    # Reshape weights to images
    weight_imgs = selected_weights.reshape(n_show, *img_shape)
    
    # Global normalization (key for consistent scale)
    all_weights = weight_imgs.reshape(-1, img_shape[0]*img_shape[1])
    vmax = np.max(np.abs(all_weights))
    vmin = -vmax  # Symmetric for weights
    
    # Create figure with space for colorbar
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5 + 1.5, n_rows * 1.5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Flatten axes
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes_flat = axes.flat
    else:
        axes_flat = axes.flat
    
    images = []  # Collect mappables for colorbar
    
    for idx in range(n_rows * n_cols):
        ax = axes_flat[idx]
        ax.axis("off")
        
        if idx < n_show:
            img = ax.imshow(
                weight_imgs[idx], cmap=cmap, vmin=vmin, vmax=vmax
            )
            # Add detector index as title
            ax.set_title(f"#{selected_indices[idx]}", fontsize=9)
            images.append(img)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])  # Leave space for colorbar and title
    
    # Add single colorbar on right
    if images:
        cbar = fig.colorbar(images[0], ax=axes, shrink=0.8, pad=0.02, aspect=20)
        cbar.set_label("Weight Value", rotation=270, labelpad=20)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved random feature detectors to {save_path}")
    
    return fig


def draw_random_feature_detectors_color(
    weights: np.ndarray,
    img_shape: tuple = (3, 32, 32),
    n_detectors: int = 20,
    title: str = "Randomly Chosen Feature Detectors",
    save_path: str = None,
    seed: int = 42,
) -> plt.Figure:
    """
    Visualize randomly chosen color feature detectors as in the paper.
    
    For CIFAR-10: weights should reshape to (C, H, W) = (3, 32, 32)
    
    Args:
        weights: Weight matrix of shape (n_hidden, input_dim)
        img_shape: Shape as (C, H, W), e.g., (3, 32, 32) for CIFAR-10
        n_detectors: Number of random detectors to show (default: 20)
        title: Plot title
        save_path: Path to save the figure (optional)
        seed: Random seed for reproducibility
        
    Returns:
        matplotlib Figure object
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Randomly select feature detectors
    n_total = weights.shape[0]
    selected_indices = np.random.choice(n_total, size=min(n_detectors, n_total), replace=False)
    selected_weights = weights[selected_indices]
    
    # Determine grid layout
    n_show = len(selected_indices)
    n_cols = min(5, n_show)
    n_rows = (n_show + n_cols - 1) // n_cols
    
    C, H, W = img_shape
    
    # Reshape weights to images (C, H, W) -> (H, W, C)
    weight_imgs = selected_weights.reshape(n_show, C, H, W)
    weight_imgs = np.transpose(weight_imgs, (0, 2, 3, 1))  # (n, H, W, C)
    
    # Normalize to [0, 1] for display
    weight_imgs = (weight_imgs - weight_imgs.min()) / (weight_imgs.max() - weight_imgs.min() + 1e-8)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes_flat = axes.flat
    else:
        axes_flat = axes.flat
    
    for idx in range(n_rows * n_cols):
        ax = axes_flat[idx]
        ax.axis("off")
        
        if idx < n_show:
            ax.imshow(weight_imgs[idx])
            ax.set_title(f"#{selected_indices[idx]}", fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved random feature detectors to {save_path}")
    
    return fig


def plot_error_rates(
    bio_history: dict,
    bp_history: dict,
    title: str = "Error Rate on Training and Test Sets",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot error rates (not accuracy) for bio and backprop methods.
    
    This reproduces the paper's visualization: "Error rate on the training and 
    test sets as training progresses for the proposed biological algorithm and 
    for the standard network trained end-to-end."
    
    Args:
        bio_history: Dict with 'train_error' and 'test_error' lists from bio training
        bp_history: Dict with 'train_error' and 'test_error' lists from backprop training
        title: Plot title
        save_path: Path to save the figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get error rates (already in percentage)
    bio_train_error = bio_history.get("train_error", [])
    bio_test_error = bio_history["test_error"]
    bp_train_error = bp_history.get("train_error", [])
    bp_test_error = bp_history["test_error"]
    
    epochs_bio = range(1, len(bio_test_error) + 1)
    epochs_bp = range(1, len(bp_test_error) + 1)
    
    # Plot bio curves (solid lines)
    if bio_train_error:
        ax.plot(epochs_bio, bio_train_error, label="Bio Train", 
                linewidth=2.5, color="#1f77b4", linestyle="-")
    ax.plot(epochs_bio, bio_test_error, label="Bio Test", 
            linewidth=2.5, color="#ff7f0e", linestyle="-")
    
    # Plot backprop curves (dashed lines)
    if bp_train_error:
        ax.plot(epochs_bp, bp_train_error, label="Backprop Train", 
                linewidth=2.5, color="#2ca02c", linestyle="--")
    ax.plot(epochs_bp, bp_test_error, label="Backprop Test", 
            linewidth=2.5, color="#d62728", linestyle="--")
    
    ax.set_xlabel("Epoch", fontsize=13, fontweight='bold')
    ax.set_ylabel("Error Rate (%)", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set y-axis to show error rate (typically 0-10% for MNIST)
    all_errors = bio_test_error + bp_test_error
    if bio_train_error:
        all_errors.extend(bio_train_error)
    if bp_train_error:
        all_errors.extend(bp_train_error)
    
    max_error = max(all_errors)
    min_error = min(all_errors)
    
    # Set reasonable y-axis limits
    y_margin = (max_error - min_error) * 0.1
    ax.set_ylim(bottom=max(0, min_error - y_margin), top=max_error + y_margin)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved error rate plot to {save_path}")
    
    return fig
