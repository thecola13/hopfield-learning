"""
Reproduce results from Krotov & Hopfield (2019)
"Unsupervised learning by competing hidden units"

This script reproduces:
- Figure 3: MNIST weights and accuracy comparison
- Figure 7: CIFAR-10 results

Usage:
    uv run python reproduce_results.py [--mnist] [--cifar10] [--device cpu|cuda|mps]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import torch

from bio_linear import BioNetwork
from backprop import BackpropNetwork
from data_utils import get_mnist_loaders, get_cifar10_loaders
from visualize import (
    draw_weights, draw_weights_color, plot_learning_curves,
    create_weight_evolution_video, save_weight_evolution_frames,
)


def create_run_dir(base_dir: str, experiment_name: str) -> Path:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_hyperparams(run_dir: Path, hyperparams: dict, results: dict = None):
    """Save hyperparameters and results to a log file."""
    log_path = run_dir / "hyperparams.log"
    
    with open(log_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Run: {run_dir.name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("HYPERPARAMETERS\n")
        f.write("-" * 40 + "\n")
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")
        
        if results:
            f.write("\nRESULTS\n")
            f.write("-" * 40 + "\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
    
    # Also save as JSON for programmatic access
    json_path = run_dir / "hyperparams.json"
    with open(json_path, "w") as f:
        json.dump({"hyperparams": hyperparams, "results": results or {}}, f, indent=2)
    
    print(f"Saved hyperparams to: {log_path}")


def run_mnist_experiment(
    device: str = "cpu",
    figures_dir: str = "figures",
    record_evolution: bool = False,
    snapshot_interval: int = 10,
):
    """
    Reproduce Figure 3: MNIST weights and accuracy comparison.
    
    Phase 1: Unsupervised training of BioLinear layer
    - p = 3, k = 7, delta = 0.4
    - 1000 epochs, LR 0.04 -> 0
    
    Phase 2: Supervised training of readout layer
    - ReLU^4.5 activation
    - Adam, LR schedule: 0.001 for 100 epochs, halve every 50
    
    Baseline: End-to-end backprop with same architecture
    
    Args:
        device: Device to use for training
        figures_dir: Directory to save figures
        record_evolution: If True, capture weight snapshots during unsupervised training
        snapshot_interval: Epochs between weight snapshots (when record_evolution=True)
    """
    print("\n" + "="*60)
    print("MNIST EXPERIMENT (Figure 3)")
    print("="*60)
    
    # Create timestamped run directory
    run_dir = create_run_dir(figures_dir, "mnist")
    print(f"Run directory: {run_dir}")
    
    # Hyperparameters from Appendix B
    INPUT_DIM = 784
    HIDDEN_DIM = 2000
    NUM_CLASSES = 10
    
    # Bio params
    P = 3.0
    K = 7
    DELTA = 0.4
    N = 4.5  # ReLU^n exponent
    
    # Training params
    PHASE1_EPOCHS = 1000
    PHASE1_LR_START = 0.04
    PHASE1_LR_END = 0.0
    PHASE2_EPOCHS = 300
    BATCH_SIZE = 100
    
    # Collect hyperparameters for logging
    hyperparams = {
        "experiment": "MNIST",
        "device": device,
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": NUM_CLASSES,
        "p": P,
        "k": K,
        "delta": DELTA,
        "n": N,
        "phase1_epochs": PHASE1_EPOCHS,
        "phase1_lr_start": PHASE1_LR_START,
        "phase1_lr_end": PHASE1_LR_END,
        "phase2_epochs": PHASE2_EPOCHS,
        "batch_size": BATCH_SIZE,
        "record_evolution": record_evolution,
        "snapshot_interval": snapshot_interval if record_evolution else None,
    }
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE)
    
    # =========================================================
    # Bio-Inspired Network
    # =========================================================
    print("\n--- Bio-Inspired Network ---")
    
    bio_net = BioNetwork(
        in_features=INPUT_DIM,
        hidden_features=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        p=P,
        k=K,
        delta=DELTA,
        n=N,
        # Parity with NumPy reference (old.py):
        dtype='float64',       # NumPy defaults to float64
        normalize_init=False,  # old.py uses randn/sqrt(in_dim) without row normalization
        device=device,
    )
    
    # Phase 1: Unsupervised learning
    print("\nPhase 1: Unsupervised training...")
    weight_snapshots = bio_net.train_unsupervised(
        train_loader,
        epochs=PHASE1_EPOCHS,
        lr_start=PHASE1_LR_START,
        lr_end=PHASE1_LR_END,
        device=device,
        shuffle=True,  # Reshuffle every epoch for parity with old.py
        record_evolution=record_evolution,
        snapshot_interval=snapshot_interval,
    )
    
    # Create weight evolution video if snapshots were recorded
    if record_evolution and weight_snapshots:
        print(f"\nCreating weight evolution video with {len(weight_snapshots)} frames...")
        
        # Save individual frames
        frames_dir = run_dir / "evolution_frames"
        save_weight_evolution_frames(
            weight_snapshots,
            img_shape=(28, 28),
            save_dir=str(frames_dir),
            n_cols=20,
            n_rows=10,
            cmap="RdBu_r",
        )
        
        # Create video/gif
        video_path = str(run_dir / "weight_evolution.gif")
        create_weight_evolution_video(
            weight_snapshots,
            img_shape=(28, 28),
            save_path=video_path,
            n_cols=20,
            n_rows=10,
            fps=3,
            cmap="RdBu_r",
        )
        print(f"Weight evolution video saved to {video_path}")
    
    # Visualize bio weights (should show ghost digits)
    bio_weights = bio_net.bio_layer.weight.data.cpu().numpy()
    draw_weights(
        bio_weights,
        img_shape=(28, 28),
        n_cols=20,
        n_rows=10,
        title="Bio-Inspired Weights (MNIST) - Ghost Digits Expected",
        save_path=str(run_dir / "mnist_bio_weights.png"),
        cmap="RdBu_r"
    )
    
    # Phase 2: Supervised learning
    print("\nPhase 2: Supervised training...")
    lr_schedule = [
        (0, 0.001),
        (100, 0.0005),
        (150, 0.00025),
        (200, 0.000125),
        (250, 0.0000625),
    ]
    
    bio_history = bio_net.train_supervised(
        train_loader,
        test_loader,
        epochs=PHASE2_EPOCHS,
        lr_schedule=lr_schedule,
        device=device,
    )
    
    bio_final_error = bio_history["test_error"][-1]
    print(f"\nBio-Inspired Final Test Error: {bio_final_error:.2f}%")
    
    # =========================================================
    # Backprop Baseline
    # =========================================================
    print("\n--- Backprop Baseline ---")
    
    bp_net = BackpropNetwork(
        in_features=INPUT_DIM,
        hidden_features=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
    )
    
    bp_history = bp_net.train_supervised(
        train_loader,
        test_loader,
        epochs=PHASE2_EPOCHS,
        lr=0.001,
        device=device,
    )
    
    bp_final_error = bp_history["test_error"][-1]
    print(f"\nBackprop Final Test Error: {bp_final_error:.2f}%")
    
    # Visualize backprop weights (should look like noise)
    bp_weights = bp_net.get_weight_images((28, 28))
    draw_weights(
        bp_weights,
        img_shape=(28, 28),
        n_cols=20,
        n_rows=10,
        title="Backprop Weights (MNIST) - Random Noise Expected",
        save_path=str(run_dir / "mnist_bp_weights.png"),
        cmap="RdBu_r"
    )
    
    # Plot learning curves
    plot_learning_curves(
        bio_history,
        bp_history,
        title="MNIST: Accuracy vs Epoch",
        save_path=str(run_dir / "mnist_learning_curves.png"),
    )
    
    # Save hyperparameters and results
    results = {
        "bio_final_error": f"{bio_final_error:.2f}%",
        "bp_final_error": f"{bp_final_error:.2f}%",
        "bio_min_error": f"{min(bio_history['test_error']):.2f}%",
        "bp_min_error": f"{min(bp_history['test_error']):.2f}%",
    }
    save_hyperparams(run_dir, hyperparams, results)
    
    # Summary
    print("\n" + "="*60)
    print("MNIST RESULTS SUMMARY")
    print("="*60)
    print(f"Bio-Inspired Error: {bio_final_error:.2f}% (target: ~1.46%)")
    print(f"Backprop Error:     {bp_final_error:.2f}% (target: ~1.50%)")
    print(f"\nFigures saved to: {run_dir}/")
    
    return bio_history, bp_history


def run_cifar10_experiment(
    device: str = "cpu",
    figures_dir: str = "figures",
    record_evolution: bool = False,
    snapshot_interval: int = 10,
):
    """
    Reproduce Figure 7: CIFAR-10 results.
    
    Phase 1: Unsupervised training of BioLinear layer
    - p = 2, k = 2, delta = 0.0
    
    Phase 2: Supervised training of readout layer
    - ReLU^10 activation
    
    Baseline: End-to-end backprop with same architecture
    
    Args:
        device: Device to use for training
        figures_dir: Directory to save figures
        record_evolution: If True, capture weight snapshots during unsupervised training
        snapshot_interval: Epochs between weight snapshots (when record_evolution=True)
    """
    print("\n" + "="*60)
    print("CIFAR-10 EXPERIMENT (Figure 7)")
    print("="*60)
    
    # Create timestamped run directory
    run_dir = create_run_dir(figures_dir, "cifar10")
    print(f"Run directory: {run_dir}")
    
    # Hyperparameters
    INPUT_DIM = 3072  # 32x32x3
    HIDDEN_DIM = 2000
    NUM_CLASSES = 10
    
    # Bio params for CIFAR-10
    P = 2.0
    K = 2
    DELTA = 0.0
    N = 10.0  # ReLU^n exponent
    
    # Training params
    PHASE1_EPOCHS = 1000
    PHASE1_LR_START = 0.02
    PHASE1_LR_END = 0.0
    PHASE2_EPOCHS = 300
    BATCH_SIZE = 1000
    
    # Collect hyperparameters for logging
    hyperparams = {
        "experiment": "CIFAR-10",
        "device": device,
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": NUM_CLASSES,
        "p": P,
        "k": K,
        "delta": DELTA,
        "n": N,
        "phase1_epochs": PHASE1_EPOCHS,
        "phase1_lr_start": PHASE1_LR_START,
        "phase1_lr_end": PHASE1_LR_END,
        "phase2_epochs": PHASE2_EPOCHS,
        "batch_size": BATCH_SIZE,
        "record_evolution": record_evolution,
        "snapshot_interval": snapshot_interval if record_evolution else None,
    }
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)
    
    # =========================================================
    # Bio-Inspired Network
    # =========================================================
    print("\n--- Bio-Inspired Network ---")
    
    bio_net = BioNetwork(
        in_features=INPUT_DIM,
        hidden_features=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        p=P,
        k=K,
        delta=DELTA,
        n=N,
        # Parity with NumPy reference (old.py):
        dtype='float64',       # NumPy defaults to float64
        normalize_init=False,  # old.py uses randn/sqrt(in_dim) without row normalization
        device=device,
    )
    
    # Phase 1: Unsupervised learning
    print("\nPhase 1: Unsupervised training...")
    weight_snapshots = bio_net.train_unsupervised(
        train_loader,
        epochs=PHASE1_EPOCHS,
        lr_start=PHASE1_LR_START,
        lr_end=PHASE1_LR_END,
        device=device,
        shuffle=True,  # Reshuffle every epoch for parity with old.py
        record_evolution=record_evolution,
        snapshot_interval=snapshot_interval,
    )
    
    # Create weight evolution video if snapshots were recorded
    if record_evolution and weight_snapshots:
        print(f"\nCreating weight evolution video with {len(weight_snapshots)} frames...")
        
        # Save individual frames
        frames_dir = run_dir / "evolution_frames"
        save_weight_evolution_frames(
            weight_snapshots,
            img_shape=(3, 32, 32),
            save_dir=str(frames_dir),
            n_cols=20,
            n_rows=10,
            cmap="RdBu_r",
            color=True,
        )
        
        # Create video/gif
        video_path = str(run_dir / "weight_evolution.gif")
        create_weight_evolution_video(
            weight_snapshots,
            img_shape=(3, 32, 32),
            save_path=video_path,
            n_cols=20,
            n_rows=10,
            fps=3,
            cmap="RdBu_r",
            color=True,
        )
        print(f"Weight evolution video saved to {video_path}")
    
    # Visualize bio weights
    bio_weights = bio_net.bio_layer.weight.data.cpu().numpy()
    draw_weights_color(
        bio_weights,
        img_shape=(3, 32, 32),
        n_cols=20,
        n_rows=10,
        title="Bio-Inspired Weights (CIFAR-10)",
        save_path=str(run_dir / "cifar10_bio_weights.png"),
    )
    
    # Phase 2: Supervised learning
    print("\nPhase 2: Supervised training...")
    lr_schedule = [
        (0, 0.001),
        (100, 0.0005),
        (150, 0.00025),
        (200, 0.000125),
        (250, 0.0000625),
    ]
    
    bio_history = bio_net.train_supervised(
        train_loader,
        test_loader,
        epochs=PHASE2_EPOCHS,
        lr_schedule=lr_schedule,
        device=device,
    )
    
    bio_final_error = bio_history["test_error"][-1]
    print(f"\nBio-Inspired Final Test Error: {bio_final_error:.2f}%")
    
    # =========================================================
    # Backprop Baseline
    # =========================================================
    print("\n--- Backprop Baseline ---")
    
    bp_net = BackpropNetwork(
        in_features=INPUT_DIM,
        hidden_features=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
    )
    
    bp_history = bp_net.train_supervised(
        train_loader,
        test_loader,
        epochs=PHASE2_EPOCHS,
        lr=0.001,
        device=device,
    )
    
    bp_final_error = bp_history["test_error"][-1]
    print(f"\nBackprop Final Test Error: {bp_final_error:.2f}%")
    
    # Visualize backprop weights (color version for CIFAR-10)
    bp_weights = bp_net.hidden.weight.data.cpu().numpy()
    draw_weights_color(
        bp_weights,
        img_shape=(3, 32, 32),
        n_cols=20,
        n_rows=10,
        title="Backprop Weights (CIFAR-10) - Random Noise Expected",
        save_path=str(run_dir / "cifar10_bp_weights.png"),
    )
    
    # Plot learning curves
    plot_learning_curves(
        bio_history,
        bp_history,
        title="CIFAR-10: Accuracy vs Epoch",
        save_path=str(run_dir / "cifar10_learning_curves.png"),
    )
    
    # Save hyperparameters and results
    results = {
        "bio_final_error": f"{bio_final_error:.2f}%",
        "bp_final_error": f"{bp_final_error:.2f}%",
        "bio_min_error": f"{min(bio_history['test_error']):.2f}%",
        "bp_min_error": f"{min(bp_history['test_error']):.2f}%",
    }
    save_hyperparams(run_dir, hyperparams, results)
    
    # Summary
    print("\n" + "="*60)
    print("CIFAR-10 RESULTS SUMMARY")
    print("="*60)
    print(f"Bio-Inspired Error: {bio_final_error:.2f}%")
    print(f"Backprop Error:     {bp_final_error:.2f}%")
    print(f"\nFigures saved to: {run_dir}/")
    
    return bio_history, bp_history


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Krotov & Hopfield (2019) experiments"
    )
    parser.add_argument(
        "--mnist",
        action="store_true",
        help="Run MNIST experiment (Figure 3)",
    )
    parser.add_argument(
        "--cifar10",
        action="store_true",
        help="Run CIFAR-10 experiment (Figure 7)",
    )
    parser.add_argument(
        "--record-evolution",
        action="store_true",
        help="Record weight evolution video during unsupervised training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="figures",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=10,
        help="Epochs between weight snapshots (for --record-evolution)",
    )
    
    args = parser.parse_args()
    
    # Default to running both mnist and cifar10 if nothing specified
    if not args.mnist and not args.cifar10:
        args.mnist = True
        args.cifar10 = True
    
    # Create figures directory
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    
    # Check device availability
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Run experiments
    if args.mnist:
        run_mnist_experiment(
            device=device,
            figures_dir=args.figures_dir,
            record_evolution=args.record_evolution,
            snapshot_interval=args.snapshot_interval,
        )
    
    if args.cifar10:
        run_cifar10_experiment(
            device=device,
            figures_dir=args.figures_dir,
            record_evolution=args.record_evolution,
            snapshot_interval=args.snapshot_interval,
        )
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
