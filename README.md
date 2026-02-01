# Hopfield Learning

PyTorch implementation of biologically-inspired unsupervised learning from [Krotov & Hopfield (2019) "Unsupervised learning by competing hidden units"](https://www.pnas.org/doi/10.1073/pnas.1820458116).

## Overview

This project reproduces the key experiments from the paper, comparing bio-inspired learning with traditional backpropagation on MNIST and CIFAR-10 datasets.

### Key Features

- **Bio-Inspired Learning**: Implements the unsupervised Hebbian-like learning rule with anti-Hebbian competition
- **Backprop Baseline**: Standard end-to-end backpropagation for comparison
- **Weight Evolution Videos**: Optional recording of weight evolution during unsupervised training
- **Learning Curve Visualization**: Train and test accuracy plots with distinct colors

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Usage

### Running Experiments

```bash
# Run both MNIST and CIFAR-10 experiments
uv run python reproduce_results.py

# Run only MNIST experiment
uv run python reproduce_results.py --mnist

# Run only CIFAR-10 experiment
uv run python reproduce_results.py --cifar10

# Use GPU (CUDA or MPS on Apple Silicon)
uv run python reproduce_results.py --device cuda
uv run python reproduce_results.py --device mps
```

### Recording Weight Evolution

Capture the unsupervised learning process as a video:

```bash
# Record weight evolution during training
uv run python reproduce_results.py --mnist --record-evolution

# Custom snapshot interval (default: 10 epochs)
uv run python reproduce_results.py --cifar10 --record-evolution --snapshot-interval 20
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--mnist` | Run MNIST experiment (Figure 3 from paper) |
| `--cifar10` | Run CIFAR-10 experiment (Figure 7 from paper) |
| `--record-evolution` | Record weight evolution video during unsupervised training |
| `--device {cpu,cuda,mps}` | Device to use for training (default: cpu) |
| `--figures-dir DIR` | Directory to save figures (default: figures) |
| `--snapshot-interval N` | Epochs between weight snapshots (default: 10) |

## Project Structure

```
hopfield-learning/
├── reproduce_results.py   # Main experiment runner
├── bio_linear.py          # Bio-inspired network implementation
├── backprop.py            # Backprop baseline implementation
├── data_utils.py          # Dataset loading utilities
├── visualize.py           # Visualization functions
├── pyproject.toml         # Project dependencies
├── data/                  # Downloaded datasets (MNIST, CIFAR-10)
├── figures/               # Output figures and videos
└── slides/                # Manim presentation slides
```

## Output

Each experiment run creates a timestamped directory in `figures/` containing:

- `*_bio_weights.png` - Visualized bio-inspired weights (ghost digits/features)
- `*_bp_weights.png` - Visualized backprop weights (random noise)
- `*_learning_curves.png` - Train/test accuracy comparison
- `hyperparams.json` - Saved hyperparameters and results
- `weight_evolution.gif` - Weight evolution video (if `--record-evolution` used)
- `evolution_frames/` - Individual snapshot frames (if `--record-evolution` used)

### Learning Curve Colors

| Method | Train | Test |
|--------|-------|------|
| Backprop (BP) | Pink | Green |
| Bio-Inspired | Blue | Orange |

## Architecture

Both networks use a two-layer architecture:
- **Hidden layer**: 2000 neurons
- **Output layer**: 10 classes (softmax)

### Bio-Inspired Network

1. **Phase 1 (Unsupervised)**: Train hidden layer with competing hidden units
   - MNIST: p=3, k=7, δ=0.4, 1000 epochs
   - CIFAR-10: p=2, k=2, δ=0.0, 1000 epochs
   
2. **Phase 2 (Supervised)**: Train readout layer only
   - ReLU^n activation (n=4.5 for MNIST, n=10 for CIFAR-10)
   - Adam optimizer with learning rate schedule

### Backprop Network

- End-to-end supervised training with standard backpropagation
- ReLU activation, Adam optimizer

## Results

Expected test error rates:
- **MNIST**: ~1.5% for both methods
- **CIFAR-10**: Varies based on hyperparameters

## Dependencies

- Python ≥3.13
- PyTorch ≥2.0
- torchvision ≥0.15
- numpy ≥1.24
- matplotlib ≥3.7
- tqdm ≥4.65
- imageio ≥2.37 (for weight evolution videos)
- manim ≥0.19 (for presentation slides)

## References

Krotov, D., & Hopfield, J. J. (2019). Unsupervised learning by competing hidden units. *Proceedings of the National Academy of Sciences*, 116(16), 7723-7731.
