# Hopfield Learning: Bio-Inspired Unsupervised Learning

Implementation of **Krotov & Hopfield (2019)** - "Unsupervised learning by competing hidden units"

This project reproduces the key experiments from the paper, demonstrating that biologically-plausible local learning rules can match the performance of backpropagation on MNIST and CIFAR-10 classification tasks.

## 📋 Overview

The implementation includes:
- **Bio-inspired learning layer** with anti-Hebbian competition (Figure 3)
- **MNIST experiments** showing "ghost digit" receptive fields
- **CIFAR-10 experiments** with color image features
- **Backpropagation baseline** for comparison
- **Visualization tools** for weights and learning curves

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone and setup**:
```bash
cd hopfield-learning
uv sync
```

This will automatically:
- Create a virtual environment
- Install all dependencies from `pyproject.toml`
- Lock versions in `uv.lock`

## 🔬 Reproducing Results

### Run All Experiments

To reproduce both MNIST and CIFAR-10 experiments:

```bash
uv run python reproduce_results.py
```

### Run Specific Experiments

**MNIST only** (Figure 3):
```bash
uv run python reproduce_results.py --mnist
```

**CIFAR-10 only** (Figure 7):
```bash
uv run python reproduce_results.py --cifar10
```

### GPU Acceleration

Use GPU if available (CUDA or Apple Silicon):

```bash
# NVIDIA GPU
uv run python reproduce_results.py --device cuda

# Apple Silicon (M1/M2/M3)
uv run python reproduce_results.py --device mps
```

### Custom Output Directory

```bash
uv run python reproduce_results.py --figures-dir my_results
```

## 📊 Understanding the Results

### Output Structure

Each experiment run creates a timestamped directory in `figures/`:

```
figures/
├── mnist_2026-01-31_21-30-00/
│   ├── mnist_bio_weights.png          # Bio-inspired receptive fields
│   ├── mnist_bp_weights.png           # Backprop weights (noise)
│   ├── mnist_learning_curves.png      # Test error vs epoch
│   ├── hyperparams.log                # Human-readable config
│   └── hyperparams.json               # Machine-readable config
└── cifar10_2026-01-31_22-15-00/
    ├── cifar10_bio_weights.png
    ├── hyperparams.log
    └── hyperparams.json
```

### Expected Results

**MNIST (Figure 3)**:
- **Bio-inspired error**: ~1.46%
- **Backprop error**: ~1.50%
- **Bio weights**: Should show clear "ghost digits" with negative halos
- **Backprop weights**: Should look like random noise

**CIFAR-10 (Figure 7)**:
- **Bio-inspired error**: ~45-50%
- **Bio weights**: Should show color edge detectors and Gabor-like filters

### Hyperparameters

All hyperparameters are logged in each run directory. Key parameters:

**MNIST**:
- Hidden units: 2000
- p=3, k=7, δ=0.4
- ReLU^4.5 activation
- Phase 1: 1000 epochs unsupervised (LR: 0.04→0)
- Phase 2: 300 epochs supervised (LR schedule)

**CIFAR-10**:
- Hidden units: 2000
- p=2, k=2, δ=0.0
- ReLU^10 activation
- Same training schedule as MNIST

## 🎨 Visualization Tools

### Standalone Visualization

You can use `visualize.py` to create custom visualizations:

```python
from visualize import draw_weights, draw_weights_color, plot_learning_curves
import numpy as np

# Visualize grayscale weights (MNIST)
weights = np.random.randn(200, 784)  # Your weight matrix
draw_weights(
    weights,
    img_shape=(28, 28),
    n_cols=20,
    n_rows=10,
    title="My Weights",
    save_path="my_weights.png"
)

# Visualize color weights (CIFAR-10)
weights_color = np.random.randn(200, 3072)
draw_weights_color(
    weights_color,
    img_shape=(3, 32, 32),
    n_cols=20,
    n_rows=10,
    save_path="my_color_weights.png"
)

# Plot learning curves
bio_history = {"test_error": [5.0, 3.0, 2.0, 1.5]}
bp_history = {"test_error": [5.5, 3.5, 2.5, 1.8]}
plot_learning_curves(
    bio_history,
    bp_history,
    title="My Learning Curves",
    save_path="my_curves.png"
)
```

## 🛠️ Using uv

### Common uv Commands

```bash
# Install/sync dependencies
uv sync

# Add a new dependency
uv add scipy

# Add a development dependency
uv add --dev pytest

# Run a Python script
uv run python reproduce_results.py

# Run a Python command
uv run python -c "import torch; print(torch.__version__)"

# Activate the virtual environment manually
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate  # Windows

# Update dependencies
uv lock --upgrade
```

### Why uv?

- **Fast**: 10-100x faster than pip
- **Reliable**: Deterministic dependency resolution
- **Simple**: No need to manually manage virtual environments
- **Compatible**: Works with standard `pyproject.toml`

## 📁 Project Structure

```
hopfield-learning/
├── README.md                    # This file
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Locked dependency versions
├── reproduce_results.py        # Main experiment script
├── bio_linear.py               # Bio-inspired learning layer
├── backprop.py                 # Backprop baseline network
├── data_utils.py               # Dataset loaders
├── visualize.py                # Visualization utilities
├── paper.pdf                   # Original paper
├── data/                       # Downloaded datasets (auto-created)
└── figures/                    # Experiment results (auto-created)
```

## 🔍 Key Implementation Details

### Bio-Inspired Learning Rule

The core learning rule implements:

```
Δw_i = η * x * (h_i^p - δ * Σ_j h_j^k)
```

where:
- `h_i = max(0, w_i · x)` - ReLU activation
- `p` - Hebbian exponent (typically 2-3)
- `k` - anti-Hebbian exponent (typically 2-7)
- `δ` - anti-Hebbian strength (0.0-0.4)

This creates **competition** between hidden units, leading to sparse, interpretable features.

### Two-Phase Training

1. **Phase 1 (Unsupervised)**: Train bio-layer with local learning rule
2. **Phase 2 (Supervised)**: Freeze bio-layer, train readout with backprop

This mimics biological learning where early sensory layers develop without supervision.

## 📚 References

**Original Paper**:
```
Krotov, D., & Hopfield, J. J. (2019).
Unsupervised learning by competing hidden units.
Proceedings of the National Academy of Sciences, 116(16), 7723-7731.
```

**Related Work**:
- Hopfield Networks (1982)
- Oja's Rule (1982)
- Sparse Coding (Olshausen & Field, 1996)

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size or use CPU
uv run python reproduce_results.py --device cpu
```

### MPS (Apple Silicon) Issues
```bash
# Fall back to CPU if MPS has issues
uv run python reproduce_results.py --device cpu
```

### Slow Training
- Use GPU acceleration (`--device cuda` or `--device mps`)
- Reduce number of epochs for testing
- MNIST takes ~30-60 minutes on CPU, ~5-10 minutes on GPU
- CIFAR-10 takes ~60-120 minutes on CPU, ~10-20 minutes on GPU

## 📝 License

This is an educational implementation for academic purposes.

## 🤝 Contributing

This is a course project. For questions or issues, please contact the author.
