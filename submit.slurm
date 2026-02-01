#!/bin/bash

#SBATCH --job-name="hopfield_learning"
#SBATCH --account=3176608
#SBATCH --partition=ai

#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=12:00:00 # 12 hours for full experiments with weight recording

#SBATCH --chdir=.
#SBATCH --output=out/%x_%j.out # %x gives job name and %j gives job id
#SBATCH --error=err/%x_%j.err

#SBATCH --mail-user=<username>@studbocconi.it
#SBATCH --mail-type=ALL

# Load necessary modules (adjust based on your HPC environment)
# module load python/3.11
module load cuda

# Activate virtual environment
source .venv/bin/activate

# Create output directories if they don't exist
mkdir -p out err figures

# Run the script
echo "Starting job on $HOSTNAME"
echo "Date: $(date)"
echo "Running MNIST experiment with weight evolution recording (time: $(date))"

# If using uv on the cluster:
# uv run python reproduce_results.py --device cuda --record-evolution --snapshot-interval 10

# If using standard python/venv:
python reproduce_results.py --device cuda --mnist --record-evolution --snapshot-interval 10

echo "Running CIFAR-10 experiment with weight evolution recording (time: $(date))"

python reproduce_results.py --device cuda --cifar10 --record-evolution --snapshot-interval 10

echo "Job finished at $(date)"