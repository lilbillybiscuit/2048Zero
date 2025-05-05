# 2048 AlphaZero Training Guide

This document explains how to train the 2048 AlphaZero model using the optimized parallel training script.

## Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA toolkit (for GPU acceleration)

## Training

Use the optimized parallel training script to train the model:

```bash
# Basic training with defaults
python train_parallel.py

# With specific configurations
python train_parallel.py --workers 8 --epochs 200 --games-per-epoch 64 --batch-size 128 --lr 0.0005 --simulations 200 --use-wandb

# Force a specific accelerator
python train_parallel.py --accelerator cuda  # Force CUDA even if multiple options are available
python train_parallel.py --accelerator mps   # Force MPS (Apple Silicon) acceleration
python train_parallel.py --accelerator cpu   # Force CPU processing
```

### Command-line Arguments

- `--workers`: Number of worker processes (default: CPU count)
- `--epochs`: Number of training epochs (default: 100)
- `--games-per-epoch`: Self-play games per epoch (default: 32)
- `--batch-size`: Training batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--simulations`: MCTS simulations per move (default: 50)
- `--checkpoint`: Path to load model from (optional)
- `--checkpoint-interval`: Save checkpoints every N epochs (default: 10)
- `--accelerator`: Force specific accelerator, one of: "cuda", "mps", or "cpu"
- `--use-wandb`: Enable Weights & Biases logging
- `--project-name`: Project name for wandb (default: "2048-zero-parallel")
- `--experiment-name`: Experiment name for wandb (auto-generated if None)

### Custom Reward Functions

The training script uses a default score-based reward function that normalizes the game score to the range [-1, 1]. 

Other built-in reward functions:
- `default_score_reward`: Uses the final score of the game
- `max_tile_reward`: Uses the maximum tile achieved
- `hybrid_reward`: Combines score and max tile rewards with customizable weights

## Checkpoints

Checkpoints are saved to the `checkpoints/` directory by default. The naming format is:
`checkpoints/{experiment_name}_epoch_{epoch}.pth`

## Resuming Training

To resume training from a checkpoint:

```bash
python train_parallel.py --checkpoint checkpoints/your_checkpoint.pth
```

## Weights & Biases Integration

To track and visualize training metrics, add the `--use-wandb` flag:

```bash
python train_parallel.py --use-wandb --project-name "my-2048-project"
```

This will log metrics including:
- Training loss (policy and value losses)
- Self-play statistics (scores, max tiles, game lengths)
- Board examples
- Learning rate schedule

## Accelerator Options

The training script supports different acceleration options:

- **CUDA**: For NVIDIA GPUs
  - Automatically distributes workers across multiple GPUs
  - Uses staggered CUDA initialization to prevent deadlocks
  - Implements timeout protection for CUDA operations

- **MPS**: For Apple Silicon (M1/M2/M3) devices
  - Uses Metal Performance Shaders for acceleration

- **CPU**: For systems without GPUs
  - Runs on CPU cores with optimized threading

You can force a specific accelerator using the `--accelerator` flag, which is useful in environments with multiple available options or for debugging.

## Error Handling

- If any worker process encounters an error during training, the entire training process will be terminated to prevent partial or invalid results.
- The training will display detailed error information to help diagnose the issue.

For optimal performance on multi-GPU systems, set the number of workers to a multiple of your GPU count.