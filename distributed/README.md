# Distributed AlphaZero 2048 Training

This module provides distributed training capabilities for AlphaZero 2048.

## Usage

```python
from zero.game import GameRules
from zeromodel import ZeroNetwork
from distributed import ParallelZeroTrainer

# Create model and rules
rules = GameRules()
model = ZeroNetwork(rules.height, rules.width, 16)

# Create parallel trainer
trainer = ParallelZeroTrainer(
   model=model,
   rules=rules,
   num_workers=8  # number of parallel workers
)

# Train using parallel self-play
trainer.train(
   epochs=100,
   games_per_epoch=32,
   batch_size=64,
   lr=0.001,
   simulations=50  # MCTS simulations per move
)
```

## CUDA and Multi-GPU Systems

The implementation optimizes GPU usage by distributing workers across available GPUs. By default, it calculates an optimal worker count based on your CPU and GPU count.

### Worker Distribution

When GPUs are available, workers are automatically distributed evenly across them. The system ensures optimal GPU utilization by adjusting worker counts if needed.

The system works as follows:

1. **Automatic mode** (default): 
   - Without any worker specification, uses a sensible default based on CPU count
   - Ensures at least 2 workers per GPU for better utilization
   - Provides balanced workload across all GPUs

2. **Manual specification with automatic adjustment**:
   - Specify total workers with the `--workers` parameter (e.g., `--workers 9`)
   - The system automatically adjusts to the nearest lower multiple of GPU count
   - For example: 9 workers with 2 GPUs becomes 8 workers (4 per GPU)
   - You'll see a message about the adjustment

```bash
# Run with a specific number of workers
python train_parallel.py --workers 8
```

This creates 8 workers in total, distributed across available GPUs.

### CUDA Troubleshooting

If you encounter CUDA freezing issues:

#### Option 1: Use CPU-only for Self-Play Workers

You can force all workers to use CPU by modifying `gpu_utils.py`:

```python
def assign_device(self, worker_id: int, num_workers: int) -> str:
    # Uncomment this line:
    return 'cpu'
```

This will still use the GPU for the training phase.

#### Option 2: Reduce Workers Per GPU

Try using fewer workers per GPU:

```bash
python train_parallel.py --workers-per-gpu 2
```

#### Option 3: Use CUDA_VISIBLE_DEVICES

To limit which GPUs are used:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_parallel.py
```

This will only use the first two GPUs.

## Command Line Options

```bash
# Run with default settings
python train_parallel.py

# Specify number of workers
python train_parallel.py --workers 16

# Configure model size
python train_parallel.py --filters 64 --blocks 5

# Training parameters
python train_parallel.py --epochs 500 --games-per-epoch 64 --batch-size 128 --lr 0.0005

# Load from checkpoint
python train_parallel.py --checkpoint checkpoints/model_epoch_50.pth
```

## Performance Considerations

1. **Worker Count**: Starting with CPU count is reasonable, but you might get better performance with fewer workers that have more resources each.

2. **Batch Size**: Larger batch sizes are more efficient for GPU training.

3. **Games Per Epoch**: This should scale with the number of workers (at least num_workers, ideally a multiple).

4. **MCTS Simulations**: Reducing simulations can speed up data collection at the cost of quality.