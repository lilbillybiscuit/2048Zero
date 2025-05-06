#!/usr/bin/env python
"""
Simplified parallel training script for AlphaZero 2048
"""

import os
import argparse
import torch

from zero.game import GameRules
from zero.zeromodel import ZeroNetwork
from distributed import ParallelZeroTrainer
from zero.reward_functions import score_reward_func, hybrid_reward_func

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train AlphaZero 2048 using parallel self-play")
    
    # Model parameters
    parser.add_argument("--filters", type=int, default=128, help="Number of filters in the model")
    parser.add_argument("--blocks", type=int, default=10, help="Number of residual blocks")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--games-per-epoch", type=int, default=32, help="Number of self-play games per epoch")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move")
    
    # Parallel settings
    parser.add_argument("--workers", type=int, default=None, 
                      help="Number of worker processes (default: CPU count)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Checkpointing
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoints every N epochs")
    
    # WandB settings
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--project-name", type=str, default="2048-zero", help="WandB project name")
    parser.add_argument("--experiment-name", type=str, default=None, help="WandB experiment name")
    
    # Visualization
    parser.add_argument("--enable-monitor", action="store_true", help="Enable web-based monitoring visualization")
    
    args = parser.parse_args()
    
    # Create game rules
    rules = GameRules()
    
    # Select device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = ZeroNetwork(rules.height, rules.width, 16, filters=args.filters, blocks=args.blocks)
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    else:
        print(f"Initializing new model with {args.filters} filters, {args.blocks} blocks")
        model = ZeroNetwork(rules.height, rules.width, 16, filters=args.filters, blocks=args.blocks)
    
    # Move model to device
    model = model.to(device)
    
    # Create trainer with simplified interface
    trainer = ParallelZeroTrainer(
        model=model,
        rules=rules,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        num_workers=args.workers,
        seed=args.seed,
        reward_function=hybrid_reward_func,  # Using the hybrid reward function by default
        enable_monitoring=args.enable_monitor
    )
    
    # Run training
    print(f"Starting parallel training with {args.workers or 'auto'} workers")
    print(f"Training for {args.epochs} epochs, {args.games_per_epoch} games per epoch")
    print(f"MCTS simulations per move: {args.simulations}")
    
    trainer.train(
        epochs=args.epochs,
        games_per_epoch=args.games_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        simulations=args.simulations
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()