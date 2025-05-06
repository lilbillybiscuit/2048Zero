#!/usr/bin/env python
"""
Parallel training script for AlphaZero 2048
"""

import os
import argparse
import torch

# Original implementation is now thread-safe

from zero.game import GameRules
from zero.zeromodel import ZeroNetwork
from distributed import ParallelZeroTrainer, DeviceManager
from zero.reward_functions import score_reward_func, hybrid_reward_func, dynamic_score_reward_func, get_reward_func

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
    
    # MCTS exploration parameters
    parser.add_argument("--dirichlet-eps", type=float, default=0.25, 
                        help="Weight of Dirichlet noise added to root node (default: 0.25)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5, 
                        help="Concentration parameter for Dirichlet noise (default: 0.5)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for action selection (default: 1.0)")
    
    # Device selection
    parser.add_argument("--accelerator", type=str, choices=["cuda", "mps", "cpu"], 
                        help="Force a specific accelerator ('cuda', 'mps', or 'cpu')")
    
    # Parallel settings
    parser.add_argument("--workers", type=int, default=None, 
                      help="Number of worker processes (default: auto-calculated based on CPU/GPU count)")
    parser.add_argument("--workers-per-gpu", type=int, default=3,
                      help="Maximum number of worker processes per GPU (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Checkpointing
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoints every N epochs")
    
    # WandB settings
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--project-name", type=str, default="2048-zero-parallel", help="WandB project name")
    parser.add_argument("--experiment-name", type=str, default=None, help="WandB experiment name")
    
    # Optimization flags
    parser.add_argument("--use-compile", action="store_true", help="Use torch.compile for model optimization")
    parser.add_argument("--dynamic-reward", action="store_true", help="Use dynamic reward scaling that adapts to highest observed score")
    parser.add_argument("--reward-type", type=str, default="hybrid", choices=["score", "max_tile", "hybrid", "dynamic_score"], 
                        help="Type of reward function to use")
    
    args = parser.parse_args()
    
    # Create game rules
    rules = GameRules()
    
    # Initialize device manager
    device_manager = DeviceManager()
    
    # Handle forced accelerator if specified
    if args.accelerator:
        if args.accelerator == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA accelerator requested but not available on this system")
        elif args.accelerator == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS accelerator requested but not available on this system")
            
        best_device = args.accelerator
        print(f"Using forced accelerator: {best_device}")
        
        # Set appropriate environment variable
        if args.accelerator == "cuda":
            os.environ["FORCE_CUDA"] = "1"
        elif args.accelerator == "mps":
            os.environ["FORCE_MPS"] = "1"
        elif args.accelerator == "cpu":
            os.environ["FORCE_CPU"] = "1"
    else:
        best_device = device_manager.get_best_device()
        print(f"Using auto-selected device: {best_device}")
    
    # Initialize model
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = ZeroNetwork(rules.height, rules.width, 16, filters=args.filters, blocks=args.blocks)
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False))
    else:
        print(f"Initializing new model with {args.filters} filters, {args.blocks} blocks")
        model = ZeroNetwork(rules.height, rules.width, 16, filters=args.filters, blocks=args.blocks)
    
    # Move model to best device
    model = model.to(best_device)
    
    # Select reward function based on command line args
    if args.dynamic_reward and args.reward_type != "dynamic_score":
        print(f"Using dynamic reward scaling with {args.reward_type} reward type")
        # If dynamic reward is requested but reward type isn't dynamic_score, override it
        reward_function = dynamic_score_reward_func
    else:
        # Otherwise use the selected reward type
        reward_function = get_reward_func(args.reward_type)
        print(f"Using reward function: {args.reward_type}")
    
    # Create trainer with specified number of workers and exploration parameters
    trainer = ParallelZeroTrainer(
        model=model,
        rules=rules,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        num_workers=args.workers,
        workers_per_gpu=args.workers_per_gpu,
        seed=args.seed,
        reward_function=reward_function,
        dirichlet_eps=args.dirichlet_eps,
        dirichlet_alpha=args.dirichlet_alpha,
        temperature=args.temperature,
        use_compile=args.use_compile
    )
    
    # Run training
    print(f"Starting parallel training with {args.workers or 'auto'} workers")
    print(f"Training for {args.epochs} epochs, {args.games_per_epoch} games per epoch")
    print(f"MCTS simulations per move: {args.simulations}")
    print(f"Exploration parameters: noise={args.dirichlet_eps}, alpha={args.dirichlet_alpha}, temp={args.temperature}")
    print(f"Optimizations: dynamic_reward={args.dynamic_reward}, torch.compile={args.use_compile}")
    
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