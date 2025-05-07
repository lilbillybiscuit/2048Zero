#!/usr/bin/env python3
"""
2048 Zero training script
"""
import os
import sys
import torch
import time
import argparse
from zero import ZeroNetwork, ZeroTrainer, GameRules


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train 2048-Zero model")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--games-per-epoch", type=int, default=1, help="Games per epoch")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    # Resume training
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--resume-from", type=str, help="Path to checkpoint file")
    
    # Model parameters
    parser.add_argument("--k-channels", type=int, default=20, help="Input channels")
    parser.add_argument("--filters", type=int, default=128, help="Conv filters")
    parser.add_argument("--blocks", type=int, default=10, help="Residual blocks")
    
    # WandB
    parser.add_argument("--wandb", action="store_true", help="Use WandB")
    parser.add_argument("--project", type=str, default="2048-zero", help="WandB project")
    parser.add_argument("--experiment", type=str, help="WandB experiment name")
    
    args = parser.parse_args()
    
    # TEMPORARILY DISABLED: Checkpoint resuming code
    # For now, we'll force these values to start fresh
    args.resume = False
    args.resume_from = None
    
    """
    # Handle R2 URLs for checkpoint resuming - DISABLED
    if args.resume and args.resume_from:
        # Only handle R2 URLs, no need for HTTP URLs
        if args.resume_from.startswith("r2://"):
            try:
                # Try to import checkpoint_utils
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                import checkpoint_utils
                print(f"Downloading checkpoint from R2: {args.resume_from}")
                local_path = checkpoint_utils.download_checkpoint_from_r2(args.resume_from)
                if local_path:
                    args.resume_from = local_path
                    print(f"Using downloaded checkpoint: {local_path}")
                else:
                    print("WARNING: Could not download checkpoint from R2, starting fresh")
                    args.resume = False
                    args.resume_from = None
            except ImportError:
                print("WARNING: checkpoint_utils module not available, cannot download from R2")
                print("Make sure checkpoint_utils.py is in the current directory and boto3 is installed")
                print("Run: pip install boto3")
                args.resume = False
                args.resume_from = None
        
        # If checkpoint doesn't exist locally, abort resuming
        if args.resume and not os.path.exists(args.resume_from):
            print(f"WARNING: Checkpoint file {args.resume_from} not found. Starting fresh.")
            args.resume = False
            args.resume_from = None
    """
    
    # Initialize game rules, model, and player
    rules = GameRules(num_spawn_tiles_per_move=1)
    model = ZeroNetwork(4, 4, args.k_channels, filters=args.filters, blocks=args.blocks)
    
    # Move model to best available device
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device for training")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device for training")
    else:
        print(f"Using CPU device for training")
    
    model.to(device)
    
    # Define a score-based reward function (now uses unbounded version)
    def score_reward(state, stats):
        """Reward based on score - use unbounded log scale"""
        import math
        score = stats['score']
        # Use raw log score without normalization
        z = math.log(score + 100)
        return z, "unbounded_score"
    
    # Create trainer with wandb enabled by default
    # Pass the score_reward function (score is now the default if no function is provided)
    trainer = ZeroTrainer(
        model=model, 
        rules=rules,
        use_wandb=args.wandb,  # Use from arguments
        project_name=args.project,
        experiment_name=args.experiment,
        reward_function=score_reward  # Using score reward
    )
    
    # Run training
    start_time = time.time()
    print(f"Starting training with {args.epochs} epochs, {args.games_per_epoch} games per epoch")
    
    trainer.train(
        epochs=args.epochs,
        games_per_epoch=args.games_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        simulations=args.simulations,
        resume=args.resume,
        resume_from=args.resume_from
    )
    
    elapsed_time = time.time() - start_time
    print(f"Training complete! Total time: {elapsed_time/60:.2f} minutes")
    print(f"Checkpoints saved in: ./checkpoints")

if __name__ == "__main__":
    main()