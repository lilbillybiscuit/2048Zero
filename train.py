#!/usr/bin/env python3
"""
2048 Zero training script
"""
import os
import torch
import time
from zero import ZeroNetwork, ZeroTrainer, GameRules


def main():
    # Training parameters
    epochs = 50
    games_per_epoch = 1
    batch_size = 64
    simulations = 50
    lr = 0.001
    checkpoint_path = None  # Change this to load from a checkpoint
    project_name = "2048-zero"
    experiment_name = None  # Auto-generated if None
    
    # Initialize game rules, model, and player
    rules = GameRules(num_spawn_tiles_per_move=1)
    model = ZeroNetwork(4, 4, 20, filters=128, blocks=10)
    
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
    
    # Load checkpoint if specified
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load(checkpoint_path)
        print(f"Loaded model from {checkpoint_path}")
    
    # Define a score-based reward function
    def score_reward(state, stats):
        """Reward based on score - normalize to [-1, 1] using log scale"""
        import math
        score = stats['score']
        # Normalize score to [-1, 1] range using log scale
        z = min(max((math.log(score + 100) / math.log(50000 + 100)) * 2 - 1, -1.0), 1.0)
        return z, "score"
    
    # Create trainer with wandb enabled by default
    # Pass the score_reward function (score is now the default if no function is provided)
    trainer = ZeroTrainer(
        model=model, 
        rules=rules,
        use_wandb=True,  # Always enable wandb
        project_name=project_name,
        experiment_name=experiment_name,
        reward_function=score_reward  # Using score reward
    )
    
    # Run training
    start_time = time.time()
    print(f"Starting training with {epochs} epochs, {games_per_epoch} games per epoch")
    
    trainer.train(
        epochs=epochs,
        games_per_epoch=games_per_epoch,
        batch_size=batch_size,
        lr=lr,
        simulations=simulations
    )
    
    elapsed_time = time.time() - start_time
    print(f"Training complete! Total time: {elapsed_time/60:.2f} minutes")
    print(f"Checkpoints saved in: ./checkpoints")

if __name__ == "__main__":
    main()