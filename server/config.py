import argparse
from typing import Dict, Any

def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="2048-Zero Trainer Server")

    # Basic server settings
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--auth-token", type=str, default="2048-zero-token", help="Authentication token")
    parser.add_argument("--min-games", type=int, default=512, help="Minimum games before training")
    parser.add_argument("--initial-deadline", type=int, default=30, help="Initial deadline in minutes")
    parser.add_argument("--training-deadline", type=int, default=30, help="Training deadline in minutes")
    parser.add_argument("--check-interval", type=int, default=10, help="Check interval in seconds")

    # Storage settings
    parser.add_argument("--localhost", action="store_true", help="Serve weights via HTTP endpoint")
    parser.add_argument("--use-r2", action="store_true", help="Use Cloudflare R2 for storing weights")
    parser.add_argument("--r2-bucket", type=str, default="2048-zero", help="R2 bucket name")
    parser.add_argument("--r2-account-id", type=str, default="", help="R2 account ID")
    parser.add_argument("--r2-access-key-id", type=str, default="", help="R2 access key ID")
    parser.add_argument("--r2-secret-access-key", type=str, default="", help="R2 secret access key")
    parser.add_argument("--r2-public-url", type=str, default="", help="R2 public URL (optional)")
    parser.add_argument("--keep-revisions", type=int, default=5, help="Number of model revisions to keep")

    # Training settings
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use for training (cuda, cpu, mps, or None for auto)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per training cycle")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--scheduler-step-size", type=int, default=20, 
                        help="Step size for learning rate scheduler")
    parser.add_argument("--scheduler-gamma", type=float, default=0.5, 
                        help="Gamma for learning rate scheduler")
    
    # Lightning specific settings
    parser.add_argument("--num-workers", type=int, default=4, 
                        help="Number of worker processes for data loading (Lightning only)")
    parser.add_argument("--accelerator", type=str, default="auto", 
                        help="PyTorch Lightning accelerator (Lightning only)")
    parser.add_argument("--val-split", type=float, default=0.1, 
                        help="Fraction of data to use for validation (Lightning only)")

    # Model settings
    parser.add_argument("--filters", type=int, default=128, help="Number of filters in model")
    parser.add_argument("--blocks", type=int, default=10, help="Number of residual blocks in model")
    parser.add_argument("--k-channels", type=int, default=16, 
                        help="Number of channels for input representation")

    # Self-play settings
    parser.add_argument("--simulations", type=int, default=800, 
                        help="Number of MCTS simulations per move")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet noise alpha")
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25, 
                        help="Dirichlet noise epsilon")
    parser.add_argument("--max-moves", type=int, default=1000, help="Maximum moves per game")
    parser.add_argument("--temperature", type=float, default=1.0, help="MCTS temperature")

    # Other settings
    parser.add_argument("--reset", action="store_true", help="Reset all state")
    
    # Resume settings
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--resume-from", type=str, help="Path or URL to checkpoint file")
    parser.add_argument("--resume-revision", type=int, help="Revision number to resume from (default: auto-detect from filename)")

    args = parser.parse_args()

    # Create configuration
    config = {
        # Server settings
        "host": args.host,
        "port": args.port,
        "auth_token": args.auth_token,
        "min_games": args.min_games,
        "initial_deadline_minutes": args.initial_deadline,
        "training_deadline_minutes": args.training_deadline,
        "check_interval_seconds": args.check_interval,

        # Storage settings
        "localhost_weights": args.localhost,
        "weights_dir": "weights",
        "use_r2": args.use_r2,
        "r2_bucket": args.r2_bucket,
        "r2_account_id": args.r2_account_id,
        "r2_access_key_id": args.r2_access_key_id,
        "r2_secret_access_key": args.r2_secret_access_key,
        "r2_public_url": args.r2_public_url or None,
        "keep_revisions": args.keep_revisions,
        
        # Training settings
        "device": args.device,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        
        # Lightning specific settings
        "num_workers": args.num_workers,
        "accelerator": args.accelerator,
        "val_split": args.val_split,

        # Model config
        "model": {
            "filters": args.filters,
            "blocks": args.blocks,
            "k_channels": args.k_channels,
        },

        # Self-play config
        "self_play": {
            "num_simulations": args.simulations,
            "dirichlet_alpha": args.dirichlet_alpha,
            "epsilon": args.dirichlet_epsilon,
            "max_moves_per_game": args.max_moves,
            "temperature": args.temperature,
        },

        # Heartbeat settings
        "heartbeat": 5,

        # State settings
        "reset": args.reset,
        
        # Resume settings
        "resume": args.resume,
        "resume_from": args.resume_from,
        "resume_revision": args.resume_revision
    }

    return config