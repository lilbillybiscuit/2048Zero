"""
Parallel trainer implementation for distributed AlphaZero 2048
"""

import os
import time
import random
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from game_alt import GameRules
from zeromodel import ZeroNetwork
from zeromonitoring import ZeroMonitor, EpochStats, print_epoch_summary
from .parallel_selfplay import MultiprocessSelfPlayWorker
from .gpu_utils import DeviceManager, seed_everything
from .cuda_utils import cuda_operation_timeout, TimeoutException, safe_cuda_initialization


class ParallelZeroTrainer:
    """Enhanced ZeroTrainer that uses parallel self-play for improved performance"""
    
    def __init__(
        self,
        model: ZeroNetwork,
        rules: GameRules,
        use_wandb: bool = True,
        project_name: str = "2048-zero-parallel",
        experiment_name: Optional[str] = None,
        num_workers: int = None,  # Default to CPU count
        seed: int = 42
    ):
        """
        Initialize the trainer with model, rules and parallel settings
        
        Args:
            model: Neural network model
            rules: Game rules
            use_wandb: Whether to use wandb for tracking
            project_name: Project name for wandb
            experiment_name: Experiment name for wandb
            num_workers: Number of worker processes for parallel self-play
            seed: Random seed for initialization
        """
        self.model = model
        self.rules = rules
        self.seed = seed
        
        # Set the random seed for reproducibility
        seed_everything(seed)
        
        # Set up device manager first to get GPU count
        self.device_manager = DeviceManager()
        
        # Set number of workers
        if num_workers is not None:
            # User specified a worker count
            original_workers = num_workers
            
            # If GPUs are available, automatically round to ensure even distribution
            if self.device_manager.num_gpus > 0:
                # Calculate workers per GPU and round down to ensure even distribution
                workers_per_gpu = num_workers // self.device_manager.num_gpus
                # If workers_per_gpu is at least 1, adjust total count
                if workers_per_gpu >= 1:
                    self.num_workers = workers_per_gpu * self.device_manager.num_gpus
                else:
                    # If fewer workers than GPUs, use at least one worker per GPU
                    self.num_workers = self.device_manager.num_gpus
                
                # Notify if adjustment was made
                if self.num_workers != original_workers:
                    print(f"Adjusted worker count from {original_workers} to {self.num_workers} "
                          f"for even distribution across {self.device_manager.num_gpus} GPUs.")
            else:
                # No GPUs, use specified count
                self.num_workers = num_workers
        else:
            # Default to CPU count
            cpu_count = os.cpu_count() or 8
            
            # If GPUs are available, ensure worker count is divisible by GPU count
            if self.device_manager.num_gpus > 0:
                # Calculate workers per GPU and ensure at least 2 per GPU if possible
                workers_per_gpu = max(2, cpu_count // self.device_manager.num_gpus)
                self.num_workers = workers_per_gpu * self.device_manager.num_gpus
            else:
                self.num_workers = cpu_count
        
        # Print worker distribution
        if self.device_manager.num_gpus > 0:
            workers_per_gpu = self.num_workers / self.device_manager.num_gpus
            print(f"Using {self.num_workers} worker processes ({workers_per_gpu:.1f} per GPU)")
        else:
            print(f"Using {self.num_workers} worker processes (CPU only)")
        
        # Generate experiment name if none provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            board_size = f"{rules.height}x{rules.width}"
            model_size = f"f{model.conv1.out_channels}_b{len(model.blocks)}"
            experiment_name = f"zero_parallel_{board_size}_{model_size}_{timestamp}"
            
        # Initialize monitoring
        config = {
            "model_filters": model.conv1.out_channels,
            "model_blocks": len(model.blocks),
            "model_k": model.k,
            "board_height": rules.height,
            "board_width": rules.width,
            "num_workers": self.num_workers,
            "seed": seed
        }
        
        self.monitor = ZeroMonitor(
            use_wandb=use_wandb,
            project_name=project_name,
            experiment_name=experiment_name,
            config=config
        )
        
        # Create directories
        self.temp_dir = "temp_data"
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        
        # Model path for sharing with workers
        self.model_path = f"{self.temp_dir}/current_model.pth"
        self.save_model()
    
    def save_model(self):
        """Save the current model to the shared model path"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
    
    def _collect_selfplay_data_parallel(
        self,
        games_per_epoch: int,
        simulations: int = 50
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Collect self-play data using parallel workers
        
        Args:
            games_per_epoch: Total number of self-play games to generate
            simulations: Number of MCTS simulations per move
            
        Returns:
            Tuple of (samples, statistics)
        """
        # Save current model for workers to use
        self.model.eval()  # Set to evaluation mode
        self.save_model()
        
        # Calculate games per worker (distribute evenly)
        games_per_worker = max(1, games_per_epoch // self.num_workers)
        actual_games = games_per_worker * self.num_workers
        
        if actual_games != games_per_epoch:
            print(f"Note: Adjusted games from {games_per_epoch} to {actual_games} for even distribution")
        
        # Create and run parallel self-play worker
        worker = MultiprocessSelfPlayWorker(
            model_path=self.model_path,
            num_workers=self.num_workers,
            games_per_worker=games_per_worker,
            simulations_per_move=simulations,
            temp_dir=self.temp_dir,
            seed=self.seed
        )
        
        # Generate games in parallel
        samples, stats = worker.generate_games()
        
        return samples, stats
    
    def _update_network(
        self, 
        samples: List[Tuple], 
        optimizer: torch.optim.Optimizer, 
        batch_size: int, 
        epoch: int
    ) -> Tuple[float, float, float]:
        """
        Update neural network weights using collected samples in batches
        
        Args:
            samples: List of (board, policy, value) tuples
            optimizer: PyTorch optimizer
            batch_size: Training batch size
            epoch: Current epoch number (for logging)
            
        Returns:
            Tuple of (average_loss, average_policy_loss, average_value_loss)
        """
        self.model.train()
        
        # Move model to best available device
        device = self.device_manager.get_best_device()
        
        # Use timeout protection for CUDA devices
        if device.startswith('cuda'):
            with cuda_operation_timeout(seconds=30):
                self.model.to(device)
        else:
            self.model.to(device)

        random.shuffle(samples)

        num_samples = len(samples)
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        # Create progress bar for batch processing
        batch_pbar = tqdm(range(num_batches), desc='Training batches')
        
        # Process samples in batches
        for i in batch_pbar:
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch = samples[start_idx:end_idx]
            
            # Unpack batch
            boards, pis, zs = zip(*batch)
            
            # Stack boards and convert to tensors
            boards_array = np.stack(boards, axis=0)  # Shape: (batch_size, height, width)

            state_tensor = self.model.to_onehot(boards_array).to(device)
            pi_tensor = torch.tensor(pis, dtype=torch.float32).to(device)
            z_tensor = torch.tensor(zs, dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()

            p_pred, v_pred = self.model(state_tensor)

            policy_loss = -(pi_tensor * torch.log(p_pred + 1e-8)).sum(dim=1).mean()
            value_loss = torch.nn.functional.mse_loss(v_pred, z_tensor)
            loss = policy_loss + value_loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate losses
            batch_size_actual = end_idx - start_idx
            total_loss += loss.item() * batch_size_actual
            total_policy_loss += policy_loss.item() * batch_size_actual
            total_value_loss += value_loss.item() * batch_size_actual
            
            # Update progress bar
            current_loss = loss.item()
            current_policy_loss = policy_loss.item()
            current_value_loss = value_loss.item()
            batch_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'π_loss': f'{current_policy_loss:.4f}',
                'v_loss': f'{current_value_loss:.4f}'
            })
            
            # Log batch metrics using the monitor
            self.monitor.log_batch_stats(
                batch_loss=current_loss,
                batch_policy_loss=current_policy_loss,
                batch_value_loss=current_value_loss,
                batch_size=batch_size_actual,
                epoch=epoch
            )
        
        # Calculate average losses
        avg_loss = total_loss / num_samples
        avg_policy_loss = total_policy_loss / num_samples
        avg_value_loss = total_value_loss / num_samples
        
        return avg_loss, avg_policy_loss, avg_value_loss
    
    def train(
        self,
        epochs: int = 100,
        games_per_epoch: int = 32,
        batch_size: int = 64,
        lr: float = 0.001,
        log_interval: int = 1,
        checkpoint_interval: int = 10,
        resume: bool = False,
        **kwargs
    ):
        """
        Train the model using parallel self-play
        
        Args:
            epochs: Number of training epochs
            games_per_epoch: Number of self-play games to generate per epoch
            batch_size: Batch size for network updates
            lr: Learning rate for optimizer
            log_interval: How often to log detailed metrics
            checkpoint_interval: How often to save model checkpoints
            resume: Whether to resume training from previous state
            **kwargs: Additional arguments
        """
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
            
        # Update monitor config with training parameters
        config_update = {
            "epochs": epochs,
            "games_per_epoch": games_per_epoch,
            "batch_size": batch_size,
            "learning_rate": lr,
            "scheduler_step_size": kwargs.get('scheduler_step_size', 20),
            "scheduler_gamma": kwargs.get('scheduler_gamma', 0.5),
            "simulations": kwargs.get('simulations', 50),
            "weight_decay": kwargs.get('weight_decay', 1e-4),
            "num_workers": self.num_workers
        }
        self.monitor.config.update(config_update)
        
        # Initialize wandb with the model
        if self.monitor.use_wandb and not self.monitor.wandb_initialized:
            self.monitor.init_wandb(watch_model=self.model)
        
        # Initialize optimizer with specified learning rate
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=config_update["weight_decay"]
        )
        
        # Add learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config_update["scheduler_step_size"],
            gamma=config_update["scheduler_gamma"]
        )
        
        # Create a run tracker file to allow resume
        self.monitor.run_data.update({
            "model_config": {
                "filters": self.model.conv1.out_channels,
                "blocks": len(self.model.blocks),
                "k": self.model.k
            },
            "board_config": {
                "height": self.rules.height,
                "width": self.rules.width
            },
            "training_config": {
                "epochs": epochs,
                "games_per_epoch": games_per_epoch,
                "batch_size": batch_size,
                "lr": lr,
                "num_workers": self.num_workers,
                "seed": self.seed
            }
        })
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 1. Collect training data with parallel self-play
            print(f"Epoch {epoch+1}/{epochs}")
            self.model.eval()  # Set to evaluation mode for self-play
            
            samples, epoch_stats = self._collect_selfplay_data_parallel(
                games_per_epoch=games_per_epoch,
                simulations=kwargs.get('simulations', 50)
            )
            
            # Log self-play statistics to wandb
            self.monitor.log_selfplay_stats(epoch_stats, epoch)
            
            # 2. Update neural network if samples were collected
            loss, pi_loss, v_loss = None, None, None
            if samples:
                # Update network with batched training
                loss, pi_loss, v_loss = self._update_network(
                    samples, optimizer, batch_size, epoch=epoch
                )
                
                # Step the learning rate scheduler
                scheduler.step()
                
                # Log training metrics
                self.monitor.log_training_stats(
                    loss=loss,
                    policy_loss=pi_loss,
                    value_loss=v_loss,
                    learning_rate=scheduler.get_last_lr()[0],
                    samples=len(samples),
                    epoch=epoch
                )
                
                # Save the updated model for next epoch
                self.save_model()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Report results
            temp_stats = EpochStats(self.rules.height, self.rules.width)
            print_epoch_summary(
                temp_stats,  # Temporary stats object for formatting
                epoch, epochs, loss, pi_loss, v_loss,
                scheduler.get_last_lr()[0] if scheduler else None
            )
            
            # Prepare epoch data for run tracker
            epoch_data = {
                "epoch": epoch,
                "samples": len(samples),
                "max_tile": epoch_stats.get('max_tile', 0),
                "max_score": epoch_stats.get('max_score', 0),
                "avg_tile": epoch_stats.get('total_tile', 0) / max(1, epoch_stats.get('games', 1)),
                "avg_score": epoch_stats.get('total_score', 0) / max(1, epoch_stats.get('games', 1)),
                "avg_turns": epoch_stats.get('total_turns', 0) / max(1, epoch_stats.get('games', 1)),
                "loss": loss if samples else None,
                "policy_loss": pi_loss if samples else None,
                "value_loss": v_loss if samples else None,
                "learning_rate": scheduler.get_last_lr()[0] if scheduler else None,
                "epoch_time_seconds": epoch_time
            }
            
            # Update run tracker
            self.monitor.update_run_data(epoch_data)
            
            # Save run tracker
            run_path = f"checkpoints/{self.monitor.experiment_name}_run.json"
            self.monitor.save_run_data(run_path)
                
            # 3. Save checkpoint periodically
            if epoch % checkpoint_interval == 0 or epoch == epochs - 1:
                checkpoint_path = f"checkpoints/{self.monitor.experiment_name}_epoch_{epoch}.pth"
                self.model.save(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
                # Log model to wandb
                self.monitor.log_checkpoint(checkpoint_path)
            
            # 4. Log example boards to wandb periodically
            if self.monitor.use_wandb and (epoch % log_interval == 0 or epoch == epochs - 1):
                # Extract some example boards from training samples
                if samples:
                    sample_indices = list(range(0, len(samples), max(1, len(samples)//8)))[:8]
                    board_examples = [samples[i][0] for i in sample_indices]
                    self.monitor.log_board_examples(board_examples, f"epoch_{epoch}_boards")
        
        # Finish wandb run when training is complete
        self.monitor.finish()
        
        return self.monitor.run_data