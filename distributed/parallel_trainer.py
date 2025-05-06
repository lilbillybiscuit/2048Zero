"""
Simplified trainer implementation for AlphaZero 2048
"""

import os
import time
import random
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from zero import GameRules, ZeroNetwork, ZeroMonitor

from .parallel_selfplay import MultiprocessSelfPlayWorker, worker_process


class ParallelZeroTrainer:
    """Simplified ZeroTrainer that uses parallel self-play for training"""
    
    def __init__(
        self,
        model: ZeroNetwork,
        rules: GameRules,
        use_wandb: bool = True,
        project_name: str = "2048-zero-parallel",
        experiment_name: Optional[str] = None,
        num_workers: int = None,  # Default to CPU count
        seed: int = 42,
        reward_function = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize the trainer with model and rules
        
        Args:
            model: Neural network model
            rules: Game rules
            use_wandb: Whether to use wandb for tracking
            project_name: Project name for wandb
            experiment_name: Experiment name for wandb
            num_workers: Number of worker processes for parallel self-play
            seed: Random seed for initialization
            reward_function: Custom function to calculate reward value
            enable_monitoring: Whether to enable the visualization web server
        """
        self.model = model
        self.rules = rules
        self.seed = seed
        self.reward_function = reward_function
        self.enable_monitoring = enable_monitoring
        
        # Set the random seed for reproducibility
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Set number of workers - simple version
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        print(f"Using {self.num_workers} worker processes")
        
        # Generate experiment name if none provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            board_size = f"{rules.height}x{rules.width}"
            model_size = f"f{model.conv1.out_channels}_b{len(model.blocks)}"
            experiment_name = f"zero_{board_size}_{model_size}_{timestamp}"
        
        self.experiment_name = experiment_name
            
        # Initialize monitoring
        config = {
            "model_filters": model.conv1.out_channels,
            "model_blocks": len(model.blocks),
            "model_k": model.k,
            "board_height": rules.height,
            "board_width": rules.width,
            "num_workers": self.num_workers,
            "seed": seed,
            "reward_function": "custom" if self.reward_function else "default_score"
        }
        
        # Initialize wandb monitoring
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
        
        # Initialize web monitor if enabled
        if self.enable_monitoring:
            try:
                from visual.monitor_client import start_monitor_server
                self.monitor_running = start_monitor_server(host='0.0.0.0', port=5000, open_browser=True)
                print("Monitoring server started at http://localhost:5000")
            except Exception as e:
                print(f"Warning: Monitoring server could not be started: {e}")
                self.monitor_running = False
    
    def save_model(self):
        """Save the current model to the shared model path"""
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
        self.model.eval()
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
            seed=self.seed,
            reward_function=self.reward_function,
            enable_monitoring=self.enable_monitoring
        )
        
        # Generate games in parallel
        samples, stats = worker.generate_games()
        
        return samples, stats
    
    def _update_network(
        self, 
        samples: List[Tuple], 
        optimizer: torch.optim.Optimizer, 
        batch_size: int, 
        epoch: int,
        device: str = 'cuda:0'
    ) -> Tuple[float, float, float]:
        """
        Update neural network weights using collected samples in batches
        
        Args:
            samples: List of (board, policy, value) tuples
            optimizer: PyTorch optimizer
            batch_size: Training batch size
            epoch: Current epoch number (for logging)
            device: Device to use for training
            
        Returns:
            Tuple of (average_loss, average_policy_loss, average_value_loss)
        """
        self.model.train()
        
        # Move model to specified device
        if not torch.cuda.is_available() and device.startswith('cuda'):
            device = 'cpu'
            print("CUDA not available, using CPU for training")
            
        self.model.to(device)
        
        # Shuffle samples
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
            
            # Convert to tensors
            boards_array = np.stack(boards, axis=0)
            state_tensor = self.model.to_onehot(boards_array).to(device)
            pi_tensor = torch.tensor(pis, dtype=torch.float32).to(device)
            z_tensor = torch.tensor(zs, dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()

            # Forward pass
            p_pred, v_pred = self.model(state_tensor)

            # Compute losses
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
            
            # Log batch metrics
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
            **kwargs: Additional arguments
        """
        # Select training device
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
            
        # Initialize monitor config with training parameters
        self.monitor.config.update({
            "epochs": epochs,
            "games_per_epoch": games_per_epoch,
            "batch_size": batch_size,
            "learning_rate": lr,
            "scheduler_step_size": kwargs.get('scheduler_step_size', 20),
            "scheduler_gamma": kwargs.get('scheduler_gamma', 0.5),
            "simulations": kwargs.get('simulations', 50),
            "weight_decay": kwargs.get('weight_decay', 1e-4),
            "num_workers": self.num_workers,
            "device": device
        })
        
        # Initialize wandb with the model
        if self.monitor.use_wandb and not self.monitor.wandb_initialized:
            self.monitor.init_wandb(watch_model=self.model)
        
        # Initialize optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=kwargs.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('scheduler_step_size', 20),
            gamma=kwargs.get('scheduler_gamma', 0.5)
        )
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 1. Collect training data with parallel self-play
            print(f"Epoch {epoch+1}/{epochs}")
            
            samples, epoch_stats = self._collect_selfplay_data_parallel(
                games_per_epoch=games_per_epoch,
                simulations=kwargs.get('simulations', 50)
            )
            
            # Log self-play statistics
            self.monitor.log_selfplay_stats(epoch_stats, epoch)
            
            # Update monitoring server if enabled
            if self.enable_monitoring:
                try:
                    from visual.monitor_client import update_training_stats
                    update_training_stats(
                        current_epoch=epoch+1,
                        total_epochs=epochs,
                        samples_collected=len(samples),
                        games_played=epoch_stats.get('games', 0),
                        max_tile_achieved=epoch_stats.get('max_tile', 0),
                        max_score=epoch_stats.get('max_score', 0)
                    )
                except:
                    pass  # Silently fail if monitoring update fails
            
            # 2. Update neural network if samples were collected
            loss, pi_loss, v_loss = None, None, None
            if samples:
                # Train on all collected samples
                loss, pi_loss, v_loss = self._update_network(
                    samples, optimizer, batch_size, epoch=epoch, device=device
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
                
                # Update monitoring server with loss metrics
                if self.enable_monitoring:
                    try:
                        from visual.monitor_client import update_training_stats
                        update_training_stats(
                            current_epoch=epoch+1,
                            total_epochs=epochs,
                            latest_loss=loss,
                            policy_loss=pi_loss,
                            value_loss=v_loss
                        )
                    except:
                        pass  # Silently fail if monitoring update fails
                
                # Save the updated model for next epoch
                self.save_model()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            max_tile = epoch_stats.get('max_tile', 0)
            max_score = epoch_stats.get('max_score', 0)
            avg_score = epoch_stats.get('total_score', 0) / max(1, epoch_stats.get('games', 1))
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            
            print(f"Epoch {epoch+1} summary: MaxScore={max_score}, MaxTile={max_tile}, " +
                 f"AvgScore={avg_score:.1f}, Loss={loss_str}, Time={epoch_time:.1f}s")
            
            # Save checkpoint periodically
            if epoch % checkpoint_interval == 0 or epoch == epochs - 1:
                checkpoint_path = f"checkpoints/{self.monitor.experiment_name}_epoch_{epoch}.pth"
                self.model.save(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
                # Log model to wandb
                self.monitor.log_checkpoint(checkpoint_path)
                
        # Finish wandb run when training is complete
        self.monitor.finish()
        
        return self.monitor