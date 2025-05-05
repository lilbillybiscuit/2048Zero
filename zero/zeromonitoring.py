"""
Monitoring and logging utilities for 2048 Zero training
"""

import os
import json
import time
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

# Custom JSON encoder to handle numpy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# Import wandb with fallback if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not installed. Run 'pip install wandb' for enhanced logging.")


class ZeroMonitor:
    """Monitoring and logging utilities for 2048 Zero training"""
    
    def __init__(
        self,
        use_wandb: bool = True,  # Default to True
        project_name: str = "2048-zero",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        resume: bool = False,
    ):
        """Initialize the monitor
        
        Args:
            use_wandb: Whether to use wandb for tracking (default: True)
            project_name: Project name for wandb
            experiment_name: Experiment name for wandb (auto-generated if None)
            config: Configuration for wandb
            resume: Whether to resume wandb run
        """
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_initialized = False
        self.project_name = project_name
        self.experiment_name = experiment_name if experiment_name else f"zero_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config or {}
        self.resume = resume
        
        # Global step counter for consistent wandb logging
        self.global_step = 1  # Start from 1 to avoid 0-step issues
        
        # Statistics history
        self.run_data = {
            "start_time": datetime.now().isoformat(),
            "epochs_completed": 0,
            "model_config": {},
            "game_config": {},
            "training_config": {},
            "epoch_stats": []
        }
        
        # Initialize wandb if enabled
        if self.use_wandb:
            self.init_wandb()
            
    def init_wandb(self, watch_model: Optional[Any] = None):
        """Initialize wandb
        
        Args:
            watch_model: Optional model to watch for gradient tracking
        """
        if not self.use_wandb or self.wandb_initialized:
            return
            
        # Initialize wandb
        wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config=self.config,
            resume=self.resume
        )
        
        # Mark as initialized
        self.wandb_initialized = True
        
        # Watch model if provided
        if watch_model is not None:
            wandb.watch(watch_model, log="all", log_freq=100)
            
    def log_selfplay_stats(self, epoch_stats: Dict[str, Any], epoch: int):
        """Log self-play statistics to wandb
        
        Args:
            epoch_stats: Dictionary of statistics from self-play
            epoch: Current epoch number
        """
        if not self.use_wandb or not self.wandb_initialized:
            return
            
        # Use global step counter to ensure monotonically increasing steps
        self.global_step += 1
            
        # Calculate statistics - convert numpy types to Python native types
        games = int(epoch_stats['games'])
        avg_score = int(epoch_stats['total_score']) / games if games > 0 else 0
        avg_tile = int(epoch_stats['total_tile']) / games if games > 0 else 0
        avg_turns = int(epoch_stats['total_turns']) / games if games > 0 else 0
        
        # Create a histogram of tile counts
        tile_counts = {f"tiles/{tile}": int(count) for tile, count in epoch_stats.get('tile_counts', {}).items()}
        
        # Calculate action distribution metrics (if available)
        action_metrics = {}
        if 'action_counts' in epoch_stats:
            total_actions = sum(int(count) for count in epoch_stats['action_counts'].values())
            if total_actions > 0:
                for action, count in epoch_stats['action_counts'].items():
                    action_name = action  # Default to number if name not available
                    if 'action_names' in epoch_stats and action in epoch_stats['action_names']:
                        action_name = epoch_stats['action_names'][action]
                    action_metrics[f"actions/{action_name}"] = int(count) / total_actions
        
        # Convert score and tile lists to histograms for wandb
        # Convert numpy arrays to Python lists if needed
        metrics_dict = {
            "selfplay/games": games,
            "selfplay/max_score": int(epoch_stats['max_score']),
            "selfplay/max_tile": int(epoch_stats['max_tile']),
            "selfplay/max_turns": int(epoch_stats['max_turns']),
            "selfplay/avg_score": avg_score,
            "selfplay/avg_tile": avg_tile,
            "selfplay/avg_turns": avg_turns,
            "selfplay/samples": int(epoch_stats.get('samples', 0)),
            "epoch": int(epoch),
            **tile_counts,  # Add tile counts
            **action_metrics,  # Add action distribution
        }
        
        # Add histograms if available
        if 'game_scores' in epoch_stats and epoch_stats['game_scores']:
            game_scores = [int(score) for score in epoch_stats['game_scores']]
            metrics_dict["selfplay/score_histogram"] = wandb.Histogram(game_scores)
        
        if 'game_max_tiles' in epoch_stats and epoch_stats['game_max_tiles']:
            game_max_tiles = [int(tile) for tile in epoch_stats['game_max_tiles']]
            metrics_dict["selfplay/max_tile_histogram"] = wandb.Histogram(game_max_tiles)
        
        if 'game_lengths' in epoch_stats and epoch_stats['game_lengths']:
            game_lengths = [int(length) for length in epoch_stats['game_lengths']]
            metrics_dict["selfplay/turns_histogram"] = wandb.Histogram(game_lengths)
        
        # Log all metrics in a single call
        wandb.log(metrics_dict, step=self.global_step)
        
    def log_training_stats(self, 
                          loss: float, 
                          policy_loss: float, 
                          value_loss: float, 
                          learning_rate: float, 
                          samples: int, 
                          epoch: int):
        """Log training statistics to wandb
        
        Args:
            loss: Total loss
            policy_loss: Policy loss
            value_loss: Value loss
            learning_rate: Current learning rate
            samples: Number of samples
            epoch: Current epoch number
        """
        if not self.use_wandb or not self.wandb_initialized:
            return
            
        # Use global step counter to ensure monotonically increasing steps
        self.global_step += 1
            
        # Convert any numpy types to Python native types
        loss_float = float(loss)
        policy_loss_float = float(policy_loss)
        value_loss_float = float(value_loss)
        learning_rate_float = float(learning_rate)
        samples_int = int(samples)
        epoch_int = int(epoch)
            
        # Log training metrics
        wandb.log({
            "train/loss": loss_float,
            "train/policy_loss": policy_loss_float,
            "train/value_loss": value_loss_float,
            "train/learning_rate": learning_rate_float,
            "train/epoch": epoch_int,
            "train/samples": samples_int
        }, step=self.global_step)
        
    def log_batch_stats(self, 
                       batch_loss: float, 
                       batch_policy_loss: float, 
                       batch_value_loss: float, 
                       batch_size: int, 
                       epoch: int, 
                       step: int = None):  # Make step optional
        """Log batch statistics to wandb
        
        Args:
            batch_loss: Loss for this batch
            batch_policy_loss: Policy loss for this batch
            batch_value_loss: Value loss for this batch
            batch_size: Batch size
            epoch: Current epoch
            step: Current step (not used, kept for backward compatibility)
        """
        if not self.use_wandb or not self.wandb_initialized:
            return
        
        # Use global step counter to ensure monotonically increasing steps
        self.global_step += 1
        
        # Convert any numpy types to Python native types
        batch_loss_float = float(batch_loss)
        batch_policy_loss_float = float(batch_policy_loss)
        batch_value_loss_float = float(batch_value_loss)
        batch_size_int = int(batch_size)
        epoch_int = int(epoch)
            
        # Log batch metrics
        wandb.log({
            "batch/loss": batch_loss_float,
            "batch/policy_loss": batch_policy_loss_float,
            "batch/value_loss": batch_value_loss_float,
            "batch/epoch": epoch_int,
            "batch/batch_size": batch_size_int,
        }, step=self.global_step)
        
    def log_game_stats(self, 
                      score: int, 
                      max_tile: int, 
                      turns: int, 
                      value_target: float, 
                      trajectory_length: int,
                      action_dist: Dict[str, float]):
        """Log individual game statistics to wandb
        
        Args:
            score: Final game score
            max_tile: Maximum tile achieved
            turns: Number of turns played
            value_target: MCTS value target
            trajectory_length: Length of game trajectory
            action_dist: Action distribution dictionary
        """
        if not self.use_wandb or not self.wandb_initialized:
            return
            
        # Use global step counter to ensure monotonically increasing steps
        self.global_step += 1
            
        # Convert any numpy types to Python native types
        score_int = int(score)
        max_tile_int = int(max_tile)
        turns_int = int(turns)
        value_target_float = float(value_target)
        trajectory_length_int = int(trajectory_length)
        
        # Convert any action distribution values that might be numpy types
        action_dist_native = {k: float(v) for k, v in action_dist.items()}
            
        # Log game metrics
        wandb.log({
            "game/score": score_int,
            "game/max_tile": max_tile_int,
            "game/turns": turns_int,
            "game/value_target": value_target_float,
            "game/trajectory_length": trajectory_length_int,
            **action_dist_native  # Add action distribution
        }, step=self.global_step)
        
    def log_board_examples(self, boards: List[np.ndarray], prefix: str = "board"):
        """Log board examples to wandb as a grid of images
        
        Args:
            boards: List of board arrays
            prefix: Prefix for wandb key
        """
        if not self.use_wandb or not self.wandb_initialized or not boards:
            return
            
        # Use global step counter to ensure monotonically increasing steps
        self.global_step += 1
            
        # Convert each board to a color image
        images = []
        for board in boards[:16]:  # Limit to max 16 examples
            # Create a colored representation
            height, width = board.shape
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Color map for different tiles (from light to dark)
            colormap = {
                0: (204, 192, 179),   # Empty cell
                1: (238, 228, 218),   # 2
                2: (237, 224, 200),   # 4
                3: (242, 177, 121),   # 8
                4: (245, 149, 99),    # 16
                5: (246, 124, 95),    # 32
                6: (246, 94, 59),     # 64
                7: (237, 207, 114),   # 128
                8: (237, 204, 97),    # 256
                9: (237, 200, 80),    # 512
                10: (237, 197, 63),   # 1024
                11: (237, 194, 46),   # 2048
                12: (118, 224, 162),  # 4096
                13: (97, 191, 152),   # 8192
                14: (18, 130, 250),   # 16384
                15: (0, 102, 204)     # 32768
            }
            
            # Fill in the image
            for i in range(height):
                for j in range(width):
                    value = int(board[i, j])
                    color = colormap.get(value, (100, 100, 100))  # Default gray for unexpected values
                    img[i, j] = color
            
            # Resize for better visibility (optional)
            img_big = np.repeat(np.repeat(img, 20, axis=0), 20, axis=1)
            images.append(img_big)
        
        # Log the grid to wandb
        if images:
            grid_image = np.vstack([
                np.hstack(images[:4]),
                np.hstack(images[4:8]) if len(images) > 4 else np.zeros_like(np.hstack(images[:4])),
                np.hstack(images[8:12]) if len(images) > 8 else np.zeros_like(np.hstack(images[:4])),
                np.hstack(images[12:16]) if len(images) > 12 else np.zeros_like(np.hstack(images[:4]))
            ])
            wandb.log({f"{prefix}_grid": wandb.Image(grid_image)}, step=self.global_step)
            
    def log_checkpoint(self, path: str):
        """Log a model checkpoint to wandb
        
        Args:
            path: Path to the checkpoint file
        """
        if not self.use_wandb or not self.wandb_initialized:
            return
            
        # Use global step counter to ensure monotonically increasing steps
        self.global_step += 1
            
        # Log a summary to wandb with the step
        wandb.log({"checkpoint/saved": 1}, step=self.global_step)
            
        # Save model checkpoint to wandb
        wandb.save(path)
        
    def update_run_data(self, epoch_data: Dict[str, Any]):
        """Update run data with epoch statistics
        
        Args:
            epoch_data: Dictionary of epoch statistics
        """
        # Add epoch data to run history
        self.run_data["epoch_stats"].append(epoch_data)
        self.run_data["epochs_completed"] = len(self.run_data["epoch_stats"])
        
    def save_run_data(self, path: str):
        """Save run data to a JSON file
        
        Args:
            path: Path to save the run data
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save run data using our custom NumpyJSONEncoder to handle numpy types
        with open(path, "w") as f:
            json.dump(self.run_data, f, indent=2, cls=NumpyJSONEncoder)
            
    def finish(self):
        """Finish wandb run"""
        if self.use_wandb and self.wandb_initialized:
            wandb.finish()


class EpochStats:
    """Utility class for collecting and computing epoch statistics"""
    
    def __init__(self, height: int, width: int, direction_names: Dict[int, str] = None):
        """Initialize epoch statistics
        
        Args:
            height: Board height
            width: Board width
            direction_names: Dictionary mapping direction codes to names
        """
        # Convert any numpy types to Python native types
        height_int = int(height)
        width_int = int(width)
        
        self.stats = {
            'max_score': 0, 
            'max_tile': 0, 
            'max_turns': 0,
            'total_score': 0, 
            'total_tile': 0, 
            'total_turns': 0, 
            'games': 0,
            'tile_counts': {2**i: 0 for i in range(1, 16)},  # Track counts of each tile value
            'game_lengths': [],  # Track individual game lengths
            'game_scores': [],   # Track individual game scores
            'game_max_tiles': [], # Track max tile for each game
            'action_counts': {},  # Track action counts
            'samples': 0,  # Number of samples collected
            'reward_type': 'score',  # Default reward type is score (not max_tile)
        }
        
        if direction_names:
            # Convert any numpy types in direction names
            clean_direction_names = {int(k): str(v) for k, v in direction_names.items()}
            self.stats['action_names'] = clean_direction_names
        
    def update_game_stats(self, 
                         score: int, 
                         max_tile: int, 
                         turns: int, 
                         action_counts: Dict[int, int],
                         board: np.ndarray = None):
        """Update statistics with a completed game
        
        Args:
            score: Final game score
            max_tile: Maximum tile achieved
            turns: Number of turns played
            action_counts: Dictionary of action counts
            board: Final game board
        """
        # Convert numpy types to native Python types
        score_int = int(score)
        max_tile_int = int(max_tile)
        turns_int = int(turns)
        
        # Update max values
        self.stats['max_score'] = max(self.stats['max_score'], score_int)
        self.stats['max_tile'] = max(self.stats['max_tile'], max_tile_int)
        self.stats['max_turns'] = max(self.stats['max_turns'], turns_int)
        
        # Update totals
        self.stats['total_score'] += score_int
        self.stats['total_tile'] += max_tile_int
        self.stats['total_turns'] += turns_int
        self.stats['games'] += 1
        
        # Update lists
        self.stats['game_lengths'].append(turns_int)
        self.stats['game_scores'].append(score_int)
        self.stats['game_max_tiles'].append(max_tile_int)
        
        # Update action counts - convert any numpy integers to Python integers
        for action, count in action_counts.items():
            action_key = int(action) if isinstance(action, np.integer) else action
            count_int = int(count) if isinstance(count, np.integer) else count
            
            if action_key not in self.stats['action_counts']:
                self.stats['action_counts'][action_key] = 0
            self.stats['action_counts'][action_key] += count_int
            
        # Count tiles on board if provided
        if board is not None:
            unique, counts = np.unique(board, return_counts=True)
            for tile_exp, count in zip(unique, counts):
                if tile_exp > 0:  # Skip empty cells
                    tile_val = 2**int(tile_exp)
                    if tile_val in self.stats['tile_counts']:
                        self.stats['tile_counts'][tile_val] += int(count)
                        
    def update_samples(self, num_samples: int):
        """Update the number of samples collected
        
        Args:
            num_samples: Number of samples
        """
        self.stats['samples'] = int(num_samples)
        
    def update_reward_type(self, reward_type: str):
        """Update the reward type used
        
        Args:
            reward_type: Name of the reward type (e.g., 'score', 'custom')
        """
        self.stats['reward_type'] = str(reward_type)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get the statistics dictionary
        
        Returns:
            Dictionary of statistics
        """
        return self.stats
        
    def compute_averages(self) -> Tuple[float, float, float]:
        """Compute average statistics
        
        Returns:
            Tuple of (avg_score, avg_tile, avg_turns)
        """
        games = int(self.stats['games'])
        avg_score = int(self.stats['total_score']) / games if games > 0 else 0
        avg_tile = int(self.stats['total_tile']) / games if games > 0 else 0
        avg_turns = int(self.stats['total_turns']) / games if games > 0 else 0
        return avg_score, avg_tile, avg_turns
        
    def get_progress_description(self, epoch: int, total_epochs: int,
                                loss: float = None, 
                                policy_loss: float = None, 
                                value_loss: float = None, 
                                learning_rate: float = None) -> str:
        """Generate a progress description for tqdm
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            loss: Total loss
            policy_loss: Policy loss
            value_loss: Value loss
            learning_rate: Current learning rate
            
        Returns:
            Progress description string
        """
        avg_score, avg_tile, avg_turns = self.compute_averages()
        
        # Basic statistics
        desc = (f"Epoch {epoch + 1}/{total_epochs}: "
                f"MaxTile: {self.stats['max_tile']} | "
                f"MaxScore: {self.stats['max_score']} | "
                f"AvgTile: {avg_tile:.1f} | "
                f"AvgScore: {avg_score:.1f} | "
                f"AvgTurns: {avg_turns:.1f}")
        
        # Add training metrics if available
        if loss is not None:
            desc += f" | loss={loss:.4f}"
            if policy_loss is not None and value_loss is not None:
                desc += f" (π:{policy_loss:.4f}, v:{value_loss:.4f})"
                
        # Add learning rate if available
        if learning_rate is not None:
            desc += f" | LR: {learning_rate:.6f}"
            
        return desc


def print_epoch_summary(stats: EpochStats, 
                      epoch: int, 
                      total_epochs: int,
                      loss: float = None, 
                      policy_loss: float = None, 
                      value_loss: float = None, 
                      learning_rate: float = None):
    """Print epoch summary using tqdm
    
    Args:
        stats: Epoch statistics
        epoch: Current epoch
        total_epochs: Total number of epochs
        loss: Total loss
        policy_loss: Policy loss
        value_loss: Value loss
        learning_rate: Current learning rate
    """
    desc = stats.get_progress_description(
        epoch, total_epochs, loss, policy_loss, value_loss, learning_rate
    )
    epoch_summary = tqdm(total=0, bar_format='{desc}', desc=desc)
    epoch_summary.close()