"""
Simplified parallel self-play implementation for AlphaZero 2048
"""

import os
import time
import pickle
import multiprocessing
from typing import List, Dict, Tuple, Any

import torch

from zero import GameRules, GameState, BitBoard, ZeroPlayer, ZeroNetwork
from zero.zeromonitoring import EpochStats
from zero.reward_functions import default_reward_func

# Set default multiprocessing start method for PyTorch compatibility
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass


def play_single_game(
    player: ZeroPlayer, 
    rules: GameRules, 
    simulations: int = 50,
    game_id: str = None,
    reward_function = None,
    monitor_updates: bool = False,
    worker_id: int = None
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Play a single self-play game and return samples and statistics
    
    Args:
        player: ZeroPlayer instance
        rules: GameRules instance
        simulations: Number of MCTS simulations per move
        game_id: Optional game identifier for debugging
        reward_function: Custom function to calculate reward
        monitor_updates: Whether to send updates to the monitor
        worker_id: ID of the worker (for monitor updates)
        
    Returns:
        Tuple of (samples, game_stats)
    """
    # Use provided reward function or default to score reward from zero module
    if reward_function is None:
        reward_function = default_reward_func
    
    samples = []
    state = rules.get_initial_state()
    trajectory = []
    turn_count = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    # Setup monitoring if enabled
    if monitor_updates and worker_id is not None:
        try:
            from visual.monitor_client import update_game_state
        except ImportError:
            monitor_updates = False
    
    # Play until terminal state
    while not rules.is_terminal(state.board):
        turn_count += 1
        
        # Get action from player with MCTS
        action, (pi_probs, value_est) = player.play(state, simulations=simulations)
        action_counts[action] += 1
        
        # Save state and policy
        trajectory.append((state.board.copy(), pi_probs))
        
        # Apply action and add random tiles
        new_board, gain = rules.apply_move(state.board, action)
        new_board = rules.add_random_tiles(new_board)
        new_bitboard = BitBoard.from_numpy(new_board)
        state = GameState(new_board, state.score + gain, new_bitboard)
        
        # Send update to monitor if enabled
        if monitor_updates and worker_id is not None and turn_count % 2 == 0:  # Update every other turn to reduce overhead
            try:
                update_game_state(worker_id, state.board.tolist(), state.score, turn_count)
            except:
                pass  # Silently fail if monitor update fails
    
    # Calculate final game statistics
    max_tile = rules.get_max_tile(state.board)
    
    # Create game statistics
    game_stats = {
        'score': state.score,
        'max_tile': max_tile,
        'turns': turn_count,
        'action_counts': action_counts,
        'final_board': state.board,
        'game_id': game_id
    }
    
    # Calculate reward using the provided function
    z, reward_type = reward_function(state, game_stats)
    
    # Create samples from trajectory with the calculated reward value
    for (board_t, pi_t) in trajectory:
        samples.append((board_t, pi_t, z))
    
    # Add the reward value to game stats for logging
    game_stats['reward'] = z
    game_stats['reward_type'] = reward_type
    
    # Send final update to monitor
    if monitor_updates and worker_id is not None:
        try:
            update_game_state(worker_id, state.board.tolist(), state.score, turn_count)
        except:
            pass  # Silently fail if monitor update fails
    
    return samples, game_stats


def worker_process(
    worker_id: int,
    model_path: str,
    num_games: int,
    simulations: int,
    samples_path: str,
    stats_path: str,
    device: str,
    seed: int,
    reward_function=None,
    enable_monitoring=True
):
    """
    Worker process function that generates self-play games
    
    Args:
        worker_id: Unique ID for this worker
        model_path: Path to model weights file
        num_games: Number of games to play
        simulations: MCTS simulations per move
        samples_path: Path to save samples
        stats_path: Path to save statistics
        device: Device to use for model inference
        seed: Random seed for reproducibility
        reward_function: Custom function to calculate reward value
        enable_monitoring: Whether to enable monitoring server updates
    """
    try:
        # Set random seed for this worker
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Staggered initialization to prevent CUDA contention
        if device.startswith('cuda'):
            # Add a small delay based on worker_id to prevent all workers
            # from initializing CUDA context simultaneously
            time.sleep(0.5 * worker_id)
            
            # Set thread limits to avoid resource contention
            if torch.cuda.is_available():
                torch.set_num_threads(1)
                
                # Explicitly set device before any other CUDA operations
                device_idx = int(device.split(':')[1]) if ':' in device else 0
                torch.cuda.set_device(device_idx)
        
        # Create game rules
        rules = GameRules()
        
        # Load model - always load to CPU first
        model = ZeroNetwork(rules.height, rules.width, 16)  # 16 for max tile exponent
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        print(f"Worker {worker_id} initializing on {device}")
        
        # Move model to specified device
        model = model.to(device)
        model.eval()
        
        # Create player
        player = ZeroPlayer(model, rules)
        
        # Initialize statistics
        samples = []
        stats = EpochStats(
            height=rules.height,
            width=rules.width,
            direction_names=rules.DIRECTION_NAMES
        )
        
        print(f"Worker {worker_id} ready on {device}, playing {num_games} games")
        
        # Play games
        for game_idx in range(num_games):
            try:
                worker_samples, game_stats = play_single_game(
                    player=player,
                    rules=rules,
                    simulations=simulations,
                    game_id=f"w{worker_id}_g{game_idx}",
                    reward_function=reward_function,
                    monitor_updates=enable_monitoring,
                    worker_id=worker_id
                )
                
                # Add samples from this game
                samples.extend(worker_samples)
                
                # Update statistics
                stats.update_game_stats(
                    score=game_stats['score'],
                    max_tile=game_stats['max_tile'],
                    turns=game_stats['turns'],
                    action_counts=game_stats['action_counts'],
                    board=game_stats['final_board']
                )
                
                # Print game completed
                print(f"W{worker_id} Game {game_idx+1}/{num_games} - Score: {game_stats['score']}, " +
                      f"Max Tile: {game_stats['max_tile']}, Reward: {game_stats['reward']:.3f} ({game_stats['reward_type']})")
                
                # Update reward type in the statistics object
                stats.update_reward_type(game_stats['reward_type'])
                
            except Exception as game_error:
                print(f"Worker {worker_id}, Game {game_idx} failed: {game_error}")
                import traceback
                traceback.print_exc()
                # Continue with the next game instead of crashing the entire worker
                continue
        
        # Save samples and statistics
        print(f"Worker {worker_id} saving {len(samples)} samples to {samples_path}")
        with open(samples_path, 'wb') as f:
            pickle.dump(samples, f)
        
        with open(stats_path, 'wb') as f:
            pickle.dump(stats.get_stats(), f)
            
        print(f"Worker {worker_id} completed successfully")
            
    except Exception as e:
        print(f"Worker {worker_id} failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


class MultiprocessSelfPlayWorker:
    """Simplified worker class for self-play game generation using multiple processes"""
    
    def __init__(
        self,
        model_path: str,
        num_workers: int = None,
        games_per_worker: int = 10,
        simulations_per_move: int = 50,
        temp_dir: str = "temp_data",
        seed: int = 42,
        reward_function = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize the self-play worker
        
        Args:
            model_path: Path to the model weights file
            num_workers: Number of worker processes (default: use CPU count)
            games_per_worker: Number of games each worker should play
            simulations_per_move: MCTS simulations per move
            temp_dir: Directory for temporary data storage
            seed: Random seed for initialization
            reward_function: Custom function to calculate reward value
            enable_monitoring: Whether to enable real-time monitoring
        """
        self.model_path = model_path
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.games_per_worker = games_per_worker
        self.simulations_per_move = simulations_per_move
        self.temp_dir = temp_dir
        self.seed = seed
        self.reward_function = reward_function
        self.enable_monitoring = enable_monitoring
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def generate_games(self) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Generate self-play games in parallel using multiple processes
        
        Returns:
            Tuple of (samples, statistics) where samples is a list of (board, policy, value) tuples
            and statistics is a dictionary of game statistics
        """
        # For very large worker counts, reduce to avoid system resource exhaustion
        effective_workers = min(self.num_workers, os.cpu_count() * 2)
        if effective_workers != self.num_workers:
            print(f"Limiting to {effective_workers} workers for system stability (from requested {self.num_workers})")
            self.num_workers = effective_workers
        
        # Determine device assignment
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} CUDA devices for self-play")
        
        # If CUDA devices are available, cap workers per GPU to avoid OOM
        if gpu_count > 0:
            max_workers_per_gpu = 4  # Limit workers per GPU to avoid memory issues
            max_gpu_workers = gpu_count * max_workers_per_gpu
            cpu_workers = max(0, self.num_workers - max_gpu_workers)
            
            print(f"Distributing workers: {min(self.num_workers, max_gpu_workers)} on GPUs, {cpu_workers} on CPU")
        
        # Prepare task arguments for each worker
        worker_args = []
        for worker_id in range(self.num_workers):
            # Assign device with a balanced strategy
            if gpu_count > 0 and worker_id < max_gpu_workers:
                device = f'cuda:{worker_id % gpu_count}'
            else:
                device = 'cpu'
            
            # Set unique seed for each worker
            worker_seed = self.seed + worker_id
            
            worker_args.append((
                worker_id,
                self.model_path,
                self.games_per_worker,
                self.simulations_per_move,
                f"{self.temp_dir}/worker_{worker_id}_samples.pkl",
                f"{self.temp_dir}/worker_{worker_id}_stats.pkl",
                device,
                worker_seed,
                self.reward_function,
                self.enable_monitoring
            ))
        
        # Launch worker processes
        start_time = time.time()
        print(f"Starting {self.num_workers} self-play workers")
        
        # Use a try-finally block to ensure proper cleanup
        try:
            # Use a smaller batch size of workers to avoid system resource exhaustion
            batch_size = 8
            for batch_start in range(0, self.num_workers, batch_size):
                batch_end = min(batch_start + batch_size, self.num_workers)
                batch_args = worker_args[batch_start:batch_end]
                batch_count = len(batch_args)
                
                print(f"Starting worker batch {batch_start//batch_size + 1} with {batch_count} workers")
                
                # Launch this batch of workers
                with multiprocessing.get_context('spawn').Pool(batch_count) as pool:
                    pool.starmap(worker_process, batch_args)
        
            print("All self-play workers completed")
        except Exception as e:
            print(f"Error during self-play: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Collect samples and statistics from all workers
        all_samples = []
        combined_stats = self._initialize_combined_stats()
        
        successful_workers = 0
        for worker_id in range(self.num_workers):
            # Load samples
            sample_path = f"{self.temp_dir}/worker_{worker_id}_samples.pkl"
            try:
                with open(sample_path, 'rb') as f:
                    worker_samples = pickle.load(f)
                    all_samples.extend(worker_samples)
                
                # Load statistics
                stats_path = f"{self.temp_dir}/worker_{worker_id}_stats.pkl"
                with open(stats_path, 'rb') as f:
                    worker_stats = pickle.load(f)
                    self._merge_worker_stats(combined_stats, worker_stats)
                    
                successful_workers += 1
                
                # Clean up temporary files
                try:
                    os.remove(sample_path)
                    os.remove(stats_path)
                except FileNotFoundError:
                    pass
                    
            except (FileNotFoundError, EOFError) as e:
                print(f"Warning: Could not load data from worker {worker_id}: {e}")
                continue
        
        # Update total samples count
        combined_stats['samples'] = len(all_samples)
        
        # Calculate elapsed time and throughput
        elapsed_time = time.time() - start_time
        combined_stats['generation_time'] = elapsed_time
        combined_stats['games_per_second'] = combined_stats['games'] / elapsed_time if elapsed_time > 0 else 0
        combined_stats['samples_per_second'] = len(all_samples) / elapsed_time if elapsed_time > 0 else 0
        
        print(f"Self-play complete: {successful_workers}/{self.num_workers} workers succeeded")
        print(f"Generated: {combined_stats['games']} games, {len(all_samples)} samples in {elapsed_time:.1f}s")
        
        return all_samples, combined_stats
    
    @staticmethod
    def _initialize_combined_stats() -> Dict[str, Any]:
        """Initialize an empty stats dictionary with zeros"""
        return {
            'max_score': 0,
            'max_tile': 0,
            'max_turns': 0,
            'total_score': 0,
            'total_tile': 0,
            'total_turns': 0,
            'games': 0,
            'tile_counts': {2**i: 0 for i in range(1, 16)},
            'game_lengths': [],
            'game_scores': [],
            'game_max_tiles': [],
            'action_counts': {0: 0, 1: 0, 2: 0, 3: 0},
            'samples': 0
        }
    
    @staticmethod
    def _merge_worker_stats(combined_stats: Dict[str, Any], worker_stats: Dict[str, Any]):
        """Merge a worker's statistics into the combined statistics"""
        # Update max values
        combined_stats['max_score'] = max(combined_stats['max_score'], worker_stats.get('max_score', 0))
        combined_stats['max_tile'] = max(combined_stats['max_tile'], worker_stats.get('max_tile', 0))
        combined_stats['max_turns'] = max(combined_stats['max_turns'], worker_stats.get('max_turns', 0))
        
        # Update totals
        combined_stats['total_score'] += worker_stats.get('total_score', 0)
        combined_stats['total_tile'] += worker_stats.get('total_tile', 0)
        combined_stats['total_turns'] += worker_stats.get('total_turns', 0)
        combined_stats['games'] += worker_stats.get('games', 0)
        
        # Extend lists
        combined_stats['game_lengths'].extend(worker_stats.get('game_lengths', []))
        combined_stats['game_scores'].extend(worker_stats.get('game_scores', []))
        combined_stats['game_max_tiles'].extend(worker_stats.get('game_max_tiles', []))
        
        # Merge tile counts
        for tile, count in worker_stats.get('tile_counts', {}).items():
            if tile in combined_stats['tile_counts']:
                combined_stats['tile_counts'][tile] += count
        
        # Merge action counts
        for action, count in worker_stats.get('action_counts', {}).items():
            if action in combined_stats['action_counts']:
                combined_stats['action_counts'][action] += count