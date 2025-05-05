"""
Parallel self-play implementation for distributed AlphaZero 2048
"""

import os
import time
import math
import pickle
import multiprocessing
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch

from game_alt import GameRules, GameState, BitBoard
from zero2048 import ZeroPlayer
from zeromodel import ZeroNetwork
from zeromonitoring import EpochStats
from .gpu_utils import DeviceManager, seed_everything

# Set default multiprocessing start method for PyTorch compatibility
# Note: This is critical for CUDA compatibility
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass


def play_single_game(
    player: ZeroPlayer, 
    rules: GameRules, 
    simulations: int = 50,
    game_id: str = None
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Play a single self-play game and return samples and statistics
    
    Args:
        player: ZeroPlayer instance
        rules: GameRules instance
        simulations: Number of MCTS simulations per move
        game_id: Optional game identifier for debugging
        
    Returns:
        Tuple of (samples, game_stats)
    """
    import math
    
    samples = []
    state = rules.get_initial_state()
    trajectory = []
    turn_count = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
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
    
    # Calculate final game statistics
    max_tile = rules.get_max_tile(state.board)
    
    # Scale the final value between -1 and 1
    z = math.log2(int(max_tile) + 1) / 11.0 * 2 - 1
    
    # Create samples from trajectory with the final value
    for (board_t, pi_t) in trajectory:
        samples.append((board_t, pi_t, z))
    
    # Create game statistics
    game_stats = {
        'score': state.score,
        'max_tile': max_tile,
        'turns': turn_count,
        'action_counts': action_counts,
        'final_board': state.board,
        'game_id': game_id
    }
    
    return samples, game_stats


class MultiprocessSelfPlayWorker:
    """Worker class that handles self-play game generation using multiple processes"""
    
    def __init__(
        self,
        model_path: str,
        num_workers: int = None,
        games_per_worker: int = 10,
        simulations_per_move: int = 50,
        temp_dir: str = "temp_data",
        seed: int = 42
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
        """
        self.model_path = model_path
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()
        self.games_per_worker = games_per_worker
        self.simulations_per_move = simulations_per_move
        self.temp_dir = temp_dir
        self.seed = seed
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize device manager with no fallback to CPU
        self.device_manager = DeviceManager(allow_cpu_fallback=False)
        
    def generate_games(self) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Generate self-play games in parallel using multiple processes
        
        Returns:
            Tuple of (samples, statistics) where samples is a list of (board, policy, value) tuples
            and statistics is a dictionary of game statistics
        """
        # Prepare task arguments for each worker
        worker_args = []
        for worker_id in range(self.num_workers):
            # Assign device based on worker ID
            device = self.device_manager.assign_device(worker_id, self.num_workers)
            
            # Set unique seed for each worker based on base seed
            worker_seed = self.seed + worker_id
            
            worker_args.append((
                worker_id,
                self.model_path,
                self.games_per_worker,
                self.simulations_per_move,
                f"{self.temp_dir}/worker_{worker_id}_samples.pkl",
                f"{self.temp_dir}/worker_{worker_id}_stats.pkl",
                device,
                worker_seed
            ))
        
        # Launch worker processes
        print(f"Launching {self.num_workers} self-play worker processes")
        start_time = time.time()
        
        # Create a fresh process pool with the 'spawn' method for CUDA compatibility
        # The 'fork' method can cause issues with CUDA
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(self.num_workers) as pool:
            pool.starmap(self._worker_process, worker_args)
        
        # Collect samples and statistics from all workers
        all_samples = []
        combined_stats = self._initialize_combined_stats()
        
        for worker_id in range(self.num_workers):
            # Load samples
            sample_path = f"{self.temp_dir}/worker_{worker_id}_samples.pkl"
            try:
                with open(sample_path, 'rb') as f:
                    worker_samples = pickle.load(f)
                    all_samples.extend(worker_samples)
            except (FileNotFoundError, EOFError) as e:
                print(f"Warning: Could not load samples from worker {worker_id}: {e}")
                continue
            
            # Load statistics
            stats_path = f"{self.temp_dir}/worker_{worker_id}_stats.pkl"
            try:
                with open(stats_path, 'rb') as f:
                    worker_stats = pickle.load(f)
                    self._merge_worker_stats(combined_stats, worker_stats)
            except (FileNotFoundError, EOFError) as e:
                print(f"Warning: Could not load stats from worker {worker_id}: {e}")
                continue
            
            # Clean up temporary files
            try:
                os.remove(sample_path)
                os.remove(stats_path)
            except FileNotFoundError:
                pass
        
        # Update total samples count
        combined_stats['samples'] = len(all_samples)
        
        # Calculate elapsed time and throughput
        elapsed_time = time.time() - start_time
        combined_stats['generation_time'] = elapsed_time
        combined_stats['games_per_second'] = combined_stats['games'] / elapsed_time if elapsed_time > 0 else 0
        combined_stats['samples_per_second'] = len(all_samples) / elapsed_time if elapsed_time > 0 else 0
        
        print(f"Parallel self-play complete - {combined_stats['games']} games, "
              f"{len(all_samples)} samples in {elapsed_time:.1f}s "
              f"({combined_stats['samples_per_second']:.1f} samples/s)")
        
        return all_samples, combined_stats
    
    @staticmethod
    def _worker_process(
        worker_id: int,
        model_path: str,
        num_games: int,
        simulations: int,
        samples_path: str,
        stats_path: str,
        device: str,
        seed: int
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
        """
        try:
            # Set random seed for this worker
            seed_everything(seed)
            
            # Create game rules
            rules = GameRules()
            
            # Load model
            # Always load to CPU first, then explicitly move to the target device
            # This is important for CUDA compatibility with multiprocessing
            model = ZeroNetwork(rules.height, rules.width, 16)  # 16 for max tile exponent
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            # If using CUDA, handle possible CUDA initialization issues
            if device.startswith('cuda'):
                # Extract device index
                device_idx = int(device.split(':')[1]) if ':' in device else 0
                
                # Use our improved setup_cuda_for_worker function
                from .cuda_utils import setup_cuda_for_worker, cuda_operation_timeout, TimeoutException
                
                try:
                    # Don't allow fallback - raise exception if CUDA initialization fails
                    setup_cuda_for_worker(device_idx, allow_cpu_fallback=False)
                    print(f"Worker {worker_id} - CUDA context successfully initialized on device {device}")
                except Exception as e:
                    # If we get here, there was a CUDA error and we need to stop the worker
                    error_msg = f"Worker {worker_id} - CUDA initialization failed: {e}"
                    print(f"ERROR: {error_msg}")
                    # Re-raise to stop the worker
                    raise RuntimeError(error_msg)
            
            # Now move model to device with timeout protection
            try:
                # This operation can sometimes hang, so we use a timeout
                if device.startswith('cuda'):
                    with cuda_operation_timeout(seconds=30):
                        model = model.to(device)
                else:
                    model = model.to(device)
                
                model.eval()  # Set to evaluation mode
            except TimeoutException as e:
                # Don't fall back to CPU - raise an exception if CUDA timeout occurs
                error_msg = f"Worker {worker_id} - Model transfer to {device} timed out: {e}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            
            print(f"Worker {worker_id} initialized model on device: {device}")
            
            # Create player (original implementation is now thread-safe)
            player = ZeroPlayer(model, rules)
            
            # Initialize statistics
            samples = []
            stats = EpochStats(
                height=rules.height,
                width=rules.width,
                direction_names=rules.DIRECTION_NAMES
            )
            
            # Play games
            for game_idx in range(num_games):
                try:
                    worker_samples, game_stats = play_single_game(
                        player=player,
                        rules=rules,
                        simulations=simulations,
                        game_id=f"w{worker_id}_g{game_idx}"
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
                    
                    print(f"Worker {worker_id} completed game {game_idx+1}/{num_games} - "
                          f"Score: {game_stats['score']}, Max Tile: {game_stats['max_tile']}")
                    
                except Exception as game_error:
                    print(f"Worker {worker_id}, Game {game_idx} failed: {game_error}")
            
            # Save samples and statistics
            with open(samples_path, 'wb') as f:
                pickle.dump(samples, f)
            
            with open(stats_path, 'wb') as f:
                pickle.dump(stats.get_stats(), f)
                
        except Exception as e:
            print(f"Worker {worker_id} failed with error: {e}")
            import traceback
            traceback.print_exc()
            
            # Save empty results to avoid hanging
            with open(samples_path, 'wb') as f:
                pickle.dump([], f)
            with open(stats_path, 'wb') as f:
                pickle.dump({}, f)
    
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