# Distributed AlphaZero training components
import os
import torch
import multiprocessing as mp

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# Basic environment variables
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '4'

# Import components
from .gpu_utils import DeviceManager
from .cuda_utils import safe_cuda_initialization, setup_cuda_for_worker
from .parallel_selfplay import MultiprocessSelfPlayWorker
from .parallel_trainer import ParallelZeroTrainer
# Import reward functions from zero module instead of local functions
from zero.reward_functions import score_reward_func, max_tile_reward_func, hybrid_reward_func

# Initialize CUDA - use minimal initialization to avoid deadlocks
if torch.cuda.is_available():
    # Just check device count but don't initialize contexts to avoid potential deadlocks
    # Let each worker process initialize its own device context as needed
    num_gpus = torch.cuda.device_count()
    # Set critical environment variables
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['NCCL_P2P_DISABLE'] = '1'  # Can help avoid deadlocks
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

__all__ = [
    "DeviceManager",
    "MultiprocessSelfPlayWorker",
    "ParallelZeroTrainer",
    "safe_cuda_initialization",
    "setup_cuda_for_worker",
    # Renamed reward functions
    "score_reward_func",
    "max_tile_reward_func", 
    "hybrid_reward_func"
]