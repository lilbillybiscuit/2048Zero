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

# Initialize CUDA - fail if any issues occur
if torch.cuda.is_available():
    safe_cuda_initialization()

__all__ = [
    "DeviceManager",
    "MultiprocessSelfPlayWorker",
    "ParallelZeroTrainer",
    "safe_cuda_initialization",
    "setup_cuda_for_worker"
]