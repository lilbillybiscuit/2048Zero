# Distributed AlphaZero training components
import os
import torch
import multiprocessing as mp

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# This needs to be done before importing anything that uses multiprocessing
try:
    mp.set_start_method('spawn')
    print("Multiprocessing start method set to 'spawn'")
except RuntimeError:
    # Already set, that's fine
    pass

# Set environment variables for better CUDA behavior
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Consistent device numbering
os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads

# Now import components
from .gpu_utils import DeviceManager
from .cuda_utils import safe_cuda_initialization, setup_cuda_for_worker
from .parallel_selfplay import MultiprocessSelfPlayWorker
from .parallel_trainer import ParallelZeroTrainer

# Initialize CUDA (will raise an error if it fails - no silent fallbacks)
if torch.cuda.is_available():
    print(f"Initializing CUDA - {torch.cuda.device_count()} GPU(s) detected")
    
    # Use False for allow_cpu_fallback to force errors to be raised
    safe_cuda_initialization(allow_cpu_fallback=False)
    
    # Print GPU details
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available, running in CPU mode")

__all__ = [
    "DeviceManager",
    "MultiprocessSelfPlayWorker",
    "ParallelZeroTrainer",
    "safe_cuda_initialization",
    "setup_cuda_for_worker"
]