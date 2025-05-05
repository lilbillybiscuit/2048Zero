"""
GPU and device management utilities for distributed training
"""

import os
import torch
import numpy as np

class DeviceManager:
    """Manages GPU resources and device assignments"""
    
    def __init__(self):
        """Initialize device manager and detect available devices"""        
        # Detect available CUDA GPUs
        self.num_gpus = torch.cuda.device_count()
        
        # Check for Apple MPS
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Set device options
        if self.num_gpus > 0:
            self.devices = [f'cuda:{i}' for i in range(self.num_gpus)]
        elif self.has_mps:
            self.devices = ['mps']
        else:
            self.devices = ['cpu']
            
        # Minimal output with device info
        if self.num_gpus > 0:
            print(f"GPUs: {self.num_gpus}")
    
    def assign_device(self, worker_id: int, num_workers: int) -> str:
        """
        Assign a device to a worker based on ID
        
        Args:
            worker_id: ID of the worker process
            num_workers: Total number of workers
            
        Returns:
            Device string ('cpu', 'cuda:0', 'mps', etc.)
            
        Raises:
            RuntimeError: If CUDA is requested but no GPUs are available
        """
        # Fail if no GPUs are available but were requested
        if self.num_gpus == 0:
            raise RuntimeError("CUDA requested but no GPUs are available")
        
        # Round-robin assignment across available GPUs
        if self.num_gpus > 0:
            gpu_index = worker_id % self.num_gpus
            return f'cuda:{gpu_index}'
        
        # Apple MPS fallback
        elif self.has_mps:
            return 'mps'
        
        # Should never reach here due to earlier check
        raise RuntimeError("GPU requested but none available")
    
    def get_best_device(self) -> str:
        """
        Get the best available device for training
        
        Returns:
            Best device string
            
        Raises:
            RuntimeError: If no GPUs are available
        """
        if self.num_gpus > 0:
            return 'cuda:0'
        elif self.has_mps:
            return 'mps'
        else:
            raise RuntimeError("GPU requested but none available")


def seed_everything(seed: int):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)