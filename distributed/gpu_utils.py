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
        
        # Check for Apple MPS (Metal Performance Shaders)
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Set device options
        self.devices = []
        if self.num_gpus > 0:
            self.devices = [f'cuda:{i}' for i in range(self.num_gpus)]
        elif self.has_mps:
            self.devices = ['mps']
        else:
            self.devices = ['cpu']
            
        # Track device memory info if available
        self.device_memory = {}
        if self.num_gpus > 0:
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                self.device_memory[i] = {
                    'name': props.name,
                    'total_memory': props.total_memory,
                    'compute_capability': f"{props.major}.{props.minor}"
                }
                
        # Print available devices
        if self.num_gpus > 0:
            print(f"Found {self.num_gpus} CUDA GPU(s):")
            for i, device in enumerate(self.devices):
                if i < self.num_gpus:
                    mem_gb = self.device_memory[i]['total_memory'] / (1024**3)
                    print(f"  {device}: {self.device_memory[i]['name']} "
                          f"({mem_gb:.1f} GB, CUDA {self.device_memory[i]['compute_capability']})")
        elif self.has_mps:
            print("Found Apple MPS (Metal Performance Shaders)")
        else:
            print("No GPUs found, using CPU")
    
    def assign_device(self, worker_id: int, num_workers: int) -> str:
        """
        Assign a device to a worker based on ID
        
        Args:
            worker_id: ID of the worker process
            num_workers: Total number of workers
            
        Returns:
            Device string ('cpu', 'cuda:0', 'mps', etc.)
        """
        if not self.devices or (len(self.devices) == 1 and self.devices[0] == 'cpu'):
            return 'cpu'
        
        # Distribute workers evenly across available devices
        device_idx = worker_id % len(self.devices)
        return self.devices[device_idx]
    
    def get_best_device(self) -> str:
        """
        Get the best available device for training
        
        Returns:
            Best device string
        """
        if self.num_gpus > 0:
            return 'cuda:0'  # Use the first CUDA device
        elif self.has_mps:
            return 'mps'  # Use Apple MPS
        else:
            return 'cpu'  # Fallback to CPU


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