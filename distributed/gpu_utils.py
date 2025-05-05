import os
import torch
import numpy as np

class DeviceManager:
    """Manages GPU resources and device assignments"""
    
    def __init__(self):
        # Check for forced accelerator from environment
        self.force_cuda = os.environ.get("FORCE_CUDA", "0") == "1"
        self.force_mps = os.environ.get("FORCE_MPS", "0") == "1"
        self.force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
        
        # Get available hardware
        self.num_gpus = torch.cuda.device_count()
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Configure devices based on what's available and what's forced
        if self.force_cuda:
            if self.num_gpus == 0:
                raise RuntimeError("CUDA accelerator forced but no CUDA devices available")
            self.devices = [f'cuda:{i}' for i in range(self.num_gpus)]
            print(f"Using CUDA (forced) with {self.num_gpus} GPUs")
            
        elif self.force_mps:
            if not self.has_mps:
                raise RuntimeError("MPS accelerator forced but MPS not available")
            self.devices = ['mps']
            print("Using MPS (forced)")
            
        elif self.force_cpu:
            self.devices = ['cpu']
            print("Using CPU (forced)")
            
        # Auto-select based on what's available
        elif self.num_gpus > 0:
            self.devices = [f'cuda:{i}' for i in range(self.num_gpus)]
            print(f"Using CUDA with {self.num_gpus} GPUs")
            
        elif self.has_mps:
            self.devices = ['mps']
            print("Using MPS")
            
        else:
            self.devices = ['cpu']
            print("Using CPU")
    
    def assign_device(self, worker_id: int, num_workers: int) -> str:
        """Assign a device to a worker based on availability and forced settings"""
        
        # If using CPU (forced or default when no accelerators available)
        if self.force_cpu or (not self.force_cuda and not self.force_mps and len(self.devices) == 1 and self.devices[0] == 'cpu'):
            return 'cpu'
        
        # If using MPS (forced or auto-selected)
        if self.force_mps or (not self.force_cuda and self.has_mps):
            return 'mps'
        
        # If using CUDA (forced or auto-selected)
        if self.num_gpus > 0:
            gpu_index = worker_id % self.num_gpus
            return f'cuda:{gpu_index}'
            
        # Fallback to CPU if no other options available
        return 'cpu'
    
    def get_best_device(self) -> str:
        """Get the best available device based on availability and forced settings"""
        
        # Handle forced device preferences
        if self.force_cuda:
            if self.num_gpus > 0:
                return 'cuda:0'
            else:
                raise RuntimeError("CUDA forced but no CUDA devices available")
                
        if self.force_mps:
            if self.has_mps:
                return 'mps'
            else:
                raise RuntimeError("MPS forced but not available")
                
        if self.force_cpu:
            return 'cpu'
            
        # Auto-select based on availability
        if self.num_gpus > 0:
            return 'cuda:0'
        elif self.has_mps:
            return 'mps'
        else:
            return 'cpu'  # Default to CPU if no accelerators available


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)