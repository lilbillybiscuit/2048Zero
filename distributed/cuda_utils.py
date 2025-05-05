"""
CUDA utilities for handling multiprocessing with GPU acceleration
"""

import os
import time
import torch
import threading
import multiprocessing as mp
import signal
from contextlib import contextmanager
from typing import Optional, Callable, Any, Dict, Union

# Constants
CUDA_INIT_TIMEOUT = 30  # seconds
MAX_THREADS_PER_GPU = 4
TENSOR_INIT_SIZE = 1024  # Small tensor size for initialization (1KB)

class TimeoutException(Exception):
    """Exception raised when a CUDA operation times out"""
    pass

@contextmanager
def cuda_operation_timeout(seconds: int = 30):
    """
    Context manager that raises TimeoutException if operation takes longer than specified seconds
    
    Usage:
        try:
            with cuda_operation_timeout(seconds=10):
                # CUDA operation that might hang
                model = model.to(device)
        except TimeoutException:
            print("Operation timed out, falling back to CPU")
            device = 'cpu'
            model = model.to(device)
    """
    def timeout_handler(signum, frame):
        raise TimeoutException(f"CUDA operation timed out after {seconds} seconds")
    
    # Only set timeout on Unix systems where signal.SIGALRM is available
    if hasattr(signal, 'SIGALRM'):
        # Set the timeout handler
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        # Set the alarm
        signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Cancel the alarm and restore the original handler
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)

def get_cuda_memory_usage(device_idx: Optional[int] = None) -> Dict[str, float]:
    """
    Get detailed GPU memory usage information
    
    Args:
        device_idx: CUDA device index, if None uses current device
        
    Returns:
        Dict with memory usage details in MB
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "free_mb": 0, "total_mb": 0}
    
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    
    try:
        # Get memory statistics
        allocated = torch.cuda.memory_allocated(device_idx) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device_idx) / 1024 / 1024
        
        # Get total memory if possible (requires pynvml)
        total = 0
        free = 0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = meminfo.total / 1024 / 1024
            free = meminfo.free / 1024 / 1024
            pynvml.nvmlShutdown()
        except (ImportError, Exception):
            # If pynvml is not available, just return what we have
            pass
            
        return {
            "allocated_mb": allocated, 
            "reserved_mb": reserved,
            "free_mb": free,
            "total_mb": total
        }
    except Exception as e:
        print(f"Error getting CUDA memory: {e}")
        return {"allocated_mb": 0, "reserved_mb": 0, "free_mb": 0, "total_mb": 0}

def safe_cuda_initialization(device_idx: Optional[int] = None) -> None:
    """
    Initialize CUDA in a safe manner for multiprocessing.
    Will fail if any CUDA operation fails.
    
    Args:
        device_idx: Specific CUDA device to initialize, or None for all devices
        
    Raises:
        RuntimeError: If any CUDA operation fails
    """
    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    
    # Set basic environment variables
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['OMP_NUM_THREADS'] = str(MAX_THREADS_PER_GPU)
    
    try:
        if device_idx is not None:
            # Initialize single device
            if device_idx >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device {device_idx} not available. System has {torch.cuda.device_count()} GPU(s)")
                
            # Set device and initialize context
            torch.cuda.set_device(device_idx)
            
            # Initialize context with a tensor
            with cuda_operation_timeout(CUDA_INIT_TIMEOUT):
                dummy = torch.zeros(1, device=f'cuda:{device_idx}')
                value = dummy.item()  # Force synchronization
                del dummy
        else:
            # Initialize all devices
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                
                with cuda_operation_timeout(CUDA_INIT_TIMEOUT):
                    dummy = torch.zeros(1, device=f'cuda:{i}')
                    value = dummy.item()  # Force synchronization
                    del dummy
    except Exception as e:
        raise RuntimeError(f"CUDA initialization failed: {e}")

def setup_cuda_for_worker(device_idx: int) -> None:
    """
    Configure the CUDA environment for a worker process.
    Will fail if any CUDA operation fails.
    
    Args:
        device_idx: The CUDA device index for this worker
        
    Raises:
        RuntimeError: If any CUDA operation fails
    """
    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
            
    # Verify device index is valid
    if device_idx >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device_idx} not available. System has {torch.cuda.device_count()} GPU(s)")
    
    # Set basic environment variables
    os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS_PER_GPU)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    
    # Set device and initialize context
    torch.cuda.set_device(device_idx)
    
    # Create a small tensor to initialize context
    try:
        with cuda_operation_timeout(CUDA_INIT_TIMEOUT):
            dummy = torch.zeros(1, device=f"cuda:{device_idx}")
            value = dummy.item()  # Force synchronization
            del dummy
    except Exception as e:
        raise RuntimeError(f"CUDA initialization failed: {e}")

def with_cuda_retry(max_retries: int = 3, retry_delay: float = 1.0) -> Callable:
    """
    Decorator for functions that use CUDA to retry on certain errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Callable: Wrapped function with retry logic
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            last_error = None
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except (torch.cuda.CudaError, RuntimeError) as e:
                    # Only retry for specific CUDA errors
                    if "CUDA" not in str(e) or retries >= max_retries:
                        raise
                    
                    retries += 1
                    last_error = e
                    print(f"CUDA error: {str(e)}. Retry {retries}/{max_retries}...")
                    torch.cuda.empty_cache()
                    time.sleep(retry_delay)
            
            # If we get here, all retries failed
            raise RuntimeError(f"Failed after {max_retries} retries: {last_error}")
        return wrapper
    return decorator

def init_worker_process():
    """
    Initialize a worker process for CUDA multiprocessing
    
    Call this at the start of each worker process function
    """
    # Set OpenMP thread limit to avoid contention
    os.environ['OMP_NUM_THREADS'] = str(MAX_THREADS_PER_GPU)
    
    # Set PyTorch multiprocessing method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, that's fine
        pass
    
    # Set thread limits for PyTorch
    torch.set_num_threads(MAX_THREADS_PER_GPU)
    
    # Print initialization confirmation
    print(f"Worker process initialized with {MAX_THREADS_PER_GPU} threads")