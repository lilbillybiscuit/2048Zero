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

def safe_cuda_initialization(device_idx: Optional[int] = None, allow_cpu_fallback: bool = False) -> bool:
    """
    Initialize CUDA in a safe manner for multiprocessing
    
    Args:
        device_idx: Specific CUDA device to initialize, or None for all devices
        allow_cpu_fallback: If True, allows silent fallback to CPU. If False, raises exceptions
        
    Returns:
        bool: True if initialization was successful
        
    Raises:
        RuntimeError: If CUDA initialization fails and allow_cpu_fallback is False
    """
    # Verify CUDA is available
    if not torch.cuda.is_available():
        error_msg = "CUDA requested but not available on this system"
        if allow_cpu_fallback:
            print(f"Error: {error_msg}. Falling back to CPU.")
            return False
        else:
            raise RuntimeError(error_msg)
        
    try:
        with cuda_operation_timeout(CUDA_INIT_TIMEOUT):
            # Set environment variables for better stability
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Don't block in production
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Consistent device numbering
            
            # Limit threads to avoid contention
            os.environ['OMP_NUM_THREADS'] = str(MAX_THREADS_PER_GPU)
            torch.set_num_threads(MAX_THREADS_PER_GPU)
            
            if device_idx is not None:
                # Initialize single device
                if device_idx >= torch.cuda.device_count():
                    error_msg = f"CUDA device {device_idx} not available. System has {torch.cuda.device_count()} GPU(s)"
                    if allow_cpu_fallback:
                        print(f"Error: {error_msg}. Falling back to CPU.")
                        return False
                    else:
                        raise RuntimeError(error_msg)
                    
                # Set device and initialize context
                torch.cuda.set_device(device_idx)
                device = f'cuda:{device_idx}'
                
                # Initialize CUDA context with a small tensor
                # Use a scalar tensor (size 1) to avoid conversion errors
                dummy = torch.zeros(1, device=device)
                value = dummy.item()  # Force synchronization
                del dummy
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                
                print(f"CUDA device {device_idx} initialized successfully.")
            else:
                # Initialize all devices
                print(f"Initializing {torch.cuda.device_count()} CUDA devices...")
                for i in range(torch.cuda.device_count()):
                    # Set device and create tensor to initialize context
                    torch.cuda.set_device(i)
                    device = f'cuda:{i}'
                    
                    # Use a scalar tensor (size 1) to avoid conversion errors
                    dummy = torch.zeros(1, device=device)
                    value = dummy.item()  # Force synchronization
                    del dummy
                    torch.cuda.empty_cache()
                    
                    # Get device info
                    device_name = torch.cuda.get_device_name(i)
                    print(f"  CUDA device {i}: {device_name} initialized")
                
                # Synchronize all devices
                torch.cuda.synchronize()
                
            return True
    except TimeoutException:
        error_msg = f"CUDA initialization timed out after {CUDA_INIT_TIMEOUT} seconds"
        if allow_cpu_fallback:
            print(f"Error: {error_msg}. Falling back to CPU.")
            return False
        else:
            print(f"Error: {error_msg}")
            raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"CUDA initialization error: {e}"
        if allow_cpu_fallback:
            print(f"Error: {error_msg}. Falling back to CPU.")
            return False
        else:
            print(f"Error: {error_msg}")
            raise RuntimeError(error_msg)

def setup_cuda_for_worker(device_idx: int, allow_cpu_fallback: bool = False) -> bool:
    """
    Configure the CUDA environment for a worker process.
    Should be called at the start of each worker process.
    
    Args:
        device_idx: The CUDA device index for this worker
        allow_cpu_fallback: If True, allows silent fallback to CPU. If False, raises exceptions
        
    Returns:
        bool: True if successful, False if fell back to CPU
        
    Raises:
        RuntimeError: If CUDA initialization fails and allow_cpu_fallback is False
    """
    # Verify CUDA is available
    if not torch.cuda.is_available():
        error_msg = "CUDA requested but not available on this system"
        if allow_cpu_fallback:
            print(f"Error: {error_msg}. Falling back to CPU.")
            return False
        else:
            raise RuntimeError(error_msg)
            
    # Verify device index is valid
    if device_idx >= torch.cuda.device_count():
        error_msg = f"CUDA device {device_idx} not available. System has {torch.cuda.device_count()} GPU(s)"
        if allow_cpu_fallback:
            print(f"Error: {error_msg}. Falling back to CPU.")
            return False
        else:
            raise RuntimeError(error_msg)
            
    # Set thread limits to avoid resource contention
    os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS_PER_GPU)
    torch.set_num_threads(MAX_THREADS_PER_GPU)
    
    # Disable P2P access which can sometimes cause issues
    os.environ["NCCL_P2P_DISABLE"] = "1"
    
    try:
        # Set device before any CUDA operations
        torch.cuda.set_device(device_idx)
        
        # Initialize context with timeout protection
        with cuda_operation_timeout(CUDA_INIT_TIMEOUT):
            # Create a small tensor to initialize context
            # This is critical to prevent deadlocks on Linux multi-GPU systems
            dummy = torch.zeros(1, device=f"cuda:{device_idx}")  # Use a scalar tensor (size 1)
            value = dummy.item()  # Force synchronization (works with scalar)
            del dummy
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device_idx)
            
        print(f"CUDA context successfully initialized for device {device_idx}")
        return True
        
    except (TimeoutException, Exception) as e:
        error_msg = f"CUDA setup error for device {device_idx}: {e}"
        if allow_cpu_fallback:
            print(f"Error: {error_msg}. Falling back to CPU.")
            return False
        else:
            print(f"Error: {error_msg}")
            raise RuntimeError(error_msg)

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