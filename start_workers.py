#!/usr/bin/env python
"""
Script to start multiple 2048-Zero workers

This script automatically starts workers according to the following policy:
- First 4*k workers use CUDA GPUs in round-robin fashion (where k is number of GPUs)
- Remaining workers use CPU (one CPU core per worker)
- Total workers count equals the number of CPUs in the system
"""

import os
import sys
import time
import torch
import argparse
import subprocess
import multiprocessing
from typing import List

def get_gpu_count() -> int:
    """Get the number of available CUDA GPUs"""
    try:
        return torch.cuda.device_count()
    except:
        return 0

def get_cpu_count() -> int:
    """Get the number of available CPU cores"""
    try:
        return multiprocessing.cpu_count()
    except:
        return 1

def start_worker(worker_id: int, args, device: str) -> subprocess.Popen:
    """Start a single worker process with specified device"""
    # Base command with common arguments
    cmd = [
        sys.executable, "run_worker.py",
        "--server-url", args.server_url,
        "--auth-token", args.auth_token,
        "--batch-size", str(args.batch_size),
        "--device", device
    ]
    
    # Create and start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if not args.verbose else None,
        stderr=subprocess.PIPE if not args.verbose else None,
        text=True,
        env=os.environ.copy()
    )
    
    # Log worker start
    if args.verbose:
        print(f"Started worker {worker_id} on {device}")
    else:
        print(f"Started worker {worker_id} on {device} (logs hidden, use --verbose to show)")
    
    return process

def main():
    """Parse arguments and start workers"""
    parser = argparse.ArgumentParser(description="Start multiple 2048-Zero workers")
    
    # Server settings
    parser.add_argument("--server-url", type=str, default="http://localhost:8000",
                        help="URL of the trainer server")
    parser.add_argument("--auth-token", type=str, default="2048-zero-token",
                        help="Authentication token")
    
    # Worker settings
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of games to collect before uploading")
    parser.add_argument("--verbose", action="store_true",
                        help="Show worker output in console")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Maximum number of workers to start (default: number of CPUs)")
    parser.add_argument("--gpu-only", action="store_true",
                        help="Only use GPU workers (no CPU workers)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Only use CPU workers (no GPU workers)")
    parser.add_argument("--workers-per-gpu", type=int, default=4,
                        help="Number of workers per GPU (default: 4)")
    
    args = parser.parse_args()
    
    # Count available resources
    gpu_count = 0 if args.cpu_only else get_gpu_count()
    cpu_count = get_cpu_count()
    workers_per_gpu = args.workers_per_gpu
    
    # Calculate worker counts
    gpu_workers = gpu_count * workers_per_gpu if gpu_count > 0 else 0
    max_workers = args.max_workers or cpu_count
    
    if args.gpu_only:
        cpu_workers = 0
        total_workers = min(gpu_workers, max_workers)
    else:
        # CPU workers fill the remaining slots
        cpu_workers = max(0, min(cpu_count, max_workers - gpu_workers))
        total_workers = min(max_workers, gpu_workers + cpu_workers)
    
    print(f"System resources: {cpu_count} CPUs, {gpu_count} GPUs")
    print(f"Starting {total_workers} workers: {gpu_workers} on GPU, {cpu_workers} on CPU")
    
    # Track processes
    processes: List[subprocess.Popen] = []
    
    try:
        # Start GPU workers first
        for i in range(gpu_workers):
            gpu_id = i % gpu_count  # Round-robin assignment
            device = f"cuda:{gpu_id}"
            process = start_worker(i+1, args, device)
            processes.append(process)
            time.sleep(0.1)  # Small delay to avoid launch conflicts
        
        # Start CPU workers
        for i in range(cpu_workers):
            worker_id = gpu_workers + i + 1
            process = start_worker(worker_id, args, "cpu")
            processes.append(process)
            time.sleep(0.1)  # Small delay to avoid launch conflicts
        
        print(f"All {len(processes)} workers started successfully")
        print("Press Ctrl+C to stop all workers")
        
        # Wait for processes to complete (or until interrupted)
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        print("\nStopping all workers...")
        
        # Terminate all processes
        for process in processes:
            try:
                process.terminate()
            except:
                pass
        
        # Wait for processes to terminate
        for process in processes:
            try:
                process.wait(timeout=5)
            except:
                pass
        
        print("All workers stopped")

if __name__ == "__main__":
    main()