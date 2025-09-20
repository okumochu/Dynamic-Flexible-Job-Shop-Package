#!/usr/bin/env python3

"""
Memory Management Utilities for PyTorch GPU Training

This module provides utilities for managing GPU memory during training,
including safe cache clearing and memory monitoring.
"""

import torch
import gc
import psutil
import os
from typing import Dict, List, Optional


def get_gpu_memory_info(device_id: int = 0) -> Dict[str, float]:
    """
    Get detailed GPU memory information.
    
    Args:
        device_id: GPU device ID to check
        
    Returns:
        Dictionary with memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    device = torch.device(f"cuda:{device_id}")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)
    free_memory = total_memory - allocated_memory
    
    return {
        "total": total_memory / 1e9,
        "allocated": allocated_memory / 1e9,
        "cached": cached_memory / 1e9,
        "free": free_memory / 1e9,
        "utilization": (allocated_memory / total_memory) * 100
    }


def get_system_memory_info() -> Dict[str, float]:
    """
    Get system RAM memory information.
    
    Returns:
        Dictionary with RAM statistics in GB
    """
    memory = psutil.virtual_memory()
    return {
        "total": memory.total / 1e9,
        "available": memory.available / 1e9,
        "used": memory.used / 1e9,
        "utilization": memory.percent
    }


def clear_gpu_cache_safe(device_id: int = 0, force: bool = False) -> Dict[str, float]:
    """
    Safely clear GPU cache without affecting other processes.
    
    Args:
        device_id: GPU device ID to clear
        force: If True, use more aggressive clearing
        
    Returns:
        Memory info before and after clearing
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    device = torch.device(f"cuda:{device_id}")
    
    # Get memory info before clearing
    before = get_gpu_memory_info(device_id)
    
    if force:
        # More aggressive clearing
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
    else:
        # Standard clearing
        torch.cuda.empty_cache()
    
    # Get memory info after clearing
    after = get_gpu_memory_info(device_id)
    
    return {
        "before": before,
        "after": after,
        "freed": before["allocated"] - after["allocated"]
    }


def check_gpu_availability(required_memory_gb: float = 2.0, device_id: int = 0) -> bool:
    """
    Check if GPU has enough free memory for training.
    
    Args:
        required_memory_gb: Required free memory in GB
        device_id: GPU device ID to check
        
    Returns:
        True if enough memory is available
    """
    memory_info = get_gpu_memory_info(device_id)
    return memory_info.get("free", 0) >= required_memory_gb


def find_best_gpu(min_memory_gb: float = 2.0) -> Optional[int]:
    """
    Find the GPU with the most available memory.
    
    Args:
        min_memory_gb: Minimum required memory in GB
        
    Returns:
        GPU device ID with most free memory, or None if none available
    """
    if not torch.cuda.is_available():
        return None
    
    best_device = None
    max_free_memory = 0
    
    for device_id in range(torch.cuda.device_count()):
        memory_info = get_gpu_memory_info(device_id)
        free_memory = memory_info.get("free", 0)
        
        if free_memory >= min_memory_gb and free_memory > max_free_memory:
            max_free_memory = free_memory
            best_device = device_id
    
    return best_device


def monitor_memory_usage(func, *args, **kwargs):
    """
    Decorator to monitor memory usage during function execution.
    
    Args:
        func: Function to monitor
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result and memory usage statistics
    """
    def wrapper(*args, **kwargs):
        # Get initial memory state
        initial_gpu = get_gpu_memory_info()
        initial_ram = get_system_memory_info()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory state
        final_gpu = get_gpu_memory_info()
        final_ram = get_system_memory_info()
        
        # Calculate memory usage
        memory_stats = {
            "gpu": {
                "initial": initial_gpu,
                "final": final_gpu,
                "delta": final_gpu["allocated"] - initial_gpu["allocated"]
            },
            "ram": {
                "initial": initial_ram,
                "final": final_ram,
                "delta": final_ram["used"] - initial_ram["used"]
            }
        }
        
        return result, memory_stats
    
    return wrapper


def print_memory_status(device_id: int = 0):
    """
    Print current memory status for debugging.
    
    Args:
        device_id: GPU device ID to check
    """
    gpu_info = get_gpu_memory_info(device_id)
    ram_info = get_system_memory_info()
    
    print("=" * 50)
    print("MEMORY STATUS")
    print("=" * 50)
    print(f"GPU {device_id}:")
    print(f"  Total: {gpu_info['total']:.1f} GB")
    print(f"  Allocated: {gpu_info['allocated']:.1f} GB")
    print(f"  Cached: {gpu_info['cached']:.1f} GB")
    print(f"  Free: {gpu_info['free']:.1f} GB")
    print(f"  Utilization: {gpu_info['utilization']:.1f}%")
    print()
    print("System RAM:")
    print(f"  Total: {ram_info['total']:.1f} GB")
    print(f"  Used: {ram_info['used']:.1f} GB")
    print(f"  Available: {ram_info['available']:.1f} GB")
    print(f"  Utilization: {ram_info['utilization']:.1f}%")
    print("=" * 50)


def safe_torch_cleanup():
    """
    Perform safe PyTorch cleanup that won't affect other processes.
    """
    # Clear Python garbage collection
    gc.collect()
    
    # Clear PyTorch cache (this only affects the current process)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection again
    gc.collect()


if __name__ == "__main__":
    # Test the memory utilities
    print("Testing memory utilities...")
    print_memory_status()
    
    print("\nClearing GPU cache...")
    result = clear_gpu_cache_safe(force=True)
    print(f"Freed {result['freed']:.2f} GB of GPU memory")
    
    print("\nFinding best GPU...")
    best_gpu = find_best_gpu(min_memory_gb=1.0)
    print(f"Best GPU: {best_gpu}")
