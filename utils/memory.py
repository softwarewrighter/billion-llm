"""Memory tracking utilities for billion-llm."""

import functools
import gc
import time
from contextlib import contextmanager

import torch


def get_memory_usage() -> dict:
    """
    Get current memory usage for CPU and GPU.

    Returns:
        Dictionary with memory stats in GB
    """
    import psutil

    stats = {
        "cpu_used_gb": psutil.Process().memory_info().rss / (1024**3),
        "cpu_available_gb": psutil.virtual_memory().available / (1024**3),
    }

    if torch.cuda.is_available():
        stats["gpu_used_gb"] = torch.cuda.memory_allocated() / (1024**3)
        stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        stats["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (
            1024**3
        )

    if torch.backends.mps.is_available():
        # MPS doesn't have detailed memory tracking
        stats["mps_available"] = True

    return stats


def clear_memory():
    """Clear GPU and CPU caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@contextmanager
def track_memory():
    """
    Context manager to track peak memory usage.

    Usage:
        with track_memory() as tracker:
            # do work
        print(f"Peak memory: {tracker['peak_gpu_gb']:.2f} GB")
    """
    clear_memory()

    stats = {"start_time": time.time()}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        stats["start_gpu_gb"] = torch.cuda.memory_allocated() / (1024**3)

    try:
        yield stats
    finally:
        stats["end_time"] = time.time()
        stats["elapsed_seconds"] = stats["end_time"] - stats["start_time"]

        if torch.cuda.is_available():
            stats["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            stats["end_gpu_gb"] = torch.cuda.memory_allocated() / (1024**3)


def memory_tracker(func):
    """
    Decorator to track peak memory during function execution.

    Usage:
        @memory_tracker
        def my_function():
            # do work

        result, memory_stats = my_function()
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with track_memory() as stats:
            result = func(*args, **kwargs)
        return result, stats

    return wrapper


def estimate_model_memory(num_params: int, dtype: str = "fp16") -> float:
    """
    Estimate memory required for a model.

    Args:
        num_params: Number of parameters
        dtype: Data type ("fp32", "fp16", "int8", "int4")

    Returns:
        Estimated memory in GB
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }

    param_memory = num_params * bytes_per_param.get(dtype, 2)

    # Add ~20% overhead for optimizer states, activations, etc.
    total_memory = param_memory * 1.2

    return total_memory / (1024**3)
