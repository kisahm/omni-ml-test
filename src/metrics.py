"""
Performance Metrics Collection
Tracks GPU usage, latency, throughput, and system metrics
"""
import time
import psutil
import torch
from typing import Dict, Optional
from contextlib import contextmanager

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class MetricsCollector:
    """Collects and tracks performance metrics"""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.nvml_initialized = False

        if self.gpu_available and NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                print(f"Warning: Could not initialize NVML: {e}")

    def get_gpu_metrics(self) -> Dict:
        """Get current GPU metrics"""
        metrics = {
            'gpu_available': self.gpu_available,
            'gpu_memory_allocated_mb': 0,
            'gpu_memory_reserved_mb': 0,
            'gpu_utilization_percent': 0,
            'gpu_memory_used_mb': 0,
            'gpu_memory_total_mb': 0,
            'gpu_temperature_c': 0,
        }

        if not self.gpu_available:
            return metrics

        # PyTorch memory stats
        metrics['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        metrics['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024

        # NVML stats (more detailed)
        if self.nvml_initialized:
            try:
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                metrics['gpu_utilization_percent'] = util.gpu

                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                metrics['gpu_memory_used_mb'] = mem_info.used / 1024 / 1024
                metrics['gpu_memory_total_mb'] = mem_info.total / 1024 / 1024

                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics['gpu_temperature_c'] = temp

            except Exception as e:
                print(f"Warning: Could not get NVML metrics: {e}")

        return metrics

    def get_system_metrics(self) -> Dict:
        """Get system-level metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
        }

    def get_all_metrics(self) -> Dict:
        """Get all metrics"""
        return {
            **self.get_gpu_metrics(),
            **self.get_system_metrics(),
            'timestamp': time.time(),
        }

    @contextmanager
    def measure_time(self, operation_name: str = "operation"):
        """Context manager to measure operation time"""
        start_time = time.time()
        start_metrics = self.get_all_metrics()

        yield

        end_time = time.time()
        end_metrics = self.get_all_metrics()

        duration = end_time - start_time
        print(f"\n=== {operation_name} Metrics ===")
        print(f"Duration: {duration:.4f}s")
        print(f"GPU Memory Change: {end_metrics['gpu_memory_allocated_mb'] - start_metrics['gpu_memory_allocated_mb']:.2f} MB")
        print(f"GPU Utilization: {end_metrics['gpu_utilization_percent']}%")

    def __del__(self):
        """Cleanup NVML"""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class LatencyTracker:
    """Tracks latency statistics"""

    def __init__(self):
        self.latencies = []

    def record(self, latency_ms: float):
        """Record a latency measurement"""
        self.latencies.append(latency_ms)

    def get_stats(self) -> Dict:
        """Get latency statistics"""
        if not self.latencies:
            return {}

        latencies = sorted(self.latencies)
        n = len(latencies)

        return {
            'count': n,
            'mean_ms': sum(latencies) / n,
            'min_ms': latencies[0],
            'max_ms': latencies[-1],
            'p50_ms': latencies[n // 2],
            'p95_ms': latencies[int(n * 0.95)],
            'p99_ms': latencies[int(n * 0.99)],
        }

    def reset(self):
        """Reset collected latencies"""
        self.latencies = []
