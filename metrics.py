import time

import psutil
from typing import Any
# --- METRICS: Import NVML for GPU stats ---
try:
    from pynvml import *

    NVML_INITIALIZED = False
except ImportError:
    NVML_INITIALIZED = False
# --- END METRICS ---
from vllm.third_party.pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, \
    NVMLError, nvmlInit

# --- METRICS: Helper class and functions ---
class MetricsTracker:
    """A simple class to hold all performance metrics for a single request."""

    def __init__(self, logger: Any):
        self.logger = logger
        self.start_time = time.monotonic()
        self.end_time = None
        self.step_latencies = {}
        self.step_token_counts = {}
        self.resource_snapshots = {}
        self.agent_step_count = 0
        self.task_completed = False
        self.task_completion_reason = "Unknown"

    def get_resource_usage(self) -> dict:
        """Captures a snapshot of current CPU, RAM, and GPU VRAM usage."""
        global NVML_INITIALIZED
        usage = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "gpu_percent": None,
            "vram_percent": None
        }
        if NVML_INITIALIZED:
            try:
                handle = nvmlDeviceGetHandleByIndex(0)
                gpu_util = nvmlDeviceGetUtilizationRates(handle)
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                usage["gpu_percent"] = gpu_util.gpu
                usage["vram_percent"] = (mem_info.used / mem_info.total) * 100
            except NVMLError as e:
                self.logger.warning(f"Could not get GPU stats: {e}")
        return usage

    def log_final_metrics(self):
        """Calculates and logs all final metrics at the end of the request."""
        self.end_time = time.monotonic()
        end_to_end_latency = self.end_time - self.start_time
        total_tokens_generated = sum(self.step_token_counts.values())

        # --- FORMAT AND LOG METRICS ---
        self.logger.info("--- AGENT EXECUTION METRICS ---")

        # Latency Metrics
        self.logger.info(f"[Latency] End-to-End: {end_to_end_latency:.2f} seconds")
        for step, latency in self.step_latencies.items():
            self.logger.info(f"[Latency] Step '{step}': {latency:.2f} seconds")

        # Throughput Metrics
        self.logger.info(f"[Throughput] Total Tokens Generated: {total_tokens_generated}")
        for step, latency in self.step_latencies.items():
            tokens = self.step_token_counts.get(step, 0)
            if latency > 0 and tokens > 0:
                throughput = tokens / latency
                self.logger.info(f"[Throughput] Step '{step}': {throughput:.2f} tokens/sec")

        # Resource Usage Metrics
        for step, usage in self.resource_snapshots.items():
            self.logger.info(
                f"[Resource Usage] After Step '{step}': "
                f"CPU: {usage['cpu_percent']:.1f}%, RAM: {usage['ram_percent']:.1f}%, "
                f"GPU: {usage.get('gpu_percent', 'N/A')}%, VRAM: {usage.get('vram_percent', 'N/A'):.1f}%"
            )

        # Agent & Task Metrics
        self.logger.info(f"[Agent] Step Count: {self.agent_step_count}")
        self.logger.info(f"[Task] Completion Status: {'Success' if self.task_completed else 'Failure'}")
        self.logger.info(f"[Task] Completion Reason: {self.task_completion_reason}")
        self.logger.info("---------------------------------")


def initialize_nvml(logger: Any):
    """Initializes the NVML library for GPU monitoring."""
    global NVML_INITIALIZED
    if not NVML_INITIALIZED:
        try:
            nvmlInit()
            NVML_INITIALIZED = True
            logger.info("NVML Initialized for GPU monitoring.")
        except NVMLError as e:
            logger.warning(f"Could not initialize NVML for GPU monitoring. GPU stats will not be available. Error: {e}")