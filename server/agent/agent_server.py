import logging
import uvicorn
import time
import psutil
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from vllm.third_party.pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, \
    nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, NVMLError

try:
    from pynvml import *
except ImportError:
    pass

from metrics import MetricsTracker
from server.agent.organizer import run_organizer_loop
from server.agent.tools.tool_defs import STATIC_TOOLS
from server.agent.tools.tool_manager import tool_manager
from server.helpers import AgentChatRequest

# --- Configuration ---
AGENT_SERVER_HOST = "0.0.0.0"
AGENT_SERVER_PORT = 8001
MODEL_SERVER_URL = "http://localhost:8000"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("--- Registering Static Tools ---")
for tool in STATIC_TOOLS:
    tool_manager.register(tool)
print("--- Static Tool Registration Complete ---")


def get_all_resource_usage(logger_instance) -> dict:
    """
    Captures a snapshot of current CPU, RAM, and usage across ALL GPUs.
    Returns raw numerical data for calculations.
    """
    usage = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "gpus": []
    }
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            gpu_util = nvmlDeviceGetUtilizationRates(handle)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            usage["gpus"].append({
                "util_percent": float(gpu_util.gpu),
                "vram_percent": (mem_info.used / mem_info.total) * 100.0 if mem_info.total > 0 else 0.0
            })
    except NameError:
        pass
    except NVMLError as e:
        logger_instance.warning(f"Could not get GPU stats: {e}")
    return usage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    logger.info("🚀 Agent Server starting up...")
    # Initialize NVML to check for GPUs
    try:
        nvmlInit()
        logger.info("✅ NVML Initialized for GPU monitoring.")
    except (NameError, NVMLError):
        logger.warning("NVML not available. GPU stats will not be monitored.")
    yield
    logger.info("🌙 Agent Server shutting down.")
    logger.info("Agent Server shutdown complete.")


app = FastAPI(title="Agent Logic Server", lifespan=lifespan)


async def metrics_collecting_stream_generator(
        stream: AsyncGenerator[str, None],
        metrics: MetricsTracker
) -> AsyncGenerator[str, None]:
    """
    A wrapper for the agent's streaming response that captures and logs
    high-level performance metrics for the entire request lifecycle.
    """
    chunk_count = 0
    # Use the local get_all_resource_usage function
    metrics.get_resource_usage = lambda: get_all_resource_usage(logger)
    metrics.resource_snapshots['agent_request_start'] = metrics.get_resource_usage()
    try:
        async for chunk in stream:
            chunk_count += 1
            yield chunk
    finally:
        metrics.end_time = time.monotonic()
        metrics.resource_snapshots['agent_request_end'] = metrics.get_resource_usage()

        e2e_latency = metrics.end_time - metrics.start_time
        throughput = chunk_count / e2e_latency if e2e_latency > 0 else 0

        logger.info("--- AGENT SERVER REQUEST METRICS ---")
        logger.info(f"[Latency] Full Request End-to-End: {e2e_latency:.2f} seconds")
        logger.info(f"[Throughput] Chunks per Second: {throughput:.2f}")
        logger.info(f"[Throughput] Total Chunks Streamed: {chunk_count}")

        start_usage = metrics.resource_snapshots['agent_request_start']
        end_usage = metrics.resource_snapshots['agent_request_end']

        start_gpu_util_str = ", ".join(
            [f"GPU{i}:{gpu['util_percent']:.1f}%" for i, gpu in enumerate(start_usage['gpus'])]) or "N/A"
        start_vram_str = ", ".join(
            [f"GPU{i}:{gpu['vram_percent']:.1f}%" for i, gpu in enumerate(start_usage['gpus'])]) or "N/A"
        logger.info(
            f"[Resources] Start - CPU: {start_usage['cpu_percent']:.1f}%, RAM: {start_usage['ram_percent']:.1f}%, "
            f"Util: {start_gpu_util_str}, VRAM: {start_vram_str}"
        )

        end_gpu_util_str = ", ".join(
            [f"GPU{i}:{gpu['util_percent']:.1f}%" for i, gpu in enumerate(end_usage['gpus'])]) or "N/A"
        end_vram_str = ", ".join(
            [f"GPU{i}:{gpu['vram_percent']:.1f}%" for i, gpu in enumerate(end_usage['gpus'])]) or "N/A"
        logger.info(
            f"[Resources] End   - CPU: {end_usage['cpu_percent']:.1f}%, RAM: {end_usage['ram_percent']:.1f}%, "
            f"Util: {end_gpu_util_str}, VRAM: {end_vram_str}"
        )

        cpu_diff = end_usage['cpu_percent'] - start_usage['cpu_percent']
        ram_diff = end_usage['ram_percent'] - start_usage['ram_percent']

        if start_usage['gpus'] and end_usage['gpus'] and len(start_usage['gpus']) == len(end_usage['gpus']):
            gpu_util_diff_str = ", ".join(
                [f"GPU{i}:{post['util_percent'] - pre['util_percent']:+.1f}%" for i, (pre, post) in
                 enumerate(zip(start_usage['gpus'], end_usage['gpus']))])
            vram_diff_str = ", ".join(
                [f"GPU{i}:{post['vram_percent'] - pre['vram_percent']:+.1f}%" for i, (pre, post) in
                 enumerate(zip(start_usage['gpus'], end_usage['gpus']))])
        else:
            gpu_util_diff_str = "N/A"
            vram_diff_str = "N/A"

        logger.info(
            f"[Resources] Difference- CPU: {cpu_diff:+.1f}%, RAM: {ram_diff:+.1f}%, "
            f"Util: {gpu_util_diff_str}, VRAM: {vram_diff_str}"
        )
        logger.info("------------------------------------")


@app.post("/v1/agent/chat")
async def agent_chat(request: AgentChatRequest):
    logger.info("Agent Server received request.")
    metrics = MetricsTracker(logger)
    raw_stream_generator = run_organizer_loop(request, logger, MODEL_SERVER_URL)

    metrics_generator = metrics_collecting_stream_generator(raw_stream_generator, metrics)

    return StreamingResponse(
        metrics_generator,
        media_type="text/plain"
    )


if __name__ == "__main__":
    logger.info(f"Starting Agent Server on {AGENT_SERVER_HOST}:{AGENT_SERVER_PORT}")
    uvicorn.run(app, host=AGENT_SERVER_HOST, port=AGENT_SERVER_PORT)
