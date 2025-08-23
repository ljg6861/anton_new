import json
import logging
import time
import psutil
import uvicorn
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from contextlib import asynccontextmanager
from typing import AsyncGenerator

try:
    from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, \
        nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, NVMLError
except ImportError:
    # Fallback if pynvml is not available
    def nvmlInit(): pass
    def nvmlDeviceGetCount(): return 0
    def nvmlDeviceGetHandleByIndex(i): return None
    def nvmlDeviceGetUtilizationRates(handle): return type('obj', (object,), {'gpu': 0, 'memory': 0})()
    def nvmlDeviceGetMemoryInfo(handle): return type('obj', (object,), {'total': 0, 'used': 0})()
    class NVMLError(Exception): pass

from server.agent.agentic_flow.full_agentic_flow import execute_agentic_flow
from server.agent.pack_builder import build_pack_centroids
from server.agent.react.react_agent import ReActAgent
from server.agent.react.token_budget import TokenBudget
build_pack_centroids()
from server.agent.config import AGENT_SERVER_HOST, AGENT_SERVER_PORT, MODEL_SERVER_URL
from server.agent.rag_manager import rag_manager
from server.agent.knowledge_store import KnowledgeStore

try:
    from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, \
        nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, NVMLError
except ImportError:
    # Fallback if pynvml is not available
    def nvmlInit(): pass
    def nvmlDeviceGetCount(): return 0
    def nvmlDeviceGetHandleByIndex(i): return None
    def nvmlDeviceGetUtilizationRates(handle): return type('obj', (object,), {'gpu': 0, 'memory': 0})()
    def nvmlDeviceGetMemoryInfo(handle): return type('obj', (object,), {'total': 0, 'used': 0})()
    class NVMLError(Exception): pass

from metrics import MetricsTracker
from server.agent.tools.tool_defs import get_all_tools
from server.agent.tools.tool_manager import tool_manager
from server.helpers import AgentChatRequest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from pathlib import Path
import json, numpy as np
from server.agent.rag_manager import rag_manager


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

    from server.agent.code_indexer import code_indexer

    # Run in a background thread to not block startup
    import threading
    def index_code():
        logger.info("Starting code indexing...")
        # Use refresh_index to ensure deleted files are cleaned up
        files_updated = code_indexer.refresh_index()
        logger.info(f"✅ Code indexing complete. {files_updated} files updated.")
        # Save the RAG index to persist embeddings
        rag_manager.save()

    indexing_thread = threading.Thread(target=index_code)
    indexing_thread.daemon = True
    indexing_thread.start()

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



@app.post("/v1/agent/chat")
async def agent_chat(request: AgentChatRequest):
    """
    Handles incoming chat requests using the ReAct agent with three-memory architecture.
    Uses token budgeting to prevent context overflow.
    """
    initial_messages = [msg.model_dump() for msg in request.messages]
        
    async def react_with_metrics():
        import re
        buffer = ''
        async for token in execute_agentic_flow(initial_messages):
            buffer += token
            yield token
        
        # Clean token tags from buffer before logging
        clean_buffer = re.sub(r'<token>.*?</token>', '', buffer)
        logger.info("Responsed to " + str(initial_messages[-1]) + ' with \n' + clean_buffer) 
        
    
    return StreamingResponse(
        react_with_metrics(),
        media_type="text/event-stream"
    )

from server.agent.code_index_refresher import code_refresher

@app.on_event("startup")
def start_code_refresher():
    code_refresher.start()

@app.on_event("shutdown")
def stop_code_refresher():
    code_refresher.stop()


if __name__ == "__main__":
    logger.info(f"Starting Agent Server on {AGENT_SERVER_HOST}:{AGENT_SERVER_PORT}")
    uvicorn.run(app, host=AGENT_SERVER_HOST, port=AGENT_SERVER_PORT)
