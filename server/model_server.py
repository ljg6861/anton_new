import logging
import os
import time
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import ollama # Import the ollama client library

from metrics import MetricsTracker
from server.helpers import AgentChatRequest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OLLAMA_MODEL_ID = 'qwen3:30b-a3b-thinking-2507-q4_K_M'
SMALL_MODEL_ID = 'mistral:7b-instruct-v0.3-q4_K_M'
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# ----------------------------

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
MAX_SEQ_LENGTH = 8192 # This is less critical here as Ollama manages context internally

def get_all_resource_usage(logger_instance) -> dict:
    """
    Captures a snapshot of current CPU, RAM usage.
    GPU monitoring via pynvml is removed as Ollama abstracts this.
    If you need GPU stats, consider `nvidia-smi` via subprocess.
    """
    usage = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "gpus": [] # Placeholder, as we don't have direct pynvml access here
    }
    # You could add subprocess calls to `nvidia-smi` here if detailed GPU stats are critical,
    # but be aware of the performance overhead for frequent calls.
    return usage

# --- FastAPI Lifespan (Startup/Shutdown Events) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Server starting up...")

    logger.info("--- OLLAMA MODEL CHECK METRICS ---")

    pre_load_usage = get_all_resource_usage(logger)

    startup_start_time = time.monotonic()
    startup_duration = time.monotonic() - startup_start_time

    post_load_usage = get_all_resource_usage(logger)

    # Simplified resource logging as GPU details are not directly accessible via pynvml
    logger.info(
        f"[Resources] Pre-Load  - CPU: {pre_load_usage['cpu_percent']:.1f}%, RAM: {pre_load_usage['ram_percent']:.1f}%"
    )
    logger.info(
        f"[Resources] Post-Load - CPU: {post_load_usage['cpu_percent']:.1f}%, RAM: {post_load_usage['ram_percent']:.1f}%"
    )

    cpu_diff = post_load_usage['cpu_percent'] - pre_load_usage['cpu_percent']
    ram_diff = post_load_usage['ram_percent'] - pre_load_usage['ram_percent']

    logger.info(
        f"[Resources] Difference- CPU: {cpu_diff:+.1f}%, RAM: {ram_diff:+.1f}%"
    )

    logger.info(f"[Latency] Ollama model check complete in {startup_duration:.2f} seconds.")
    logger.info("-----------------------------")

    logger.info("✅ Server is fully initialized and ready to accept requests.")
    yield
    logger.info("🌙 Server shutting down.")
    logger.info("Server shutdown complete.")


app = FastAPI(title="Ollama API Server", version="1.0.0", lifespan=lifespan)


async def metrics_collecting_stream_generator(
        ollama_stream: AsyncGenerator[dict, None], # Ollama streams dictionaries
        metrics: MetricsTracker
) -> AsyncGenerator[str, None]:
    chunk_count = 0
    metrics.get_resource_usage = lambda: get_all_resource_usage(logger)
    metrics.resource_snapshots['request_start'] = metrics.get_resource_usage()
    try:
        async for chunk in ollama_stream:
            # Ollama's chat API yields dicts with a 'content' field in 'message'
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
                chunk_count += 1
            elif 'done' in chunk and chunk['done']:
                # The 'done' chunk signifies the end and contains final metrics
                pass # We'll process final metrics in finally block
    finally:
        metrics.end_time = time.monotonic()
        metrics.step_token_counts['generation'] = chunk_count
        e2e_latency = metrics.end_time - metrics.start_time
        throughput = chunk_count / e2e_latency if e2e_latency > 0 else 0
        metrics.resource_snapshots['request_end'] = metrics.get_resource_usage()
        logger.info("--- REQUEST METRICS ---")
        logger.info(f"[Latency] End-to-End: {e2e_latency:.2f} seconds")
        logger.info(f"[Throughput] Chunks per Second: {throughput:.2f}")
        logger.info(f"[Throughput] Total Chunks: {chunk_count}")
        start_usage = metrics.resource_snapshots['request_start']
        end_usage = metrics.resource_snapshots['request_end']
        logger.info(
            f"[Resources] Start - CPU: {start_usage['cpu_percent']:.1f}%, "
            f"RAM: {start_usage['ram_percent']:.1f}%"
        )
        logger.info(
            f"[Resources] End   - CPU: {end_usage['cpu_percent']:.1f}%, "
            f"RAM: {end_usage['ram_percent']:.1f}%"
        )
        logger.info("-----------------------")


@app.post("/v1/chat/stream")
async def chat_completions_stream(request: AgentChatRequest):
    logger.info("Received request on /v1/chat/stream")
    metrics = MetricsTracker(logger)
    client = ollama.AsyncClient(host=OLLAMA_HOST)

    try:
        ollama_options = {
            "temperature": request.temperature,
            "num_predict": 2048,
        }

        if request.complex:
            model_to_use = OLLAMA_MODEL_ID
            logger.info("Switching to large model for complex query.")
        else:
            model_to_use = SMALL_MODEL_ID
            logger.info("Using small model for simple query.")

        # Step 2: Use the determined model for the actual chat
        actual_ollama_stream_generator = await client.chat(
            model=model_to_use,
            messages=request.messages,
            stream=True,
            options=ollama_options
        )

    except ollama.ResponseError as e:
        logger.error(f"Error from Ollama during routing or chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama inference error: {e.error}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during routing or Ollama call setup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during Ollama call setup.")

    metrics_generator = metrics_collecting_stream_generator(actual_ollama_stream_generator, metrics)

    return StreamingResponse(metrics_generator, media_type="text/event-stream")


if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on {SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)