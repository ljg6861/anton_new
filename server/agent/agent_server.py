import json
import logging
from copy import deepcopy

import uvicorn
import time
import psutil
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from vllm.third_party.pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, \
    nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, NVMLError

from server.agent.doer import execute_turn
from server.agent.prompts import get_intent_router_prompt

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
    """
    Handles incoming chat requests by first classifying the user's intent
    and then routing to the appropriate workflow (Agent, RAG, or Chat).
    """
    logger.info("Agent Server received request. Routing intent...")

    async def router_and_stream_generator():
        """
        This generator first determines the user's intent and then yields
        the appropriate response stream.
        """
        # === STEP 1: Get Intent Classification from the Router ===
        # We need the full response from the router, so we don't stream this part.
        # The router's only job is to return a JSON object, not chat.

        # Prepare messages for the router prompt
        router_messages = [{'role': 'system', 'content': get_intent_router_prompt()}] + [
            msg.model_dump() for msg in request.messages
        ]

        # Execute the router call to get the JSON classification
        router_gen = execute_turn(
            api_base_url=MODEL_SERVER_URL,
            messages=router_messages,
            logger=logger,
            tools=[],
            temperature=0.0,  # Use 0 temp for deterministic JSON output
            complex=False
        )

        # Collect the full JSON response from the generator
        router_response_str = "".join([token async for token in router_gen])
        logger.info(f"Router full response: {router_response_str}")

        # === STEP 2: Parse the Intent and Route to the Correct Workflow ===
        try:
            # Clean up potential markdown code fences sometimes added by models
            if router_response_str.strip().startswith("```json"):
                router_response_str = router_response_str.strip()[7:-3].strip()

            parsed_intent = json.loads(router_response_str)
            intent = parsed_intent.get("intent")
            query = parsed_intent.get("query")

            logger.info(f"Successfully parsed intent: '{intent}' for query: '{query}'")

        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse JSON from router: {e}. Defaulting to agent workflow.")
            # Failsafe: If the router fails, assume it's a complex task.
            intent = "EXECUTE_TOOL"
            query = request.messages[-1].content  # Use the last message as the query

        # === STEP 3: Execute the Chosen Workflow and Stream the Result ===

        # --- PATH A: Agentic Workflows (Learning & Tool Use) ---
        if intent in ["COMPLEX_CHAT"]:
            logger.info(f"Routing to main agent for intent: '{intent}'")

            # Create a deepcopy of the request to avoid side effects
            agent_request = deepcopy(request)

            # CRITICAL: We replace the user's original message with the
            # clean query identified by the router. This focuses the agent.
            agent_request.messages[-1].content = query

            metrics = MetricsTracker(logger)
            raw_stream_generator = run_organizer_loop(agent_request, logger, MODEL_SERVER_URL)
            metrics_generator = metrics_collecting_stream_generator(raw_stream_generator, metrics)
            async for agent_token in metrics_generator:
                yield agent_token

        # --- PATH B: Knowledge Retrieval Workflow (Future RAG implementation) ---
        elif intent == "QUERY_KNOWLEDGE":
            logger.info("Routing to RAG workflow for intent: 'QUERY_KNOWLEDGE'")
            f"""
            You are Anton, an AI assistant. Using the knowledge you have, answer the user's question.
            User's question: {query}
            Answer:
            """
            chat_messages = [{'role': 'system', 'content': 'You are Anton, a helpful AI.'},
                             {'role': 'user', 'content': query}]
            simple_chat_gen = execute_turn(api_base_url=MODEL_SERVER_URL, messages=chat_messages, logger=logger,
                                           complex=False)
            async for chat_token in simple_chat_gen:
                yield chat_token

        # --- PATH C: Simple Conversational Chat ---
        elif intent == "GENERAL_CHAT":
            logger.info("Routing to simple chat for intent: 'GENERAL_CHAT'")
            # This is a simple chat. We generate a new response.
            # We use the original messages so the AI has conversation history.
            chat_messages = [{'role': 'system', 'content': 'You are Anton, a friendly and helpful AI assistant.'}] + [
                msg.model_dump() for msg in request.messages
            ]

            simple_chat_gen = execute_turn(
                api_base_url=MODEL_SERVER_URL,
                messages=chat_messages,
                logger=logger,
                tools=[],
                temperature=0.7,  # Allow more creativity in chat
                complex=False
            )
            async for chat_token in simple_chat_gen:
                yield chat_token

        else:
            # Fallback for unknown intents
            logger.warning(f"Unknown intent '{intent}'. Yielding fallback response.")
            fallback_response = "I'm not sure how to handle that request. Could you please rephrase?"
            for char in fallback_response:
                yield char

    return StreamingResponse(
        router_and_stream_generator(),
        media_type="text/plain"
    )


if __name__ == "__main__":
    logger.info(f"Starting Agent Server on {AGENT_SERVER_HOST}:{AGENT_SERVER_PORT}")
    uvicorn.run(app, host=AGENT_SERVER_HOST, port=AGENT_SERVER_PORT)
