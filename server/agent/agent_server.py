import json
import logging
import time
import psutil
import uvicorn
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from vllm.third_party.pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, \
    nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, NVMLError

from server.agent.doer import execute_turn
from server.agent.prompts import get_intent_router_prompt
from server.agent.rag_manager import rag_manager

try:
    from pynvml import *
except ImportError:
    pass

from metrics import MetricsTracker
from server.agent.tools.tool_defs import get_all_tools
from server.agent.tools.tool_manager import tool_manager
from server.helpers import AgentChatRequest

# --- Configuration ---
AGENT_SERVER_HOST = "0.0.0.0"
AGENT_SERVER_PORT = 8001
MODEL_SERVER_URL = "http://localhost:8000"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("--- Discovering and Registering Tools ---")
# The enhanced tool manager now automatically discovers and registers tools
# No need for manual registration loop
print(f"--- Tool Registration Complete: {tool_manager.get_tool_count()} tools registered ---")


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
        files_indexed = code_indexer.index_directory()
        logger.info(f"✅ Code indexing complete. {files_indexed} files indexed.")
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
            intent = "COMPLEX_CHAT"
            query = request.messages[-1].content  # Use the last message as the query

        # === STEP 3: Execute the Chosen Workflow and Stream the Result ===

        # --- PATH A: Agentic Workflows (ReAct Agent) ---
        if intent in ["COMPLEX_CHAT"]:
            logger.info(f"Routing to ReAct agent for intent: '{intent}'")

            # Create conversation state for this request
            from server.agent.conversation_state import ConversationState
            from server.agent.react_agent import ReActAgent
            
            # Initialize conversation state with request messages
            initial_messages = [msg.model_dump() for msg in request.messages]
            conversation_state = ConversationState(initial_messages)
            
            # Create ReAct agent
            react_agent = ReActAgent(
                api_base_url=MODEL_SERVER_URL,
                tools=request.tools or [],
                max_iterations=10
            )
            
            # Process with ReAct agent (replaces the complex organizer loop)
            metrics = MetricsTracker(logger)
            
            async def react_with_metrics():
                async for token in react_agent.process_request(conversation_state, logger):
                    yield token
            
            metrics_generator = metrics_collecting_stream_generator(react_with_metrics(), metrics)
            async for agent_token in metrics_generator:
                yield agent_token

        # --- PATH B: Knowledge Retrieval Workflow (Proper RAG implementation) ---
        elif intent == "QUERY_KNOWLEDGE":
            logger.info("Routing to RAG workflow for intent: 'QUERY_KNOWLEDGE'")
            
            # Retrieve relevant knowledge from the indexed knowledge base
            relevant_docs = rag_manager.retrieve_knowledge(query, top_k=5)
            
            if relevant_docs:
                # Build context from retrieved documents
                context_parts = []
                for i, doc in enumerate(relevant_docs):
                    context_parts.append(f"Source {i+1}: {doc['source']}\n{doc['text'][:500]}...")
                
                context = "\n\n".join(context_parts)
                
                rag_system_prompt = f"""You are Anton, an AI assistant. Answer the user's question using the provided context from the knowledge base.

Context from knowledge base:
{context}

If the context doesn't contain relevant information, say so and provide a general response."""
            else:
                rag_system_prompt = "You are Anton, an AI assistant. The knowledge base doesn't contain relevant information for this query. Provide a helpful general response."
            
            rag_messages = [
                {'role': 'system', 'content': rag_system_prompt},
                {'role': 'user', 'content': query}
            ]
            
            rag_chat_gen = execute_turn(
                api_base_url=MODEL_SERVER_URL, 
                messages=rag_messages, 
                logger=logger,
                tools=[],
                temperature=0.7,
                complex=False
            )
            async for chat_token in rag_chat_gen:
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
