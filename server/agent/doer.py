# server/agent/doer.py
import re
import time
from typing import AsyncGenerator, Any, List

# Make httpx optional for testing environments
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

from server.agent import config
from server.agent.config import ASSISTANT_ROLE
from server.agent.tool_executor import process_tool_calls


async def execute_turn(
    api_base_url: str,
    messages: list[dict],
    logger: Any,
    tools,
    temperature,
    is_complex: bool,
) -> AsyncGenerator[str, None]:
    # If httpx is not available, return a mock response for testing
    if not HTTPX_AVAILABLE:
        logger.warning("httpx not available, returning mock response for testing")
        yield "FINAL ANSWER: Mock response for testing - httpx not available"
        return
    
    request_payload = {
        "messages": messages,
        "temperature": temperature,
        'tools': tools,
        'complex': is_complex,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", f"{api_base_url}/v1/chat/stream", json=request_payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    yield chunk
        except httpx.RequestError as e:
            logger.error(f"Doer: API request to model server failed: {e}")
            yield f"\n[ERROR: Could not connect to the model server: {e}]\n"
        except Exception as e:
            logger.error(f"Doer: An unexpected error occurred during model streaming: {e}", exc_info=True)
            yield f"\n[ERROR: An unexpected error occurred: {e}]\n"


async def run_doer_loop(
        messages: List[dict],
        tools: List,
        logger: Any,
        api_base_url: str,
        is_complex: bool,
        context_store: dict = None,
) -> AsyncGenerator[str, None]:
    # Track recent responses to detect thought loops
    recent_responses = []
    
    for turn in range(config.MAX_TURNS):
        logger.info(f"Doer Turn {turn + 1}/{config.MAX_TURNS}")

        llm_call_start_time = time.monotonic()
        response_buffer = ""
        chunk_count = 0
        async for token in execute_turn(api_base_url, messages, logger, tools, 0.6, is_complex):
            response_buffer += token
            content = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
            if content.startswith('FINAL ANSWER:'):
                yield token

        llm_call_latency = time.monotonic() - llm_call_start_time
        throughput = chunk_count / llm_call_latency if llm_call_latency > 0 else 0
        logger.info(f"[Metrics] Doer LLM Call (Turn {turn+1}): Latency={llm_call_latency:.2f}s, Throughput={throughput:.2f} chunks/sec")

        logger.info("Doer said:\n" + response_buffer)
        content = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
        
        # Check for thought loops - if we've seen similar content recently
        content_words = set(content.lower().split())
        for prev_response in recent_responses:
            prev_words = set(prev_response.lower().split())
            # If 80% of words overlap, we might be in a loop
            if len(content_words.intersection(prev_words)) / max(len(content_words), 1) > 0.8:
                logger.warning("Potential thought loop detected. Adding guidance to break out.")
                messages.append({
                    "role": "user",
                    "content": "You appear to be repeating similar responses. Try a different approach or provide a FINAL ANSWER if you've gathered enough information."
                })
                break
        
        # Keep track of recent responses (last 3)
        recent_responses.append(content)
        if len(recent_responses) > 3:
            recent_responses.pop(0)
        
        messages.append({"role": ASSISTANT_ROLE, "content": content})

        tool_call_start_time = time.monotonic()
        await process_tool_calls(content, config.TOOL_CALL_REGEX, messages, logger, context_store)
        tool_call_latency = time.monotonic() - tool_call_start_time
        logger.info(f"[Metrics] Doer Tool Processing (Turn {turn+1}): Latency={tool_call_latency:.2f}s")

        if "FINAL ANSWER:" in content:
            logger.info("Doer received 'FINAL ANSWER:', breaking loop.")
            break