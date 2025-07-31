# server/agent/doer.py
import re
import time
from typing import AsyncGenerator, Any, List
import httpx

from server.agent import config
from server.agent.config import ASSISTANT_ROLE
from server.agent.tool_executor import process_tool_calls


async def execute_turn(
    api_base_url: str,
    messages: list[dict],
    logger: Any,
    tools
) -> AsyncGenerator[str, None]:
    request_payload = {
        "messages": messages,
        "temperature": 0.6,
        'tools': tools,
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
        api_base_url: str
) -> AsyncGenerator[str, None]:
    for turn in range(config.MAX_TURNS):
        logger.info(f"Doer Turn {turn + 1}/{config.MAX_TURNS}")

        llm_call_start_time = time.monotonic()
        response_buffer = ""
        chunk_count = 0
        async for token in execute_turn(api_base_url, messages, logger, tools):
            response_buffer += token
            chunk_count += 1
            yield token

        llm_call_latency = time.monotonic() - llm_call_start_time
        throughput = chunk_count / llm_call_latency if llm_call_latency > 0 else 0
        logger.info(f"[Metrics] Doer LLM Call (Turn {turn+1}): Latency={llm_call_latency:.2f}s, Throughput={throughput:.2f} chunks/sec")

        logger.info("Doer said:\n" + response_buffer)
        content = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
        messages.append({"role": ASSISTANT_ROLE, "content": content})

        tool_call_start_time = time.monotonic()
        await process_tool_calls(content, config.TOOL_CALL_REGEX, messages, logger)
        tool_call_latency = time.monotonic() - tool_call_start_time
        logger.info(f"[Metrics] Doer Tool Processing (Turn {turn+1}): Latency={tool_call_latency:.2f}s")

        if "FINAL ANSWER:" in content:
            logger.info("Doer received 'FINAL ANSWER:', breaking loop.")
            break

    yield "\n\n[Doer loop finished.]"