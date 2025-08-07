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
    tools,
    temperature,
    complex: bool,
) -> AsyncGenerator[str, None]:
    request_payload = {
        "messages": messages,
        "temperature": temperature,
        'tools': tools,
        'complex': complex,
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
        complex: bool,
        knowledge_store = None  # New parameter for knowledge tracking
) -> None:

    for i in range(config.MAX_TURNS):
        response_buffer = ""
        async for token in execute_turn(api_base_url, messages, logger, tools, 0.6, complex):
            response_buffer += token

        logger.info("Doer said:\n" + response_buffer)
        content = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
        messages.append({"role": ASSISTANT_ROLE, "content": content})

        made_tool_calls = await process_tool_calls(content, config.TOOL_CALL_REGEX, messages, logger, knowledge_store)

        if not made_tool_calls:
            return