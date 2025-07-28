# agent/agent_loop.py
from typing import AsyncGenerator, Any

import httpx

from client.context_builder import ContextBuilder
from server.agent.message_handler import prepare_initial_messages, handle_loop_detection
from server.agent import config
from server.agent.prompts import get_thinking_prompt
from server.agent.tool_executor import process_tool_call
from server.model_server import AgentChatRequest


async def _stream_model_response(api_base_url: str, messages: list[dict], logger: Any, tools) -> AsyncGenerator[str, None]:
    """
    Calls the server's streaming chat endpoint and yields the response tokens.
    """
    request_payload = {
        "messages": messages,
        "temperature": 0.6,
        'tools' : tools,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        print("Calling endpoint: " + api_base_url + '/v1/chat/stream')
        try:
            async with client.stream("POST", f"{api_base_url}/v1/chat/stream", json=request_payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    yield chunk
        except httpx.RequestError as e:
            logger.error(f"API request to model server failed: {e}")
            yield f"\n[ERROR: Could not connect to the model server: {e}]\n"
        except Exception as e:
            logger.error(f"An unexpected error occurred during model streaming: {e}", exc_info=True)
            yield f"\n[ERROR: An unexpected error occurred: {e}]\n"


async def run_agent_loop(request: AgentChatRequest, logger: Any, api_base_url: str) -> AsyncGenerator[str, None]:
    """
    Manages the agentic loop by making API calls to the model server.
    This function is now decoupled from the model and tokenizer objects.
    """
    messages = prepare_initial_messages(request.messages)
    messages.insert(0, {'role' : 'system', 'content' : await ContextBuilder().build_system_prompt()})

    for turn in range(config.MAX_TURNS):
        logger.info(f"--- Agent Turn {turn + 1}/{config.MAX_TURNS} ---")

        response_buffer = ""
        thinking = True

        stream_generator = _stream_model_response(api_base_url, messages, logger, request.tools)

        async for token in stream_generator:
            if thinking:
                if "</think>" in response_buffer:
                    thinking = False
            response_buffer += token
            yield token

        if len(response_buffer.split("</think>")) > 1 :
            content = response_buffer.split("</think>")[1]
        else:
            content = response_buffer
        messages.append({"role": "assistant", "content": content})

        was_called = await process_tool_call(content, config.TOOL_CALL_REGEX, messages, logger)

        if not was_called:
            logger.info("No tool call detected. Ending conversation.")
            return

    logger.warning("Agent reached maximum turn limit.")
    yield "\n[Agent reached maximum turns. Ending conversation.]"