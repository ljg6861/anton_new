# agent/agent_loop.py

"""
The core orchestrator for the multi-turn, streaming agent.
"""
import asyncio
from typing import AsyncGenerator, Any

# Local, refactored module imports
from server.agent.message_handler import prepare_initial_messages, handle_loop_detection
from server.agent import config
from server.agent.rag_handler import RAGHandler
from server.agent.tool_executor import process_tool_call

# External dependency for streaming
from server.agent.streaming import stream_response


async def run_agent_loop(request: Any, logger: Any, model: Any, tokenizer: Any) -> AsyncGenerator[str, None]:
    """
    Manages the agentic loop, orchestrating message handling, streaming,
    loop detection, and tool execution.
    """
    messages = prepare_initial_messages(request.messages)
    gen_kwargs = config.get_generation_kwargs(tokenizer)
    recent_thoughts: list[str] = []

    # ✨ NEW: Instantiate the handler to be used for post-response learning.
    rag_handler = RAGHandler(model, tokenizer, logger)

    for turn in range(config.MAX_TURNS):
        logger.info(f"--- Agent Turn {turn + 1}/{config.MAX_TURNS} ---")

        response_buffer = ""

        stream_generator = stream_response(model, tokenizer, messages, gen_kwargs)
        async for token in stream_generator:
            yield token
            response_buffer += token

        thought_match = config.THOUGHT_SUMMARY_REGEX.search(response_buffer)
        current_thought = thought_match.group(1).strip() if thought_match else ""

        if current_thought:
            if handle_loop_detection(recent_thoughts, current_thought, messages, logger,
                                     config.LOOP_DETECTION_THRESHOLD):
                yield "\n[INFO: Agent stuck in a loop. Forcing a new approach...]\n"
                continue
            yield f"🤔 {current_thought}\n"
        else:
            recent_thoughts.clear()

        content_to_append = config.THOUGHT_SUMMARY_REGEX.sub("", response_buffer).lstrip()
        messages.append({"role": "assistant", "content": content_to_append})

        tool_message, was_called = await process_tool_call(response_buffer, config.TOOL_CALL_REGEX, messages, logger)
        if tool_message:
            yield tool_message

        if not was_called:
            final_answer = content_to_append.strip()
            yield final_answer
            logger.info("No tool call detected. Ending conversation.")

            asyncio.create_task(rag_handler.review_and_learn(
                conversation_history=list(messages)  # Pass a copy
            ))
            return

    logger.warning("Agent reached maximum turn limit.")
    yield "\n[Agent reached maximum turns. Ending conversation.]"