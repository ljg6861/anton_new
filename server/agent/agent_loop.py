# agent/agent_loop.py

"""
The core logic for the multi-turn, streaming agent, now with a fast-path
complexity check.
"""
import json
import re
from typing import AsyncGenerator, Dict, Any, List

from server.agent.prompts import get_thinking_prompt, get_first_pass_prompt
from server.agent.streaming import stream_model_response
from server.agent.tool_executor import execute_tool


def _get_generation_kwargs(request: Any, tokenizer: Any) -> Dict[str, Any]:
    """
    Constructs the generation arguments for the model for the main thinking loop.
    """
    temperature = request.temperature if request.temperature > 0 else 0.01
    return {
        "max_new_tokens": 4096,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": temperature,
    }


def _prepare_thinking_messages(request: Any) -> List[Dict[str, Any]]:
    """
    Prepares the list of messages for the main thinking loop, injecting the
    system prompt and tool definitions.
    """
    system_prompt = get_thinking_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    if request.tools:
        tool_schemas = [tool.model_dump() for tool in request.tools]
        tool_prompt = (
            "You have access to the following tools. "
            "You must use them when needed by wrapping your call in <tool_call> XML tags. "
            f"The tool definitions are: {json.dumps(tool_schemas, indent=2)}"
        )
        messages.append({"role": "system", "content": tool_prompt})

    messages.extend([msg.model_dump() for msg in request.messages])
    return messages


async def run_agent_loop(request: Any, logger: Any, model: Any, tokenizer: Any) -> AsyncGenerator[str, None]:
    """
    Manages the agentic loop. It first performs a fast-path check to see if
    the query is simple. If so, it answers directly. If not, it enters a
    multi-turn thinking and tool-use loop.
    """
    # --- First Pass: Fast-track simple requests ---
    logger.info("--- Agent: Decision Pass ---")
    decision_messages = [
        {"role": "system", "content": get_first_pass_prompt()},
        request.messages[-1].model_dump(),
    ]
    decision_gen_kwargs = {
        "max_new_tokens": 1024,
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": 0.01,
        "do_sample": False,
    }

    first_pass_buffer = ""
    is_complex = False

    # Stream the first pass to see if it's a simple answer or a "THINK" command.
    async for token in stream_model_response(model, tokenizer, decision_messages, decision_gen_kwargs, think=False):
        first_pass_buffer += token
        if "THINK" in first_pass_buffer:
            is_complex = True
            break  # Complexity detected, break to enter the main loop.
        else:
            yield token  # Not complex yet, so stream the token as part of a simple answer.

    if not is_complex:
        logger.info("Simple request. Direct response has been streamed.")
        return  # The entire simple answer was streamed, so we are done.

    # --- Main Loop: Handle complex requests that require thinking ---
    logger.info("Complex task detected. Entering thinking loop.")
    messages = _prepare_thinking_messages(request)
    gen_kwargs = _get_generation_kwargs(request, tokenizer)
    MAX_TURNS = 25
    tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    thought_summary_regex = re.compile(r"<thought_summary>(.*?)</thought_summary>", re.DOTALL)

    for turn in range(MAX_TURNS):
        logger.info(f"--- Agent Turn {turn + 1}/{MAX_TURNS} (Thinking) ---")

        response_buffer = ""
        is_in_tool_call = False
        stream_generator = stream_model_response(model, tokenizer, messages, gen_kwargs)
        last_summary_end_pos = 0
        async for token in stream_generator:
            response_buffer += token
            thoughts = thought_summary_regex.search(response_buffer, last_summary_end_pos)
            if thoughts:
                summary_text = thoughts.group(1).strip()
                yield f"🤔 {summary_text}\n"
            if not is_in_tool_call:
                if "<tool_call>" in response_buffer:
                    is_in_tool_call = True
                    preamble = response_buffer.split("<tool_call>", 1)[0]
                    yield preamble
                else:
                    yield token

        messages.append({"role": "assistant", "content": response_buffer})
        match = tool_call_regex.search(response_buffer)

        if not match:
            logger.info("No tool call detected. Ending conversation.")
            if is_in_tool_call:
                yield response_buffer.split("<tool_call>", 1)[1]
            return

        logger.info("Tool call detected. Executing tool.")
        tool_call_content = match.group(1).strip()

        try:
            tool_data = json.loads(tool_call_content)
            tool_name = tool_data.get("name")
            tool_args = tool_data.get("arguments", {})

            yield f"\n\n[INFO: Calling tool `{tool_name}` with arguments: {json.dumps(tool_args)}]"
            tool_result = execute_tool(tool_name, tool_args, logger)
            tool_result_str = json.dumps({"result": tool_result})
            yield f"\n[INFO: Tool `{tool_name}` returned.]\n\n"
            messages.append({"role": "tool", "content": tool_result_str})

        except json.JSONDecodeError as e:
            error_msg = f"Error: The model produced invalid JSON for a tool call: {e}"
            logger.error(f"{error_msg}\nContent: {tool_call_content}", exc_info=True)
            yield f"\n[{error_msg}]\n"
            messages.append({"role": "tool", "content": json.dumps({"error": error_msg})})
            return

    logger.warning("Agent reached maximum turn limit.")
    yield "\n[Agent reached maximum turns. Ending conversation.]"
