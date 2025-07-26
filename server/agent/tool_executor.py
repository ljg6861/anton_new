# agent/tool_executor.py

"""
Handles the execution of tools from the tool registry.
"""

from server.tools.tool_manager import tool_manager

# agent/tool_handler.py

"""
Handles parsing and execution of tool calls from the model's output.
"""
import json
from typing import Tuple, Any

async def process_tool_call(
        response_buffer: str,
        tool_call_regex: Any,
        messages: list[dict],
        logger: Any
) -> Tuple[str, bool]:
    """
    Parses and executes a tool call from the model's response buffer.

    Returns:
        A tuple containing (yield_message, was_tool_called).
    """
    tool_match = tool_call_regex.search(response_buffer)
    if not tool_match:
        return "", False

    tool_call_content = tool_match.group(1).strip()
    yield_message = ""
    try:
        tool_data = json.loads(tool_call_content)
        tool_name = tool_data["name"]
        tool_args = tool_data.get("arguments", {})

        yield_message = f"\n[INFO: Calling tool `{tool_name}`...]\n"
        tool_result = execute_tool(tool_name, tool_args, logger)
        tool_result_str = json.dumps({"result": tool_result})
        messages.append({"role": "tool", "content": tool_result_str})

    except (json.JSONDecodeError, KeyError) as e:
        error_msg = f"Error: Invalid tool call format. Reason: {e}"
        logger.error(f"{error_msg}\nContent: {tool_call_content}")
        messages.append({"role": "tool", "content": json.dumps({"error": error_msg})})
        yield_message = f"\n[{error_msg}]\n"

    return yield_message, True


def execute_tool(tool_name: str, tool_args: dict, logger) -> str:
    """
    Looks up a tool by name in the registry and executes it with the given arguments.
    """
    try:
        logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
        # Assumes the tool's value is a dict with a 'run' callable
        result = tool_manager.run_tool(tool_name, tool_args)
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
        return f'{{"error": "Failed to execute tool: {str(e)}"}}'