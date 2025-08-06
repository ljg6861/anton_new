# agent/tool_executor.py

"""
Handles the execution of tools from the tool registry.
"""

from server.agent.tools.tool_manager import tool_manager

# agent/tool_handler.py

"""
Handles parsing and execution of tool calls from the model's output.
"""
import json
from typing import Any


async def process_tool_calls(
        response_buffer: str,
        tool_call_regex: Any,
        messages: list[dict],
        logger: Any,
        knowledge_store = None  # Updated parameter to use KnowledgeStore
) -> bool:
    """
    Parses and executes all tool calls from the model's response buffer.
    Now updates a knowledge store with information about accessed files and tool results.

    Returns:
        True if at least one tool was called, False otherwise.
    """
    tool_calls_made = False
    matches = tool_call_regex.finditer(response_buffer)

    # Loop through all found tool calls
    for match in matches:
        tool_calls_made = True
        tool_call_content = match.group(1).strip()

        try:
            tool_data = json.loads(tool_call_content)
            tool_name = tool_data.get("name")
            if not tool_name:
                raise KeyError("'name' not found in tool data.")

            logger.info(f"Processing tool call: {tool_name}")
            tool_args = tool_data.get("arguments", {})

            # Execute the tool and get the result
            tool_result = execute_tool(tool_name, tool_args, logger)
            logger.info(f"tool result: {tool_result}")

            # Update knowledge store for file operations
            if knowledge_store is not None:
                knowledge_store.update_from_tool_execution(tool_name, tool_args, tool_result)

            # Append the structured tool result to messages
            messages.append({
                "role": "tool",
                "content": json.dumps({
                    "tool_name": tool_name,
                    "result": tool_result
                })
            })

        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Error: Invalid tool call format. Reason: {e}"
            logger.error(f"{error_msg}\nContent: {tool_call_content}")
            messages.append({"role": "tool", "content": json.dumps({"error": error_msg})})

    return tool_calls_made




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