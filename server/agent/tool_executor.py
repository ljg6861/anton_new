# agent/tool_executor.py

"""
Handles the execution of tools from the tool registry.
"""
import json

from tools.tool_manager import tool_manager


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