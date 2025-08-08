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
import asyncio
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
    Executes independent tool calls in parallel for better performance.

    Returns:
        True if at least one tool was called, False otherwise.
    """
    tool_calls_made = False
    matches = tool_call_regex.finditer(response_buffer)

    # Collect all tool calls first
    tool_calls = []
    for match in matches:
        tool_calls_made = True
        tool_call_content = match.group(1).strip()

        try:
            tool_data = json.loads(tool_call_content)
            tool_name = tool_data.get("name")
            if not tool_name:
                raise KeyError("'name' not found in tool data.")
            
            tool_args = tool_data.get("arguments", {})
            tool_calls.append({
                "name": tool_name,
                "arguments": tool_args,
                "raw_content": tool_call_content
            })
        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Error: Invalid tool call format. Reason: {e}"
            logger.error(f"{error_msg}\nContent: {tool_call_content}")
            # Convert tool error to user role for Ollama compatibility
            messages.append({"role": "user", "content": f"Tool error: {error_msg}"})

    # Execute all valid tool calls in parallel if there are any
    if tool_calls:
        logger.info(f"Executing {len(tool_calls)} tool calls in parallel...")
        
        # Create tasks for parallel execution
        tasks = []
        for tool_call in tool_calls:
            task = execute_tool_async(tool_call["name"], tool_call["arguments"], logger)
            tasks.append(task)
        
        # Execute all tools in parallel
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and update messages
            for i, (tool_call, result) in enumerate(zip(tool_calls, results)):
                tool_name = tool_call["name"]
                
                if isinstance(result, Exception):
                    logger.error(f"Tool {tool_name} failed with exception: {result}")
                    tool_result = f"Error: {str(result)}"
                else:
                    tool_result = result
                    logger.info(f"Tool {tool_name} completed successfully")

                # Update knowledge store for file operations
                if knowledge_store is not None:
                    knowledge_store.update_from_tool_execution(tool_name, tool_call["arguments"], tool_result)

                # Append the structured tool result to messages as user role for Ollama compatibility
                # Ollama expects system/user/assistant roles, not "tool" role
                messages.append({
                    "role": "user",
                    "content": f"Tool result for {tool_name}: {tool_result}"
                })
                
        except Exception as e:
            logger.error(f"Error during parallel tool execution: {e}", exc_info=True)
            messages.append({"role": "user", "content": f"Tool execution error: {str(e)}"})

    return tool_calls_made




async def execute_tool_async(tool_name: str, tool_args: dict, logger) -> str:
    """
    Looks up a tool by name in the registry and executes it with the given arguments.
    Uses asyncio.to_thread to run blocking IO-bound tools in a thread pool.
    """
    try:
        logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
        # Run the potentially blocking tool in a thread pool
        result = await asyncio.to_thread(tool_manager.run_tool, tool_name, tool_args)
        return result
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
        return f'{{"error": "Failed to execute tool: {str(e)}"}}'


def execute_tool(tool_name: str, tool_args: dict, logger) -> str:
    """
    Synchronous version for backward compatibility.
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