# agent/tool_executor.py

"""
Handles the execution of tools using the centralized ToolsRouter.
Provides safety, observability, and retry capabilities.
"""

import html
import json
import asyncio
from typing import Any

from server.agent.tools_router import tools_router, ExecutionStatus


async def process_tool_calls(
        response_buffer: str,
        tool_call_regex: Any,
        messages: list[dict],
        logger: Any,
        knowledge_store = None,  # Updated parameter to use KnowledgeStore
        result_callback = None  # Callback to stream tool results to UI
) -> bool:
    """
    Parses and executes all tool calls from the model's response buffer using ToolsRouter.
    Now updates a knowledge store with information about accessed files and tool results.
    Executes independent tool calls in parallel for better performance.

    Returns:
        True if at least one tool was called, False otherwise.
    """
    tool_calls_made = False
    response_buffer = html.unescape(response_buffer)
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

    # Execute all valid tool calls - but enforce single tool per turn for safety
    logger.info('Detected tool calls:\n' + str(tool_calls))
    if tool_calls:
        # Limit to single tool per turn to avoid dependency issues
        if len(tool_calls) > 1:
            logger.warning(f"Multiple tool calls detected ({len(tool_calls)}), executing only the first one to avoid dependencies")
            tool_calls = [tool_calls[0]]
        
        logger.info(f"Executing {len(tool_calls)} tool call...")
        
        # Execute the tool call using ToolsRouter
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        
        try:
            # Use ToolsRouter for safe execution with timeout and retries
            execution_result = await tools_router.call(
                name=tool_name,
                args=tool_call["arguments"],
                timeout_ms=30000  # 30 second timeout
            )
            
            if execution_result.ok:
                tool_result = execution_result.result
                logger.info(f"Tool {tool_name} completed successfully in {execution_result.execution_time_ms:.1f}ms "
                           f"(attempts: {execution_result.attempts})")
                status = "success"
            else:
                # Handle different failure types
                if execution_result.status == ExecutionStatus.BLOCKED:
                    logger.error(f"Tool {tool_name} blocked by allowlist")
                    tool_result = f"Error: Tool '{tool_name}' is not allowed"
                elif execution_result.status == ExecutionStatus.TIMEOUT:
                    logger.error(f"Tool {tool_name} timed out")
                    tool_result = f"Error: Tool '{tool_name}' timed out"
                elif execution_result.status == ExecutionStatus.RETRY_EXHAUSTED:
                    logger.error(f"Tool {tool_name} failed after all retries")
                    tool_result = f"Error: Tool '{tool_name}' failed after retries: {execution_result.error_message}"
                else:
                    logger.error(f"Tool {tool_name} failed: {execution_result.error_message}")
                    tool_result = f"Error: {execution_result.error_message}"
                status = "error"

            # Update knowledge store for file operations
            if knowledge_store is not None:
                knowledge_store.update_from_tool_execution(tool_name, tool_call["arguments"], tool_result)

            # Stream tool result to UI if callback provided
            if result_callback:
                # Create a concise, user-facing summary of the tool result
                brief_result = str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else str(tool_result)
                
                tool_result_summary = {
                    "name": tool_name,
                    "status": status,
                    "brief_result": brief_result,
                    "arguments": tool_call["arguments"],
                    "execution_time_ms": execution_result.execution_time_ms,
                    "attempts": execution_result.attempts,
                    "result": tool_result,
                    "error": execution_result.error_message if not execution_result.ok else None
                }
                await result_callback(tool_result_summary)

            # Append the structured tool result to messages as system role for better model understanding
            # Use "system" role instead of "user" to clearly indicate this is an observation
            messages.append({
                "role": "system",
                "content": f"OBSERVATION: Tool '{tool_name}' result: {tool_result}"
            })
                
        except Exception as e:
            logger.error(f"Error during tool execution: {e}", exc_info=True)
            messages.append({"role": "system", "content": f"TOOL_ERROR: {tool_name} failed: {str(e)}"})
            
            # Stream error to UI if callback provided
            if result_callback:
                error_summary = {
                    "name": tool_name,
                    "status": "error", 
                    "brief_result": f"Error: {str(e)}",
                    "arguments": tool_call["arguments"],
                    "execution_time_ms": 0.0,
                    "attempts": 1,
                    "result": None,
                    "error": str(e)
                }
                await result_callback(error_summary)

    return tool_calls_made


async def execute_tool_async(tool_name: str, tool_args: dict, logger) -> str:
    """
    Executes a tool using ToolsRouter with safety features.
    Uses the centralized router for allowlist checking, timeouts, and retries.
    """
    try:
        logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
        
        # Use ToolsRouter for safe execution
        execution_result = await tools_router.call(
            name=tool_name,
            args=tool_args,
            timeout_ms=30000  # 30 second timeout
        )
        
        if execution_result.ok:
            return execution_result.result
        else:
            error_msg = f"Tool execution failed: {execution_result.error_message}"
            logger.error(error_msg)
            return f'{{"error": "{error_msg}"}}'
            
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
        return f'{{"error": "Failed to execute tool: {str(e)}"}}'


def execute_tool(tool_name: str, tool_args: dict, logger) -> str:
    """
    Synchronous version - delegates to async version using asyncio.
    Maintains backward compatibility while using the new ToolsRouter.
    """
    try:
        logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
        
        # Run the async version in a new event loop if needed
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            pass
        
        if loop and loop.is_running():
            # We're already in an async context, need to use asyncio.create_task
            # This is a fallback that should rarely be used
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, execute_tool_async(tool_name, tool_args, logger))
                return future.result()
        else:
            # No event loop or not running, safe to use asyncio.run
            return asyncio.run(execute_tool_async(tool_name, tool_args, logger))
            
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
        return f'{{"error": "Failed to execute tool: {str(e)}"}}'