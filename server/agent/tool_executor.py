# agent/tool_executor.py

"""
Handles the execution of tools from the tool registry.
Enhanced with immediate tool failure recording and learning capabilities.
"""

import html
import uuid
import logging
from server.agent.tools.tool_manager import tool_manager
from server.agent.tool_learning_store import tool_learning_store, ToolOutcome

logger = logging.getLogger(__name__)

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
        knowledge_store = None,  # Updated parameter to use KnowledgeStore
        result_callback = None,  # Callback to stream tool results to UI
        llm_analysis_callback = None  # Callback for LLM learning analysis
) -> bool:
    """
    Parses and executes all tool calls from the model's response buffer.
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
        
        # Execute the tool call
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        execution_id = str(uuid.uuid4())
        
        # Check for relevant past learnings before execution
        relevant_learnings = tool_learning_store.query_relevant_learnings(
            tool_name, 
            tool_call["arguments"], 
            context=str(messages[-3:]) if len(messages) > 3 else ""
        )
        
        # If we have high-confidence learnings, inform the LLM
        if relevant_learnings:
            high_confidence_learnings = [l for l in relevant_learnings if l.confidence > 0.8]
            if high_confidence_learnings:
                learning_advice = "\n".join([
                    f"⚠️ LEARNING: {learning.failure_pattern} → {learning.successful_alternative}" 
                    for learning in high_confidence_learnings[:2]
                ])
                messages.append({
                    "role": "system", 
                    "content": f"Tool Learning Advisory:\n{learning_advice}\n\nConsider these learnings before proceeding with {tool_name}."
                })
        
        try:
            result = await execute_tool_async(tool_name, tool_call["arguments"], logger)
            
            # Determine success/failure based on result content
            is_success, error_details = _analyze_tool_result(result)
            outcome = ToolOutcome.SUCCESS if is_success else ToolOutcome.FAILURE
            
            # Record execution immediately with proper outcome detection
            tool_learning_store.record_tool_execution(
                tool_name=tool_name,
                arguments=tool_call["arguments"],
                result=str(result),
                outcome=outcome,
                execution_id=execution_id,
                error_details=error_details
            )
            
            if is_success:
                logger.info(f"Tool {tool_name} completed successfully")
                status = "success"
                tool_result = result
                
                # Check for failure-success learning opportunities
                if llm_analysis_callback:
                    _check_for_learning_opportunities(execution_id, llm_analysis_callback)
                    
            else:
                logger.error(f"Tool {tool_name} failed: {error_details}")
                status = "error" 
                tool_result = f"Error: {error_details}"

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
                    "arguments": tool_call["arguments"]
                }
                await result_callback(tool_result_summary)

            # Append the structured tool result to messages as system role for better model understanding
            # Use "function" role instead of "user" to clearly indicate this is an observation
            messages.append({
                "role": "function",
                'name' : tool_name,
                "content": tool_result
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
                    "arguments": tool_call["arguments"]
                }
                await result_callback(error_summary)

    return tool_calls_made


def _analyze_tool_result(result: Any) -> tuple[bool, str]:
    """
    Analyze tool result to determine if it was actually successful or failed.
    This fixes the issue where tools return success=True even when they error out.
    
    Returns:
        Tuple of (is_success, error_details)
    """
    result_str = str(result)
    
    # Check for explicit error indicators
    error_indicators = [
        "❌ Error",
        "❌ Failed", 
        "Error:",
        "Exception:",
        "command not found",
        "permission denied",
        "file not found",
        "no such file",
        "connection refused",
        "timeout",
        "failed to",
        "cannot",
        "unable to",
        "fatal:",
        "error:"
    ]
    
    for indicator in error_indicators:
        if indicator.lower() in result_str.lower():
            return False, result_str
    
    # Check for success indicators
    success_indicators = [
        "✅ Success",
        "successfully",
        "completed",
        "done",
        "ok"
    ]
    
    has_success_indicator = any(indicator.lower() in result_str.lower() for indicator in success_indicators)
    
    # If result is very short and no clear indicators, consider it potentially successful
    if len(result_str.strip()) < 10 and not any(indicator.lower() in result_str.lower() for indicator in error_indicators):
        return True, None
    
    # If we have explicit success indicators, it's successful
    if has_success_indicator:
        return True, None
    
    # If result contains structured data (JSON, lists, etc.), likely successful
    if result_str.strip().startswith(('{', '[', '"')) or 'data' in result_str.lower():
        return True, None
    
    # Default to success if no clear error indicators
    return True, None


def _check_for_learning_opportunities(success_execution_id: str, llm_analysis_callback):
    """
    Check if this successful execution follows a recent failure and could be a learning opportunity.
    """
    try:
        # Look for recent failures in the same conversation that could be related
        import asyncio
        
        async def async_learning_check():
            try:
                # Get recent execution history
                recent_executions = tool_learning_store.current_execution_sequence[-10:]  # Last 10 executions
                
                # Find recent failures that could be related to this success
                for i, exec_record in enumerate(recent_executions):
                    if (exec_record.outcome == ToolOutcome.FAILURE and 
                        exec_record.execution_id != success_execution_id):
                        
                        # Check if this failure is recent enough to be related
                        success_record = tool_learning_store._get_execution_record(success_execution_id)
                        if success_record and (success_record.timestamp - exec_record.timestamp) < 300:  # Within 5 minutes
                            
                            logger.info(f"Potential learning opportunity: failure {exec_record.execution_id} -> success {success_execution_id}")
                            
                            # Trigger learning analysis
                            learning = tool_learning_store.analyze_failure_success_pattern(
                                exec_record.execution_id,
                                success_execution_id,
                                llm_analysis_callback
                            )
                            
                            if learning:
                                logger.info(f"New learning created: {learning.learning_id}")
                                break
                            
            except Exception as e:
                logger.error(f"Error in learning opportunity check: {e}", exc_info=True)
        
        # Schedule async learning check
        asyncio.create_task(async_learning_check())
        
    except Exception as e:
        logger.error(f"Failed to check for learning opportunities: {e}", exc_info=True)




async def execute_tool_async(tool_name: str, tool_args: dict, logger) -> str:
    """
    Looks up a tool by name in the registry and executes it with the given arguments.
    Uses asyncio.to_thread to run blocking IO-bound tools in a thread pool.
    Enhanced to properly detect and return actual tool results vs errors.
    """
    try:
        logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
        # Run the potentially blocking tool in a thread pool
        result = await asyncio.to_thread(tool_manager.run_tool, tool_name, tool_args)
        
        # tool_manager.run_tool returns the actual result from the tool
        # No need to wrap in additional JSON here since the tool should return its own formatted result
        return result
        
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
        # Return the error in a format that _analyze_tool_result can detect
        return f"❌ Error executing tool '{tool_name}': {str(e)}"


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