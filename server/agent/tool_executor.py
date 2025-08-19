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
            # Convert tool error to system role for Ollama compatibility
            messages.append({"role": "system", "content": f"Tool error: {error_msg}"})

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
            # Execute tool without catching exceptions - let real failures bubble up
            result = await execute_tool_async(tool_name, tool_call["arguments"], logger)
            
            # If we get here, the tool succeeded (no exception was thrown)
            logger.info(f"Tool {tool_name} completed successfully")
            status = "success"
            
            # Record successful execution
            execution_id, suggested_alternatives = tool_learning_store.record_tool_execution(
                tool_name=tool_name,
                arguments=tool_call["arguments"],
                result=str(result),
                outcome=ToolOutcome.SUCCESS,
                execution_id=execution_id,
                error_details=None
            )
            
            # Check for failure-success learning opportunities
            if llm_analysis_callback:
                _check_for_learning_opportunities(execution_id, llm_analysis_callback)

            # Update knowledge store for file operations
            if knowledge_store is not None:
                knowledge_store.update_from_tool_execution(tool_name, tool_call["arguments"], result)

            # Stream tool result to UI if callback provided
            if result_callback:
                # Create a concise, user-facing summary of the tool result
                brief_result = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                
                tool_result_summary = {
                    "name": tool_name,
                    "status": status,
                    "brief_result": brief_result,
                    "arguments": tool_call["arguments"]
                }
                await result_callback(tool_result_summary)

            # Present tool result clearly to avoid agent confusion
            # Use a clear format that distinguishes observation from instruction
            tool_result_message = f"TOOL_RESULT from {tool_name}:\n{result}"
            
            messages.append({
                "role": "function",
                'name': tool_name,
                "content": tool_result_message
            })
                
        except Exception as e:
            # Only exceptions are treated as failures
            logger.error(f"Tool {tool_name} failed with exception: {e}", exc_info=True)
            
            # Record failed execution
            execution_id, suggested_alternatives = tool_learning_store.record_tool_execution(
                tool_name=tool_name,
                arguments=tool_call["arguments"],
                result=f"Exception: {str(e)}",
                outcome=ToolOutcome.FAILURE,
                execution_id=execution_id,
                error_details=str(e)
            )
            
            # Check if we have corrective action suggestions
            if suggested_alternatives:
                logger.info(f"Found {len(suggested_alternatives)} suggested alternatives for failed {tool_name}")
                
                # Create corrective action message with suggestions
                alternatives_text = "\n".join([
                    f"• {alt.successful_alternative} (confidence: {alt.confidence:.1%})"
                    for alt in suggested_alternatives[:3]  # Top 3 alternatives
                ])
                
                error_message = f"TOOL_ERROR: {tool_name} failed with exception: {str(e)}\n\nCORRECTIVE ACTION SUGGESTED:\nBased on past learnings, try these alternatives:\n{alternatives_text}"
                
                # Add a system message with the learning-based suggestion
                corrective_message = {
                    "role": "system",
                    "content": f"TOOL FAILURE RECOVERY: {tool_name} failed. High-confidence alternatives available:\n{alternatives_text}\n\nConsider using these learned alternatives instead of retrying the same approach."
                }
                messages.append(corrective_message)
                
            else:
                error_message = f"TOOL_ERROR: {tool_name} failed with exception: {str(e)}"
            
            messages.append({"role": "system", "content": error_message})
            
            # Stream error to UI if callback provided
            if result_callback:
                error_summary = {
                    "name": tool_name,
                    "status": "error", 
                    "brief_result": f"Exception: {str(e)}",
                    "arguments": tool_call["arguments"]
                }
                await result_callback(error_summary)

    return tool_calls_made


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
    Let exceptions bubble up naturally - they indicate real tool failures.
    """
    logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
    # Run the potentially blocking tool in a thread pool
    # Do NOT catch exceptions here - let them bubble up to indicate real failures
    result = await asyncio.to_thread(tool_manager.run_tool, tool_name, tool_args)
    
    # tool_manager.run_tool returns the actual result from the tool
    return result


def execute_tool(tool_name: str, tool_args: dict, logger) -> str:
    """
    Synchronous version for backward compatibility.
    Looks up a tool by name in the registry and executes it with the given arguments.
    Let exceptions bubble up naturally - they indicate real tool failures.
    """
    logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
    # Do NOT catch exceptions here - let them bubble up to indicate real failures
    result = tool_manager.run_tool(tool_name, tool_args)
    return json.dumps(result)