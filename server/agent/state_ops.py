"""
State-based operations and utilities for ReAct Agent.
Provides functions to manipulate and query the structured state.
"""
import json
from typing import List, Dict, Any, Optional
from server.agent.state import State, ContextBlob, Evidence, ToolCallTrace, AgentStatus


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 chars for English)"""
    return len(text) // 4


def truncate_to_budget(text: str, budget: int) -> str:
    """Truncate text to fit within token budget"""
    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= budget:
        return text
    
    # Truncate to approximately fit budget
    target_chars = budget * 4
    if len(text) <= target_chars:
        return text
        
    # Truncate at word boundary
    truncated = text[:target_chars]
    last_space = truncated.rfind(' ')
    if last_space > target_chars * 0.8:  # Don't cut too much
        truncated = truncated[:last_space]
    
    return truncated + "..."


def build_working_memory(state: State) -> str:
    """Build working memory from recent messages in state"""
    if not state.working_memory:
        return ""
    
    budget = state.budgets.working_memory_budget
    
    # Start from most recent and work backwards
    selected_messages = []
    total_tokens = 0
    
    for msg in reversed(state.working_memory):
        role = msg.get('role', 'unknown')
        content = f"[{role}]: {msg.get('content', '')}"
        msg_tokens = estimate_tokens(content)
        
        if total_tokens + msg_tokens <= budget:
            selected_messages.insert(0, content)
            total_tokens += msg_tokens
        else:
            break
    
    return "\n".join(selected_messages)


def build_session_memory(state: State) -> str:
    """Build session memory summary from state"""
    parts = []
    
    if state.session_context:
        parts.append(f"Context: {state.session_context}")
        
    if state.session_decisions:
        decisions_text = "; ".join(state.session_decisions[-5:])  # Last 5 decisions
        parts.append(f"Decisions: {decisions_text}")
        
    if state.session_todos:
        todos_text = "; ".join(state.session_todos[-3:])  # Last 3 TODOs
        parts.append(f"TODOs: {todos_text}")
        
    session_text = " | ".join(parts)
    return truncate_to_budget(session_text, state.budgets.session_summary_budget)


def build_context_summary(state: State) -> str:
    """Build comprehensive context summary from state"""
    summary_parts = []
    
    # Recent high-importance context
    high_importance = state.get_high_importance_context(min_importance=2.0)
    if high_importance:
        summary_parts.append("Key Context:")
        for ctx in high_importance[:5]:
            summary_parts.append(f"- [{ctx.context_type}] {ctx.content[:100]}...")
    
    # Recent evidence
    recent_evidence = sorted(state.evidence, key=lambda x: x.timestamp, reverse=True)[:3]
    if recent_evidence:
        summary_parts.append("Recent Evidence:")
        for evidence in recent_evidence:
            summary_parts.append(f"- [{evidence.type}] {evidence.content[:100]}...")
    
    # Tool call summary
    successful_calls = state.get_successful_tool_calls()
    if successful_calls:
        recent_tools = successful_calls[-3:]
        summary_parts.append("Recent Tool Usage:")
        for call in recent_tools:
            summary_parts.append(f"- {call.name}: success")
    
    # Failed calls (important for debugging)
    failed_calls = state.get_failed_tool_calls()
    if failed_calls:
        summary_parts.append("Failed Tools:")
        for call in failed_calls[-2:]:  # Last 2 failures
            summary_parts.append(f"- {call.name}: {call.error_message or 'unknown error'}")
    
    return "\n".join(summary_parts)


def format_thought_action_observation(state: State, thought: str, action: Optional[Dict] = None, 
                                     observation: Optional[str] = None) -> str:
    """Format a thought-action-observation cycle for the scratchpad"""
    parts = []
    
    if thought:
        parts.append(f"Thought: {thought}")
    
    if action:
        action_str = f"Action: {action.get('name', 'unknown')}"
        if action.get('arguments'):
            action_str += f" with {json.dumps(action['arguments'])}"
        parts.append(action_str)
    
    if observation:
        parts.append(f"Observation: {observation}")
    
    parts.append("")  # Add blank line for separation
    return "\n".join(parts)


def update_state_from_tool_result(state: State, tool_name: str, tool_args: Dict[str, Any], 
                                 result: Any, success: bool = True, cost: float = 0.0, 
                                 error: Optional[str] = None, execution_time_ms: float = 0.0,
                                 attempts: int = 1) -> ToolCallTrace:
    """Update state after tool execution using ToolsRouter results"""
    # Find the most recent incomplete tool call for this tool
    trace = None
    for call in reversed(state.tool_calls):
        if call.name == tool_name and call.end_time is None:
            trace = call
            break
    
    if not trace:
        # Create a new trace if none found (shouldn't happen in normal flow)
        trace = state.start_tool_call(tool_name, tool_args)
    
    # Complete the tool call with execution metadata
    state.complete_tool_call(trace, result, success, cost, error)
    
    # Add execution timing information
    if hasattr(trace, 'execution_time_ms'):
        trace.execution_time_ms = execution_time_ms
    if hasattr(trace, 'attempts'):
        trace.attempts = attempts
    
    # Add specific context based on tool type
    if tool_name == "read_file" and success:
        file_path = tool_args.get("file_path", "unknown")
        state.add_context(
            content=str(result)[:1000] if result else "",
            source=f"file:{file_path}",
            importance=2.0,
            context_type="file_content",
            metadata={"file_path": file_path, "execution_time_ms": execution_time_ms}
        )
    elif tool_name == "list_directory" and success:
        path = tool_args.get("path", ".")
        state.add_context(
            content=str(result),
            source=f"directory:{path}",
            importance=1.0,
            context_type="directory_listing",
            metadata={"path": path, "execution_time_ms": execution_time_ms}
        )
    elif tool_name in ["run_git_command", "run_shell_command"] and success:
        command = str(tool_args.get("command", ""))
        state.add_context(
            content=f"Command: {command}\nResult: {str(result)[:500]}",
            source=f"command:{tool_name}",
            importance=1.5,
            context_type="command_output",
            metadata={"command": command, "execution_time_ms": execution_time_ms}
        )
    
    # Update session memory based on successful tool results
    if success:
        if tool_name == "create_file":
            file_path = tool_args.get("file_path", "unknown")
            state.add_decision(f"Created file: {file_path}")
        elif tool_name == "run_git_command" and "checkout -b" in str(tool_args.get("command", "")):
            state.add_decision("Created new git branch")
            state.complete_todo("Create feature branch")
        elif tool_name == "edit_file":
            file_path = tool_args.get("file_path", "unknown")
            state.add_decision(f"Modified file: {file_path}")
    
    return trace


def parse_action_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse tool call action from LLM response"""
    import re
    
    # Look for tool_call pattern
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    match = re.search(tool_call_pattern, response, re.DOTALL)
    
    if match:
        try:
            action_data = json.loads(match.group(1))
            return action_data
        except json.JSONDecodeError:
            return None
    
    return None


def is_final_response(response: str) -> bool:
    """Check if response contains final answer"""
    return "Final Answer:" in response


def update_session_context_from_goal(state: State):
    """Update session context based on the goal"""
    goal_lower = state.goal.lower()
    
    if any(word in goal_lower for word in ["implement", "create", "build", "develop"]):
        state.set_session_context("Implementation task")
        state.add_todo("Ensure code compiles")
        state.add_todo("Test implementation")
    elif any(word in goal_lower for word in ["analyze", "review", "examine", "investigate"]):
        state.set_session_context("Analysis task")
        state.add_todo("Gather relevant information")
    elif any(word in goal_lower for word in ["fix", "debug", "resolve", "solve"]):
        state.set_session_context("Debugging task")
        state.add_todo("Identify root cause")
        state.add_todo("Test fix")
    elif any(word in goal_lower for word in ["refactor", "improve", "optimize"]):
        state.set_session_context("Refactoring task")
        state.add_todo("Preserve functionality")
        state.add_todo("Test changes")
    else:
        state.set_session_context("General task")


def get_budget_status(state: State) -> Dict[str, Any]:
    """Get current budget utilization status"""
    return {
        "turns": {
            "used": state.turns,
            "max": state.budgets.max_iterations,
            "remaining": state.budgets.max_iterations - state.turns
        },
        "cost": {
            "used": state.cost,
            "max": state.budgets.max_cost,
            "remaining": state.budgets.max_cost - state.cost
        },
        "time": {
            "elapsed": state.get_duration(),
            "max": state.budgets.total_timeout,
            "remaining": state.budgets.total_timeout - state.get_duration()
        },
        "exceeded": state.is_budget_exceeded()
    }


def summarize_execution(state: State) -> str:
    """Create a summary of the execution for final response"""
    summary_parts = []
    
    # Goal and status
    summary_parts.append(f"Goal: {state.goal}")
    summary_parts.append(f"Status: {state.status.value}")
    summary_parts.append(f"Duration: {state.get_duration():.1f}s")
    summary_parts.append(f"Turns: {state.turns}")
    
    # Tool usage summary
    successful_tools = state.get_successful_tool_calls()
    failed_tools = state.get_failed_tool_calls()
    
    if successful_tools:
        tool_names = [call.name for call in successful_tools]
        summary_parts.append(f"Tools used: {', '.join(set(tool_names))}")
    
    if failed_tools:
        summary_parts.append(f"Failed tools: {len(failed_tools)}")
    
    # Key decisions
    if state.session_decisions:
        summary_parts.append("Key decisions:")
        for decision in state.session_decisions[-3:]:
            summary_parts.append(f"- {decision}")
    
    # Final scratchpad excerpt
    if state.scratchpad:
        scratchpad_lines = state.scratchpad.strip().split('\n')
        if len(scratchpad_lines) > 10:
            summary_parts.append("Final reasoning:")
            summary_parts.extend(scratchpad_lines[-5:])  # Last 5 lines
    
    return "\n".join(summary_parts)
