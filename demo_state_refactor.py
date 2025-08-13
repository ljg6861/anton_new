#!/usr/bin/env python3
"""
Demo script showing the new structured state management system for ReAct Agent.
This demonstrates the clean separation of state from execution logic.
"""

import asyncio
import json
from server.agent.state import make_state, Budgets, AgentStatus
from server.agent.state_ops import (
    update_session_context_from_goal, 
    get_budget_status, 
    summarize_execution,
    format_thought_action_observation
)


def demo_state_creation():
    """Demonstrate creating and using structured state"""
    print("=== Demo: Structured State Management ===\n")
    
    # Create a state following the make_state pattern
    goal = "Refactor the ReAct agent to use structured state management"
    budgets = Budgets(
        total_tokens=8192,
        max_iterations=5,
        per_tool_timeout=30.0,
        max_cost=0.5
    )
    
    state = make_state(goal, budgets, user_id="demo_user")
    
    print(f"1. Created state for goal: {state.goal}")
    print(f"   Session ID: {state.session_id}")
    print(f"   Status: {state.status.value}")
    print(f"   Budgets: {state.budgets.max_iterations} iterations, {state.budgets.total_tokens} tokens")
    print()
    
    # Update session context based on goal
    update_session_context_from_goal(state)
    print(f"2. Session context set to: {state.session_context}")
    print(f"   TODOs added: {state.session_todos}")
    print()
    
    # Simulate adding context throughout execution
    state.add_context("Analyzing current ReAct implementation", "analysis", 2.0, "file_analysis")
    state.add_context("Found TokenBudget class to be replaced", "analysis", 3.0, "key_finding")
    state.add_evidence("code_structure", "ReAct agent uses ad-hoc state management", "code_review", 0.9)
    
    print("3. Added context and evidence:")
    print(f"   Context items: {len(state.context)}")
    print(f"   Evidence items: {len(state.evidence)}")
    print()
    
    # Simulate tool calls
    trace1 = state.start_tool_call("read_file", {"file_path": "react_agent.py"})
    state.complete_tool_call(trace1, "File content...", success=True, cost=0.01)
    
    trace2 = state.start_tool_call("create_file", {"file_path": "state.py", "content": "..."})
    state.complete_tool_call(trace2, "File created successfully", success=True, cost=0.02)
    
    print("4. Executed tool calls:")
    for i, call in enumerate(state.tool_calls, 1):
        print(f"   {i}. {call.name}: {'✓' if call.success else '✗'} ({call.duration:.2f}s, cost: ${call.cost:.3f})")
    print()
    
    # Simulate iterations
    for turn in range(3):
        state.increment_turn()
        state.add_to_scratchpad(format_thought_action_observation(
            state, 
            f"Turn {turn + 1}: Planning next step",
            {"name": "analyze_code", "arguments": {"pattern": "state management"}},
            f"Found {turn + 2} instances to refactor"
        ))
    
    print(f"5. Completed {state.turns} turns")
    print(f"   Total cost: ${state.cost:.3f}")
    print()
    
    # Check budget status
    budget_status = get_budget_status(state)
    print("6. Budget status:")
    for resource, info in budget_status.items():
        if resource != "exceeded" and isinstance(info, dict) and 'used' in info:
            print(f"   {resource}: {info['used']}/{info['max']} ({info['remaining']} remaining)")
    print(f"   Budget exceeded: {budget_status['exceeded']}")
    print()
    
    # Complete the task
    state.set_status(AgentStatus.DONE)
    state.add_decision("Successfully refactored to structured state")
    state.complete_todo("Preserve functionality")
    
    # Generate summary
    summary = summarize_execution(state)
    print("7. Execution summary:")
    print(summary)
    print()
    
    # Show final state structure
    print("8. Final state structure (JSON):")
    # Create a simplified version for display
    state_dict = {
        "goal": state.goal,
        "status": state.status.value,
        "turns": state.turns,
        "cost": state.cost,
        "context_items": len(state.context),
        "evidence_items": len(state.evidence),
        "tool_calls": len(state.tool_calls),
        "session_context": state.session_context,
        "session_decisions": state.session_decisions,
        "budget_exceeded": state.is_budget_exceeded()
    }
    print(json.dumps(state_dict, indent=2))


def demo_state_comparison():
    """Show the difference between old and new approaches"""
    print("\n=== Comparison: Old vs New State Management ===\n")
    
    print("OLD APPROACH (ad-hoc, scattered state):")
    print("- TokenBudget class with percentage-based allocations")
    print("- MemoryManager with separate session_decisions, session_todos")
    print("- State scattered across agent instance variables")
    print("- Manual token estimation and truncation")
    print("- No centralized budget enforcement")
    print()
    
    print("NEW APPROACH (structured, centralized state):")
    print("- State dataclass with all agent state in one place")
    print("- Budgets dataclass with resource limits and timeouts")
    print("- ContextBlob and Evidence for structured information tracking")
    print("- ToolCallTrace for detailed tool execution tracking")
    print("- Centralized budget enforcement with enforce_budgets()")
    print("- State operations module for clean separation of concerns")
    print()
    
    print("BENEFITS:")
    print("✓ Clean separation between state and logic")
    print("✓ Type-safe state management with dataclasses")
    print("✓ Easy to test state operations independently")
    print("✓ Better debugging with structured state inspection")
    print("✓ Follows functional programming principles")
    print("✓ Matches the pseudocode requirements exactly")


if __name__ == "__main__":
    demo_state_creation()
    demo_state_comparison()
