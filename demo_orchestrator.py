#!/usr/bin/env python3
"""
Demo script for the Task Orchestrator state machine.
Tests the PLANNING → EXECUTING → CRITIQUING → DONE/FAILED flow.
"""
import asyncio
import sys
import os
import logging

# Add the server directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

from server.agent.task_orchestrator import (
    TaskOrchestrator, 
    OrchestrationStatus,
    run_minimal_task,
    run_task_with_evaluation
)
from server.agent.evaluator_node import AcceptanceCriteria
from server.agent.state import Budgets

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockLLMClient:
    """Mock LLM client for testing orchestrator"""
    
    async def complete(self, prompt: str) -> str:
        if "score" in prompt.lower() and "evaluate" in prompt.lower():
            # Extract content being evaluated
            if "ANSWER TO EVALUATE:" in prompt:
                lines = prompt.split('\n')
                answer_section = False
                answer_lines = []
                
                for line in lines:
                    if "ANSWER TO EVALUATE:" in line:
                        answer_section = True
                        continue
                    elif answer_section and line.strip():
                        answer_lines.append(line.strip())
                    elif answer_section and not line.strip():
                        break
                        
                answer = " ".join(answer_lines)
                
                # Score based on content quality
                if len(answer) > 100 and ("python" in answer.lower() or "example" in answer.lower()):
                    score = 0.8
                elif len(answer) > 50:
                    score = 0.6
                else:
                    score = 0.4
                
                return f"""SCORE: {score}
REASONING: Evaluated based on content quality and length.
STRENGTHS: Provides relevant information
WEAKNESSES: Could be more comprehensive
SUGGESTIONS: Add more examples and details"""
            
            return """SCORE: 0.7
REASONING: Standard evaluation
STRENGTHS: Covers basic points
WEAKNESSES: Needs more detail
SUGGESTIONS: Expand with examples"""
        
        return "Mock LLM response for non-evaluation prompt."


class MockReActAgent:
    """Mock ReAct agent for testing without full integration"""
    
    def __init__(self, api_base, model_name, user_id, default_budgets):
        self.api_base = api_base
        self.model_name = model_name
        self.user_id = user_id
        self.default_budgets = default_budgets
    
    def react_run_with_state(self, goal, budgets):
        """Mock react_run_with_state that returns a state with context"""
        from server.agent.state import make_state
        
        state = make_state(goal, budgets, self.user_id)
        
        # Generate mock response based on goal
        if "python" in goal.lower():
            mock_answer = f"""Python is a high-level programming language known for its simplicity and readability. 

Key features:
- Easy to learn syntax
- Extensive library ecosystem
- Cross-platform compatibility
- Dynamic typing

Common uses:
- Web development
- Data science
- Machine learning
- Automation

Example:
```python
print("Hello, World!")
```

Python's philosophy emphasizes code readability and developer productivity."""
        
        elif "algorithm" in goal.lower() or "search" in goal.lower():
            mock_answer = f"""Here's a binary search algorithm implementation:

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

This algorithm has O(log n) time complexity and works on sorted arrays."""
        
        else:
            mock_answer = f"This is a comprehensive answer to the question: {goal}. It covers the main points and provides relevant information."
        
        # Add the answer to state context
        state.context = f"Final Answer: {mock_answer}"
        state.cost = 150  # Mock token cost
        
        return state


async def demo_orchestrator():
    """Demonstrate the task orchestrator state machine"""
    print("=== Demo: Task Orchestrator State Machine ===\n")
    
    llm_client = MockLLMClient()
    
    # Temporarily monkey patch to use mock ReAct agent
    import server.agent.task_orchestrator
    original_react_agent = server.agent.task_orchestrator.ReActAgent
    server.agent.task_orchestrator.ReActAgent = MockReActAgent
    
    try:
        print("1. Testing MINIMAL_FLOW: High Score (DONE)")
        print("-" * 60)
        
        # Test 1: High score scenario (should reach DONE)
        result = await run_task_with_evaluation(
            goal="Explain Python programming language with examples",
            min_score=0.7,
            required_elements=["python", "programming"],
            llm_client=llm_client
        )
        
        print(f"   Goal: {result.state.goal}")
        print(f"   Status: {result.status.value}")
        print(f"   Answer Length: {len(result.answer) if result.answer else 0}")
        print(f"   Final Score: {result.metadata.get('final_score', 'N/A')}")
        print(f"   Iterations: {result.metadata.get('iteration', 'N/A')}")
        print(f"   Route: {getattr(result.state, 'route', 'N/A')}")
        print(f"   Subgoals: {getattr(result.state, 'subgoals', 'N/A')}")
        print(f"   Evidence Count: {len(result.state.evidence) if result.state.evidence else 0}")
        print()
        
        print("2. Testing MINIMAL_FLOW: Low Score (FAILED)")
        print("-" * 60)
        
        # Test 2: Low score scenario (should reach FAILED)
        result = await run_task_with_evaluation(
            goal="Short answer",  # This will generate a short, low-quality answer
            min_score=0.9,  # Very high threshold
            llm_client=llm_client
        )
        
        print(f"   Goal: {result.state.goal}")
        print(f"   Status: {result.status.value}")
        print(f"   Error: {result.error}")
        print(f"   Final Score: {result.metadata.get('final_score', 'N/A')}")
        print(f"   Threshold: {result.metadata.get('threshold', 'N/A')}")
        print(f"   Iterations: {result.metadata.get('iteration', 'N/A')}")
        print()
        
        print("3. Testing with Custom Orchestrator")
        print("-" * 60)
        
        # Test 3: Custom orchestrator with specific criteria
        orchestrator = TaskOrchestrator(llm_client=llm_client)
        
        acceptance = AcceptanceCriteria(
            min_score=0.6,
            required_elements=["algorithm", "example"],
            domain_specific_checks={"code_quality": {"require_code": True}}
        )
        
        budgets = Budgets(max_iterations=3, total_tokens=2048)
        
        result = await orchestrator.run_task(
            goal="Implement a binary search algorithm",
            budgets=budgets,
            acceptance=acceptance,
            user_id="demo_user"
        )
        
        print(f"   Goal: {result.state.goal}")
        print(f"   Status: {result.status.value}")
        print(f"   Answer Preview: {result.answer[:100] if result.answer else 'None'}...")
        print(f"   Cost: {getattr(result.state, 'cost', 'N/A')}")
        print(f"   User ID: {result.state.user_id}")
        print(f"   State Machine Route: {getattr(result.state, 'route', 'N/A')}")
        print()
        
        print("4. Testing State Transitions")
        print("-" * 60)
        
        # Test 4: Trace state transitions
        result = await run_minimal_task(
            goal="Explain machine learning basics",
            llm_client=llm_client
        )
        
        print("   State Machine Flow:")
        print("   PLANNING → Set subgoals and route")
        print("   EXECUTING → Run ReAct agent")
        print("   CRITIQUING → Evaluate answer")
        print(f"   {result.status.value.upper()} → Final result")
        print()
        print(f"   Final Status: {result.status.value}")
        print(f"   Evidence Added: {len(result.state.evidence)} entries")
        if result.state.evidence:
            entry = result.state.evidence[0]
            print(f"   Evidence Type: {entry.type}")
            print(f"   Evidence Summary: {entry.metadata.get('summary', entry.content[:50])}...")
        print()
        
        print("✅ Orchestrator Demo Completed!")
        print("\nKEY FEATURES DEMONSTRATED:")
        print("• Linear state machine: PLANNING → EXECUTING → CRITIQUING → DONE/FAILED")
        print("• Integration with existing ReAct agent in EXECUTING state")
        print("• Evaluation-based decision making in CRITIQUING state")
        print("• State preservation and evidence collection")
        print("• Route and subgoal tracking")
        print("• Configurable acceptance criteria")
        print("• Error handling and metadata collection")
        
    finally:
        # Restore original ReAct agent
        server.agent.task_orchestrator.ReActAgent = original_react_agent


async def demo_state_machine_details():
    """Show detailed state machine behavior"""
    print("\n=== Demo: State Machine Internal Details ===\n")
    
    print("State Machine Definition:")
    print("  PLANNING:")
    print("    → Set S.subgoals = ['use current ReAct']")
    print("    → Set S.route = MINIMAL_FLOW")
    print("    → Transition to EXECUTING")
    print()
    print("  EXECUTING:")
    print("    → react := react_run_with_state(goal, budgets)")
    print("    → S.cost += react.state.cost")
    print("    → S.evidence.push({kind:DOC, ref:save(react.answer), summary:head(react.answer,200)})")
    print("    → tmp_answer := react.answer")
    print("    → Transition to CRITIQUING")
    print()
    print("  CRITIQUING:")
    print("    → score := evaluator_node(tmp_answer, S, acceptance)")
    print("    → if score >= acceptance.min_score: S.status = DONE")
    print("    → else: S.status = FAILED")
    print()
    
    print("Orchestration vs ReAct Comparison:")
    print("  OLD: Direct ReAct agent call")
    print("  NEW: ReAct wrapped in orchestration state machine")
    print("  BENEFIT: Foundation for complex multi-node workflows")
    print("  RESULT: Same answers, but through structured graph execution")


if __name__ == "__main__":
    async def main():
        await demo_orchestrator()
        await demo_state_machine_details()
    
    asyncio.run(main())
