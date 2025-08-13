#!/usr/bin/env python3
"""
Simple demo for the Task Orchestrator without ReAct agent imports.
Tests the state machine logic and structure.
"""
import asyncio
import sys
import os
import logging

# Add the server directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

from server.agent.evaluator_node import AcceptanceCriteria, EvaluatorNode
from server.agent.state import State, make_state, Budgets, AgentStatus, Evidence

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockLLMClient:
    """Mock LLM client for testing"""
    
    async def complete(self, prompt: str) -> str:
        if "score" in prompt.lower() and "evaluate" in prompt.lower():
            # Extract content for realistic scoring
            if "comprehensive" in prompt or len(prompt) > 1000:
                return """SCORE: 0.8
REASONING: Good comprehensive answer with relevant details.
STRENGTHS: Covers key points, good structure
WEAKNESSES: Could add more examples
SUGGESTIONS: Include concrete examples"""
            else:
                return """SCORE: 0.6
REASONING: Basic answer with room for improvement.
STRENGTHS: Addresses the question
WEAKNESSES: Lacks detail and examples
SUGGESTIONS: Expand with more comprehensive information"""
        
        return "Mock response for non-evaluation prompt."


class SimpleOrchestrator:
    """Simplified orchestrator for testing without full dependencies"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def run_task(self, goal: str, budgets: Budgets, acceptance: AcceptanceCriteria, user_id: str = "test_user"):
        """Run the state machine following the exact pseudocode"""
        
        logger.info(f"Starting orchestration for: {goal}")
        
        # Initialize state
        S = make_state(goal, budgets, user_id)
        S.status = AgentStatus.PLANNING
        
        # Track orchestration status separately
        orchestration_status = "PLANNING"
        tmp_answer = None
        
        # State machine loop
        iteration = 0
        while iteration < 5:  # Safety limit
            iteration += 1
            logger.info(f"Iteration {iteration}: {orchestration_status}")
            
            if orchestration_status == "PLANNING":
                # Exact pseudocode implementation
                S.subgoals = ["use current ReAct"]
                S.route = "MINIMAL_FLOW"
                orchestration_status = "EXECUTING"
                logger.info("PLANNING: Set subgoals and route, transitioning to EXECUTING")
                
            elif orchestration_status == "EXECUTING":
                # Mock the ReAct call
                mock_react_result = self._mock_react_call(goal, budgets, user_id)
                
                # Update state as per pseudocode
                S.cost += getattr(mock_react_result, 'cost', 100)
                
                # Extract answer
                tmp_answer = self._extract_answer(mock_react_result)
                
                # Add evidence as specified in pseudocode
                if tmp_answer:
                    evidence = Evidence(
                        type="DOC",
                        content=tmp_answer,
                        source="react_agent", 
                        metadata={"ref": "react_answer", "summary": tmp_answer[:200]}
                    )
                    S.evidence.append(evidence)
                
                orchestration_status = "CRITIQUING"
                logger.info(f"EXECUTING: Generated answer ({len(tmp_answer)} chars), transitioning to CRITIQUING")
                
            elif orchestration_status == "CRITIQUING":
                # Evaluate using evaluator_node
                evaluator = EvaluatorNode(self.llm_client)
                evaluation = await evaluator.evaluate(tmp_answer, S, acceptance)
                score = evaluation.overall_score
                
                logger.info(f"CRITIQUING: Score {score:.2f}, threshold {acceptance.min_score}")
                
                # Decision as per pseudocode
                if score >= acceptance.min_score:
                    S.status = AgentStatus.DONE
                    return {"answer": tmp_answer, "state": S, "status": "DONE", "score": score}
                else:
                    S.status = AgentStatus.FAILED
                    return {"error": "low_score", "state": S, "status": "FAILED", "score": score}
                    
            else:
                break
        
        return {"error": "max_iterations", "state": S, "status": "FAILED"}
    
    def _mock_react_call(self, goal: str, budgets: Budgets, user_id: str):
        """Mock ReAct agent call"""
        state = make_state(goal, budgets, user_id)
        
        # Generate different answers based on goal
        if "python" in goal.lower():
            answer = """Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. 

Key Features:
- Simple, readable syntax
- Dynamic typing
- Extensive standard library
- Cross-platform compatibility
- Large ecosystem of third-party packages

Common Applications:
- Web development (Django, Flask)
- Data science and analysis (pandas, NumPy)
- Machine learning (TensorFlow, scikit-learn)
- Automation and scripting
- Scientific computing

Example Code:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
```

Python's philosophy emphasizes code readability and developer productivity, making it an excellent choice for both beginners and experienced programmers."""

        elif "algorithm" in goal.lower():
            answer = """Binary search is an efficient algorithm for finding a target value in a sorted array.

Algorithm Steps:
1. Compare target with middle element
2. If equal, return the index
3. If target is smaller, search left half
4. If target is larger, search right half
5. Repeat until found or array is empty

Implementation:
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

Time Complexity: O(log n)
Space Complexity: O(1)"""

        else:
            answer = f"This is a comprehensive answer addressing: {goal}. It provides relevant information and covers the key aspects of the topic."
        
        state.context = [f"Final Answer: {answer}"]
        state.cost = 120
        return state
    
    def _extract_answer(self, react_state):
        """Extract answer from mock ReAct state"""
        if hasattr(react_state, 'context') and react_state.context:
            context_str = str(react_state.context[0]) if isinstance(react_state.context, list) else str(react_state.context)
            if "Final Answer:" in context_str:
                return context_str.split("Final Answer:", 1)[1].strip()
        return "Mock answer extracted from ReAct state"


async def demo_simple_orchestrator():
    """Demo the simplified orchestrator"""
    print("=== Simple Task Orchestrator Demo ===\n")
    
    llm_client = MockLLMClient()
    orchestrator = SimpleOrchestrator(llm_client)
    
    test_cases = [
        {
            "name": "High Score Test (DONE)",
            "goal": "Explain Python programming with examples",
            "acceptance": AcceptanceCriteria(min_score=0.7, required_elements=["python"])
        },
        {
            "name": "Low Score Test (FAILED)",
            "goal": "Short question",
            "acceptance": AcceptanceCriteria(min_score=0.9)  # Very high threshold
        },
        {
            "name": "Algorithm Test",
            "goal": "Implement binary search algorithm",
            "acceptance": AcceptanceCriteria(
                min_score=0.6,
                required_elements=["algorithm"],
                domain_specific_checks={"code_quality": {"require_code": True}}
            )
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}")
        print("-" * 50)
        
        budgets = Budgets(total_tokens=2048, max_iterations=3)
        result = await orchestrator.run_task(
            test_case["goal"],
            budgets,
            test_case["acceptance"]
        )
        
        print(f"   Goal: {test_case['goal']}")
        print(f"   Status: {result['status']}")
        
        if result['status'] == 'DONE':
            print(f"   Answer Length: {len(result['answer'])}")
            print(f"   Score: {result['score']:.2f}")
            print(f"   Subgoals: {result['state'].subgoals}")
            print(f"   Route: {result['state'].route}")
            print(f"   Evidence Count: {len(result['state'].evidence)}")
            print(f"   Cost: {result['state'].cost}")
        else:
            print(f"   Error: {result['error']}")
            if 'score' in result:
                print(f"   Score: {result['score']:.2f}")
                print(f"   Threshold: {test_case['acceptance'].min_score}")
        
        print()
    
    print("✅ State Machine Flow Demonstrated!")
    print("\nState Transitions Verified:")
    print("1. PLANNING → Set subgoals=['use current ReAct'], route='MINIMAL_FLOW'")
    print("2. EXECUTING → Call ReAct, update cost, add evidence")
    print("3. CRITIQUING → Evaluate answer, decide DONE/FAILED")
    print("\nKey Features:")
    print("• Linear state machine implementation")
    print("• Evidence collection and state updates")
    print("• Integration with evaluator_node")
    print("• Exact pseudocode implementation")


async def demo_state_structure():
    """Show the enhanced state structure"""
    print("\n=== Enhanced State Structure ===\n")
    
    # Create a sample state
    state = make_state("Test goal", Budgets(), "test_user")
    state.subgoals = ["use current ReAct"]
    state.route = "MINIMAL_FLOW"
    
    evidence = Evidence(
        type="DOC",
        content="Sample answer content",
        source="react_agent",
        metadata={"ref": "react_answer", "summary": "Sample answer content"}
    )
    state.evidence.append(evidence)
    
    print("State Structure:")
    print(f"  Goal: {state.goal}")
    print(f"  Status: {state.status}")
    print(f"  Subgoals: {state.subgoals}")
    print(f"  Route: {state.route}")
    print(f"  Evidence: {len(state.evidence)} entries")
    print(f"  Cost: {state.cost}")
    print(f"  User ID: {state.user_id}")
    print()
    
    print("Evidence Entry:")
    if state.evidence:
        entry = state.evidence[0]
        print(f"  Type: {entry.type}")
        print(f"  Source: {entry.source}")
        print(f"  Content: {entry.content[:50]}...")
        print(f"  Metadata: {entry.metadata}")


if __name__ == "__main__":
    async def main():
        await demo_simple_orchestrator()
        await demo_state_structure()
    
    asyncio.run(main())
