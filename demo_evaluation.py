#!/usr/bin/env python3
"""
Demo script for the evaluation-gated ReAct agent system.
Tests react_with_eval with different acceptance criteria and scenarios.
"""
import asyncio
import sys
import os
import logging

# Add the server directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

from server.agent.react_with_eval import react_with_eval, react_with_basic_eval
from server.agent.evaluator_node import AcceptanceCriteria, EvaluatorNode
from server.agent.state import Budgets
from server.agent.models import LLMClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_evaluation_gated_react():
    """Demonstrate the evaluation-gated ReAct agent system"""
    print("=== Demo: Evaluation-Gated ReAct Agent ===\n")
    
    # Mock LLM client for demonstration
    # In real usage, this would be your actual LLM client
    class MockLLMClient:
        async def complete(self, prompt: str) -> str:
            if "score" in prompt.lower() and "evaluate" in prompt.lower():
                # Mock evaluation response
                return """SCORE: 0.8
REASONING: The answer provides relevant information and addresses the question adequately.
STRENGTHS: Clear structure, relevant content, appropriate length.
WEAKNESSES: Could include more specific examples.
SUGGESTIONS: Add concrete examples to illustrate key points."""
            elif "improve" in prompt.lower() and "feedback" in prompt.lower():
                # Mock repair response
                return "This is an improved answer that addresses the feedback by including more specific examples and clearer explanations."
            else:
                # Mock general response
                return "This is a sample answer to demonstrate the evaluation system."
    
    llm_client = MockLLMClient()
    
    print("1. Testing Basic Evaluation (High Score - No Repair Needed)")
    print("-" * 60)
    
    # Test 1: High score scenario
    try:
        result = await react_with_basic_eval(
            goal="Explain what Python is",
            llm_client=llm_client,
            min_score=0.7,
            required_elements=["python", "programming"],
            user_id="demo_user"
        )
        
        print(f"   Goal: {result.state.goal}")
        print(f"   Answer: {result.answer[:100]}...")
        print(f"   Score: {result.score_after:.2f}")
        print(f"   Evaluation attempted: {result.evaluation_attempted}")
        print(f"   Repair attempted: {result.repair_attempted}")
        print()
        
    except Exception as e:
        print(f"   ❌ Test 1 failed: {e}")
        print()
    
    print("2. Testing Low Score Scenario (Triggers Repair)")
    print("-" * 60)
    
    # Test 2: Low score scenario that would trigger repair
    try:
        acceptance = AcceptanceCriteria(
            min_score=0.9,  # Very high threshold to trigger repair
            required_elements=["detailed", "comprehensive", "examples"],
            prohibited_elements=["unclear", "vague"]
        )
        
        budgets = Budgets(max_iterations=3, total_tokens=2048)
        
        result = await react_with_eval(
            goal="Provide a comprehensive guide to machine learning",
            budgets=budgets,
            acceptance=acceptance,
            llm_client=llm_client,
            user_id="demo_user"
        )
        
        print(f"   Goal: {result.state.goal}")
        print(f"   Answer: {result.answer[:100]}...")
        print(f"   Final Score: {result.score_after:.2f}")
        print(f"   Original Score: {result.original_score}")
        print(f"   Evaluation attempted: {result.evaluation_attempted}")
        print(f"   Repair attempted: {result.repair_attempted}")
        if result.evaluation_feedback:
            print(f"   Feedback preview: {result.evaluation_feedback[:150]}...")
        print()
        
    except Exception as e:
        print(f"   ❌ Test 2 failed: {e}")
        print()
    
    print("3. Testing Domain-Specific Acceptance Criteria")
    print("-" * 60)
    
    # Test 3: Domain-specific criteria
    try:
        acceptance = AcceptanceCriteria(
            min_score=0.6,
            required_elements=["code", "example"],
            domain_specific_checks={
                "code_quality": {"require_code": True},
                "math_notation": {"require_latex": False}
            }
        )
        
        result = await react_with_eval(
            goal="Show how to implement a binary search algorithm",
            budgets=Budgets(),
            acceptance=acceptance,
            llm_client=llm_client,
            user_id="demo_user"
        )
        
        print(f"   Goal: {result.state.goal}")
        print(f"   Answer: {result.answer[:100]}...")
        print(f"   Score: {result.score_after:.2f}")
        print(f"   Domain checks passed: {result.evaluation_attempted}")
        print()
        
    except Exception as e:
        print(f"   ❌ Test 3 failed: {e}")
        print()
    
    print("4. Testing Direct Evaluator Node")
    print("-" * 60)
    
    # Test 4: Direct evaluator usage
    try:
        evaluator = EvaluatorNode(llm_client)
        
        test_output = "Python is a programming language. It's used for web development, data science, and automation."
        test_criteria = AcceptanceCriteria(
            min_score=0.5,
            required_elements=["python", "programming"],
            prohibited_elements=["difficult", "complex"]
        )
        
        # Mock state for testing
        from server.agent.state import make_state
        mock_state = make_state("Test goal", Budgets())
        mock_state.context = "Context about Python programming language"
        
        evaluation = await evaluator.evaluate(test_output, mock_state, test_criteria)
        
        print(f"   Test Output: {test_output}")
        print(f"   LLM Score: {evaluation.llm_score:.2f}")
        print(f"   Fact Score: {evaluation.fact_score:.2f}")
        print(f"   Overall Score: {evaluation.overall_score:.2f}")
        print(f"   Passed: {evaluation.passed}")
        print(f"   Required elements check: {evaluation.details.get('required_elements_check', {})}")
        print()
        
    except Exception as e:
        print(f"   ❌ Test 4 failed: {e}")
        print()
    
    print("5. Testing Answer Repair Process")
    print("-" * 60)
    
    # Test 5: Answer repair
    try:
        evaluator = EvaluatorNode(llm_client)
        
        poor_answer = "Python is a thing."
        criteria = AcceptanceCriteria(min_score=0.8, required_elements=["programming", "language", "uses"])
        
        from server.agent.state import make_state
        state = make_state("Explain Python programming language", Budgets())
        state.context = "User asked about Python programming language"
        
        # Evaluate the poor answer
        evaluation = await evaluator.evaluate(poor_answer, state, criteria)
        print(f"   Original Answer: {poor_answer}")
        print(f"   Original Score: {evaluation.overall_score:.2f}")
        
        # Attempt repair
        if not evaluation.passed:
            repaired = await evaluator.repair_answer(poor_answer, state, evaluation)
            print(f"   Repaired Answer: {repaired[:100]}...")
            
            # Re-evaluate
            new_eval = await evaluator.evaluate(repaired, state, criteria)
            print(f"   New Score: {new_eval.overall_score:.2f}")
            print(f"   Improvement: {new_eval.overall_score - evaluation.overall_score:.2f}")
        
    except Exception as e:
        print(f"   ❌ Test 5 failed: {e}")
        print()
    
    print("✅ Demo completed!")
    print("\nKEY FEATURES DEMONSTRATED:")
    print("• Automatic evaluation of ReAct agent outputs")
    print("• Configurable acceptance criteria with thresholds")
    print("• One-shot self-repair for below-threshold answers")
    print("• Domain-specific checks (code, math, citations)")
    print("• Factuality scoring (stub implementation)")
    print("• Comprehensive evaluation feedback")
    print("• Integration with existing ReAct agent workflow")


if __name__ == "__main__":
    asyncio.run(demo_evaluation_gated_react())
