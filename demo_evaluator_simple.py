#!/usr/bin/env python3
"""
Simple demo script for the evaluator node system.
Tests the core evaluation functionality without full ReAct agent integration.
"""
import asyncio
import sys
import os
import logging

# Add the server directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

from server.agent.evaluator_node import EvaluatorNode, AcceptanceCriteria, EvaluationResult
from server.agent.state import State, make_state, Budgets

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockLLMClient:
    """Mock LLM client for testing"""
    
    async def complete(self, prompt: str) -> str:
        if "score" in prompt.lower() and "evaluate" in prompt.lower():
            # Mock evaluation response - vary score based on content
            if "comprehensive" in prompt or "detailed" in prompt:
                return """SCORE: 0.9
REASONING: Comprehensive answer with good detail and structure.
STRENGTHS: Thorough coverage, clear explanations, good examples.
WEAKNESSES: Could be more concise in some areas.
SUGGESTIONS: Consider adding summary section."""
            elif "python" in prompt.lower():
                return """SCORE: 0.8
REASONING: Good explanation of Python with relevant details.
STRENGTHS: Accurate information, covers key uses.
WEAKNESSES: Could include more examples.
SUGGESTIONS: Add concrete code examples."""
            else:
                return """SCORE: 0.6
REASONING: Basic answer but lacks depth.
STRENGTHS: Covers basic points.
WEAKNESSES: Too brief, needs more detail.
SUGGESTIONS: Expand with examples and more thorough explanations."""
        elif "improve" in prompt.lower() and "feedback" in prompt.lower():
            # Mock repair response
            return """Python is a high-level, interpreted programming language known for its readable syntax and versatility. 

Key features:
- Easy to learn and use
- Extensive library ecosystem
- Cross-platform compatibility
- Strong community support

Common uses:
- Web development (Django, Flask)
- Data science and machine learning
- Automation and scripting
- Scientific computing

Example:
```python
print("Hello, World!")
```

Python's philosophy emphasizes code readability and simplicity, making it an excellent choice for beginners and experts alike."""
        else:
            # Mock general response
            return "Python is a programming language used for various applications."


async def demo_evaluator_node():
    """Demonstrate the evaluator node functionality"""
    print("=== Demo: Evaluator Node System ===\n")
    
    llm_client = MockLLMClient()
    evaluator = EvaluatorNode(llm_client)
    
    print("1. Testing Basic Answer Evaluation")
    print("-" * 50)
    
    # Test 1: Basic evaluation
    test_answer = "Python is a programming language used for web development and data science."
    criteria = AcceptanceCriteria(
        min_score=0.7,
        required_elements=["python", "programming"],
        prohibited_elements=["difficult", "complex"]
    )
    
    # Create mock state
    state = make_state("Explain what Python is", Budgets())
    state.context = "User asked about Python programming language"
    
    evaluation = await evaluator.evaluate(test_answer, state, criteria)
    
    print(f"   Answer: {test_answer}")
    print(f"   LLM Score: {evaluation.llm_score:.2f}")
    print(f"   Fact Score: {evaluation.fact_score:.2f}")
    print(f"   Overall Score: {evaluation.overall_score:.2f}")
    print(f"   Passed: {evaluation.passed}")
    print(f"   Required elements check: {evaluation.details['required_elements_check']}")
    print()
    
    print("2. Testing Low Score with Repair")
    print("-" * 50)
    
    # Test 2: Low score that triggers repair
    poor_answer = "Python is a thing."
    high_criteria = AcceptanceCriteria(
        min_score=0.8,
        required_elements=["programming", "language", "uses"]
    )
    
    evaluation = await evaluator.evaluate(poor_answer, state, high_criteria)
    print(f"   Original Answer: {poor_answer}")
    print(f"   Original Score: {evaluation.overall_score:.2f}")
    print(f"   Passed: {evaluation.passed}")
    
    if not evaluation.passed:
        print("   Attempting repair...")
        repaired = await evaluator.repair_answer(poor_answer, state, evaluation)
        print(f"   Repaired Answer: {repaired[:100]}...")
        
        # Re-evaluate
        new_eval = await evaluator.evaluate(repaired, state, high_criteria)
        print(f"   New Score: {new_eval.overall_score:.2f}")
        print(f"   Improvement: {new_eval.overall_score - evaluation.overall_score:.2f}")
        print(f"   Now Passes: {new_eval.passed}")
    print()
    
    print("3. Testing Domain-Specific Checks")
    print("-" * 50)
    
    # Test 3: Domain-specific checks
    code_answer = """To implement binary search:
    
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

This algorithm has O(log n) time complexity."""
    
    code_criteria = AcceptanceCriteria(
        min_score=0.6,
        required_elements=["binary", "search"],
        domain_specific_checks={
            "code_quality": {"require_code": True},
            "math_notation": {"require_latex": False}
        }
    )
    
    evaluation = await evaluator.evaluate(code_answer, state, code_criteria)
    print(f"   Answer contains code: {len(code_answer) > 100}")
    print(f"   Score: {evaluation.overall_score:.2f}")
    print(f"   Code quality check: {evaluation.details['domain_checks']['code_quality']}")
    print(f"   Passed: {evaluation.passed}")
    print()
    
    print("4. Testing Prohibited Elements")
    print("-" * 50)
    
    # Test 4: Prohibited elements
    bad_answer = "Python is a very difficult and complex programming language that's hard to learn."
    strict_criteria = AcceptanceCriteria(
        min_score=0.5,
        required_elements=["python"],
        prohibited_elements=["difficult", "complex", "hard"]
    )
    
    evaluation = await evaluator.evaluate(bad_answer, state, strict_criteria)
    print(f"   Answer: {bad_answer}")
    print(f"   Score: {evaluation.overall_score:.2f}")
    print(f"   Prohibited elements found: {evaluation.details['prohibited_elements_check']['found']}")
    print(f"   Passed: {evaluation.passed}")
    print()
    
    print("5. Testing Factuality Checks")
    print("-" * 50)
    
    # Test 5: Factuality edge cases
    uncertain_answer = "Python is likely a programming language that seems to be used for web development."
    overconfident_answer = "Python is definitely the only programming language you'll ever need, 100% guaranteed."
    contradictory_answer = "Python is both easy and difficult, true and false at the same time."
    
    basic_criteria = AcceptanceCriteria(min_score=0.5)
    
    for answer, label in [
        (uncertain_answer, "Uncertain (good)"),
        (overconfident_answer, "Overconfident"),
        (contradictory_answer, "Contradictory")
    ]:
        evaluation = await evaluator.evaluate(answer, state, basic_criteria)
        print(f"   {label}: Fact Score = {evaluation.fact_score:.2f}")
    print()
    
    print("✅ Evaluator Node Demo completed!")
    print("\nKEY FEATURES DEMONSTRATED:")
    print("• LLM-based scoring with structured feedback")
    print("• Factuality checking (stub implementation)")
    print("• Required/prohibited element validation")
    print("• Domain-specific checks (code, math, citations)")
    print("• One-shot answer repair capability")
    print("• Comprehensive evaluation feedback")
    print("• Configurable acceptance criteria")


async def demo_acceptance_criteria():
    """Demonstrate different acceptance criteria configurations"""
    print("\n=== Demo: Acceptance Criteria Configurations ===\n")
    
    # Math-focused criteria
    math_criteria = AcceptanceCriteria(
        min_score=0.8,
        required_elements=["equation", "solve"],
        domain_specific_checks={
            "math_notation": {"require_latex": True}
        }
    )
    print("1. Math Criteria:", math_criteria.__dict__)
    
    # Code-focused criteria  
    code_criteria = AcceptanceCriteria(
        min_score=0.7,
        required_elements=["function", "example"],
        prohibited_elements=["deprecated", "outdated"],
        domain_specific_checks={
            "code_quality": {"require_code": True}
        }
    )
    print("2. Code Criteria:", code_criteria.__dict__)
    
    # Research-focused criteria
    research_criteria = AcceptanceCriteria(
        min_score=0.9,
        required_elements=["evidence", "sources"],
        domain_specific_checks={
            "citation_format": {"require_citations": True}
        }
    )
    print("3. Research Criteria:", research_criteria.__dict__)
    print()


if __name__ == "__main__":
    async def main():
        await demo_evaluator_node()
        await demo_acceptance_criteria()
    
    asyncio.run(main())
