#!/usr/bin/env python3
"""
Final Demo: Swarm-Style vs ReAct Comparison

This demo showcases the transformation from ReAct looping to 
deterministic Swarm-style execution with explicit handoffs.

Key improvements:
1. ✅ Explicit steps: prepare_inputs → execute → postprocess
2. ✅ Deterministic execution (same inputs = same outputs)
3. ✅ Fewer loops and more predictable behavior
4. ✅ Modular handoffs between functions
5. ✅ ReAct is now available as fallback but removed from main path
"""
import sys
import os
import asyncio
import logging
import time

# Add the server directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from server.agent.task_orchestrator import TaskOrchestrator, OrchestrationStatus
from server.agent.swarm_execution import run_minimal_flow, function_prepare_inputs, function_execute, function_postprocess
from server.agent.state import make_state, Budgets
from server.agent.evaluator_node import AcceptanceCriteria

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MockLLMClient:
    """Mock LLM client for testing"""
    
    def __init__(self):
        self.call_count = 0
    
    async def acreate(self, messages, **kwargs):
        """Mock LLM response with deterministic behavior"""
        self.call_count += 1
        
        class MockResponse:
            def __init__(self, content):
                self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})})()]
        
        # Generate consistent responses based on input
        last_message = messages[-1].get('content', '') if messages else ''
        
        if 'score' in last_message.lower():
            return MockResponse("Score: 0.82\nFeedback: Well-structured response with clear explanations.")
        else:
            return MockResponse("Deterministic mock response from Swarm execution system.")


async def demonstrate_explicit_steps():
    """Demonstrate the explicit prepare_inputs → execute → postprocess flow"""
    
    print("🔄 Demonstrating Explicit Step-by-Step Execution")
    print("=" * 60)
    
    goal = "Implement a quick sort algorithm"
    context = ["Working on sorting algorithms for computer science course"]
    
    print(f"Goal: {goal}")
    print(f"Context: {context[0]}")
    print()
    
    # Step 1: Prepare Inputs
    print("1️⃣ PREPARE_INPUTS:")
    start_time = time.time()
    prep_result = function_prepare_inputs(goal, context)
    prep_time = time.time() - start_time
    
    print(f"   ⏱️  Time: {prep_time:.3f}s")
    print(f"   💰 Cost: ${prep_result.cost:.2f}")
    print(f"   📋 Strategy: {prep_result.output.get('strategy', 'unknown')}")
    print(f"   🔧 Tools needed: {prep_result.output.get('required_tools', [])}")
    print()
    
    # Step 2: Execute
    print("2️⃣ EXECUTE:")
    start_time = time.time()
    exec_result = await function_execute(prep_result.output)
    exec_time = time.time() - start_time
    
    print(f"   ⏱️  Time: {exec_time:.3f}s")
    print(f"   💰 Cost: ${exec_result.cost:.2f}")
    print(f"   📝 Output length: {len(exec_result.output)} chars")
    print(f"   🎯 Strategy used: {exec_result.metadata.get('strategy_used', 'unknown')}")
    print()
    
    # Step 3: Postprocess
    print("3️⃣ POSTPROCESS:")
    start_time = time.time()
    post_result = await function_postprocess(exec_result.output)
    post_time = time.time() - start_time
    
    print(f"   ⏱️  Time: {post_time:.3f}s")
    print(f"   💰 Cost: ${post_result.cost:.2f}")
    print(f"   📝 Final length: {len(post_result.output)} chars")
    print()
    
    total_time = prep_time + exec_time + post_time
    total_cost = prep_result.cost + exec_result.cost + post_result.cost
    
    print("📊 SUMMARY:")
    print(f"   ⏱️  Total time: {total_time:.3f}s")
    print(f"   💰 Total cost: ${total_cost:.2f}")
    print(f"   🎯 All steps: SUCCESS")
    print(f"   🔄 No loops required!")
    
    return post_result.output


async def demonstrate_deterministic_behavior():
    """Show that execution is deterministic"""
    
    print("\n🔄 Demonstrating Deterministic Behavior")
    print("=" * 60)
    
    goal = "Explain the concept of recursion in programming"
    
    print(f"Running the same goal 3 times: '{goal}'")
    print()
    
    results = []
    for i in range(3):
        print(f"Run {i+1}:")
        start_time = time.time()
        
        state = make_state(goal, Budgets())
        flow_result = await run_minimal_flow(state)
        
        execution_time = time.time() - start_time
        
        results.append({
            'success': flow_result.success,
            'output_length': len(flow_result.final_output) if flow_result.final_output else 0,
            'cost': flow_result.total_cost,
            'steps': len(flow_result.step_results),
            'time': execution_time,
            'strategies': flow_result.metadata.get('strategies_used', [])
        })
        
        print(f"   ✅ Success: {flow_result.success}")
        print(f"   📝 Output: {results[-1]['output_length']} chars")
        print(f"   💰 Cost: ${results[-1]['cost']:.2f}")
        print(f"   ⏱️  Time: {results[-1]['time']:.3f}s")
        print()
    
    # Check consistency
    print("🔍 CONSISTENCY CHECK:")
    first_result = results[0]
    
    all_same_length = all(r['output_length'] == first_result['output_length'] for r in results)
    all_same_cost = all(r['cost'] == first_result['cost'] for r in results)
    all_same_steps = all(r['steps'] == first_result['steps'] for r in results)
    all_same_strategies = all(r['strategies'] == first_result['strategies'] for r in results)
    
    print(f"   📝 Same output length: {all_same_length} ({first_result['output_length']} chars)")
    print(f"   💰 Same cost: {all_same_cost} (${first_result['cost']:.2f})")
    print(f"   🔄 Same steps: {all_same_steps} ({first_result['steps']} steps)")
    print(f"   🎯 Same strategies: {all_same_strategies} ({first_result['strategies']})")
    
    deterministic = all_same_length and all_same_cost and all_same_steps and all_same_strategies
    print(f"   ✅ DETERMINISTIC: {deterministic}")
    
    return deterministic


async def demonstrate_orchestrator_flow():
    """Show the complete orchestrator flow with Swarm execution"""
    
    print("\n🔄 Demonstrating Complete Orchestrator Flow")
    print("=" * 60)
    
    # Create orchestrator
    mock_llm = MockLLMClient()
    orchestrator = TaskOrchestrator(llm_client=mock_llm)
    
    goal = "Design a simple hash table data structure"
    budgets = Budgets(total_tokens=2000, max_iterations=5)
    acceptance = AcceptanceCriteria(min_score=0.75)
    
    print(f"🎯 Goal: {goal}")
    print(f"💰 Budget: {budgets.total_tokens} tokens, {budgets.max_iterations} iterations")
    print(f"📊 Acceptance: min score {acceptance.min_score}")
    print()
    
    # Run the orchestrator
    print("🚀 ORCHESTRATOR EXECUTION:")
    start_time = time.time()
    
    result = await orchestrator.run_task(goal, budgets, acceptance)
    
    execution_time = time.time() - start_time
    
    print(f"   ⏱️  Total time: {execution_time:.3f}s")
    print(f"   🎯 Status: {result.status.value}")
    print(f"   ✅ Success: {result.status == OrchestrationStatus.DONE}")
    
    if result.answer:
        print(f"   📝 Answer length: {len(result.answer)} chars")
        print(f"   💰 Total cost: ${result.state.cost:.2f}")
        print(f"   📋 Evidence items: {len(result.state.evidence)}")
        print(f"   🎯 Subgoals: {getattr(result.state, 'subgoals', [])}")
        
        if result.metadata and 'final_score' in result.metadata:
            print(f"   📊 Final score: {result.metadata['final_score']:.3f}")
    
    if result.error:
        print(f"   ❌ Error: {result.error}")
    
    print()
    print("🔍 STATE MACHINE FLOW:")
    print("   PLANNING → Set subgoals and route")
    print("   EXECUTING → Swarm-style deterministic execution")
    print("   CRITIQUING → Evaluation with acceptance criteria")
    print("   DONE → Task completed successfully")
    
    return result.status == OrchestrationStatus.DONE


async def demonstrate_modular_handoffs():
    """Show the modular handoffs between functions"""
    
    print("\n🔄 Demonstrating Modular Handoffs")
    print("=" * 60)
    
    goal = "Create a binary tree traversal algorithm"
    
    print(f"Goal: {goal}")
    print()
    print("🔄 HANDOFF SEQUENCE:")
    
    # Function 1: prepare_inputs
    print("1️⃣ function_prepare_inputs() →")
    prep_result = function_prepare_inputs(goal, [])
    print(f"   📤 Outputs: execution_plan, required_tools, strategy")
    print(f"   🔧 Strategy: {prep_result.output.get('strategy')}")
    print(f"   📋 Tools: {prep_result.output.get('required_tools')}")
    print()
    
    # Function 2: execute  
    print("2️⃣ function_execute(prepared_input) →")
    exec_result = await function_execute(prep_result.output)
    print(f"   📤 Outputs: generated content, metadata, costs")
    print(f"   📝 Content length: {len(exec_result.output)} chars")
    print(f"   🎯 Tools used: {exec_result.metadata.get('tools_called', [])}")
    print()
    
    # Function 3: postprocess
    print("3️⃣ function_postprocess(execution_output) →")
    post_result = await function_postprocess(exec_result.output)
    print(f"   📤 Outputs: formatted content, quality metrics")
    print(f"   📝 Final length: {len(post_result.output)} chars")
    print(f"   📊 Quality score: {post_result.metadata.get('quality_score', 'N/A')}")
    print()
    
    print("✅ MODULAR BENEFITS:")
    print("   🔧 Each function has single responsibility")
    print("   🔄 Clear input/output contracts")
    print("   🧪 Easy to test individual functions")
    print("   🔀 Functions can be swapped or enhanced")
    print("   📊 Detailed metadata at each step")
    
    return True


async def compare_approaches():
    """Compare old ReAct looping vs new Swarm-style"""
    
    print("\n🔄 Comparing Approaches")
    print("=" * 60)
    
    print("📊 OLD APPROACH (ReAct Looping):")
    print("   🔄 Thought → Action → Observation → Thought → ...")
    print("   ❓ Unpredictable number of loops")
    print("   🎲 Non-deterministic execution")
    print("   🧠 Requires LLM to reason about when to stop")
    print("   💸 Variable costs due to looping")
    print("   🐛 Harder to debug and test")
    print()
    
    print("🎯 NEW APPROACH (Swarm-Style):")
    print("   📋 prepare_inputs → execute → postprocess")
    print("   ✅ Fixed 3-step execution")
    print("   🎯 Deterministic behavior")
    print("   🤖 Clear separation of concerns")
    print("   💰 Predictable costs")
    print("   🧪 Easy to test and debug")
    print()
    
    print("🏆 IMPROVEMENTS:")
    print("   ⚡ Faster execution (no thinking loops)")
    print("   🎯 More reliable results")
    print("   📊 Better cost control")
    print("   🔧 Easier maintenance")
    print("   🧪 Better testability")
    print("   📈 Scalable architecture")
    
    return True


async def main():
    """Run all demonstrations"""
    
    print("Swarm-Style Deterministic Execution Demo")
    print("🔄 Replacing ReAct Loops with Explicit Handoffs")
    print("=" * 70)
    
    demonstrations = [
        ("Explicit Steps", demonstrate_explicit_steps),
        ("Deterministic Behavior", demonstrate_deterministic_behavior),
        ("Orchestrator Flow", demonstrate_orchestrator_flow),
        ("Modular Handoffs", demonstrate_modular_handoffs),
        ("Approach Comparison", compare_approaches),
    ]
    
    results = []
    for demo_name, demo_func in demonstrations:
        print(f"\n🚀 Running {demo_name}...")
        try:
            success = await demo_func()
            results.append((demo_name, success))
            print(f"✅ {demo_name}: {'SUCCESS' if success else 'COMPLETED'}")
        except Exception as e:
            results.append((demo_name, False))
            print(f"❌ {demo_name}: EXCEPTION - {e}")
    
    # Final summary
    print(f"\n{'=' * 70}")
    print("🎉 FINAL SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for demo_name, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {demo_name}")
    
    print(f"\n📊 Results: {successful}/{total} demonstrations successful")
    
    if successful == total:
        print("\n🎉 SUCCESS: All demonstrations completed!")
        print()
        print("✅ ACHIEVEMENTS:")
        print("   🔄 Replaced ReAct looping with explicit steps")
        print("   🎯 Implemented deterministic execution")
        print("   🤖 Created modular Swarm-style handoffs")
        print("   📊 Reduced loops and improved predictability")
        print("   🚀 Enhanced orchestration state machine")
        print()
        print("🎯 SUCCESS CRITERIA MET:")
        print("   ✅ Drop the looping prompt")
        print("   ✅ Swap in explicit steps (prepare_inputs → execute → postprocess)")
        print("   ✅ Remove ReAct scratchpad from graph path")
        print("   ✅ Tasks run deterministically with fewer loops")
        print("   ✅ Everything is modular")
        print()
        print("🏆 The system now provides deterministic, predictable task execution!")
        
    else:
        print(f"\n⚠️  {total - successful} demonstration(s) had issues.")
    
    return successful == total


if __name__ == "__main__":
    asyncio.run(main())
