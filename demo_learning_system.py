#!/usr/bin/env python3
"""
Example demonstrating the Anton agent learning system.

This script shows how the learning system identifies insights from agent interactions
and uses them to enhance future agent performance through memory injection.
"""

import asyncio
import logging
from server.agent.learning_identifier import learning_identifier
from server.agent.rag_manager import rag_manager
from client.context_builder import ContextBuilder

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


async def demonstrate_learning_workflow():
    """Demonstrate the complete learning workflow with realistic examples."""
    
    print("🧠 Anton Agent Learning System Demonstration")
    print("=" * 50)
    
    # Scenario 1: Agent discovers system architecture
    print("\n📚 Scenario 1: Learning about system architecture")
    print("-" * 40)
    
    task = "Explore the codebase to understand how the agent system works"
    doer_response = """
    I successfully explored the codebase and discovered several key insights:
    
    1. The system uses a planner-doer architecture where the planner breaks down tasks
       and the doer executes them using available tools.
    
    2. The organizer.py file contains the main orchestration logic that coordinates
       between planner and doer agents.
    
    3. Error handling is implemented throughout the tool execution pipeline with
       comprehensive try-catch blocks and logging.
    
    4. The system implements a context store that tracks explored files and maintains
       state across multiple agent interactions.
    
    FINAL ANSWER: The agent system follows a hierarchical architecture with clear
    separation of concerns between planning and execution.
    """
    
    tool_outputs = [
        "Tool: list_directory | Args: {'path': 'server/agent'} | Result: Found organizer.py, doer.py, tool_executor.py, and other agent components",
        "Tool: read_file | Args: {'file_path': 'server/agent/organizer.py'} | Result: Contains main orchestration logic with planner-doer coordination",
        "Tool: read_file | Args: {'file_path': 'server/agent/doer.py'} | Result: Implements task execution logic with tool calling capabilities"
    ]
    
    # Analyze this interaction for learning opportunities
    insights = learning_identifier.analyze_interaction(
        task_description=task,
        doer_response=doer_response,
        tool_outputs=tool_outputs,
        context={"original_task": "Understand system architecture"}
    )
    
    print(f"🔍 Learning Analysis Results:")
    print(f"   Found {len(insights)} valuable insights")
    
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. \"{insight.insight_text[:80]}...\"")
        print(f"      Confidence: {insight.confidence:.2f}, Keywords: {insight.keywords[:3]}")
    
    # Check if learning should be triggered
    should_learn = learning_identifier.should_trigger_learning(
        insights=insights,
        task_success=True,  # Task completed successfully
        novel_information_discovered=True  # New files were explored
    )
    
    print(f"\n💡 Learning Decision: {'Store insights' if should_learn else 'Skip storage'}")
    
    if should_learn:
        print("   Storing top insights in knowledge base...")
        for insight in insights[:2]:  # Store top 2 insights
            knowledge_text = (
                f"Context: {task}\n"
                f"Insight: {insight.insight_text}\n"
                f"Keywords: {', '.join(insight.keywords)}\n"
                f"Confidence: {insight.confidence:.2f}"
            )
            
            rag_manager.add_knowledge(
                text=knowledge_text,
                source=f"architecture_analysis_{insight.source.split('_')[-1]}"
            )
        
        rag_manager.save()
        print(f"   💾 Knowledge base now contains {rag_manager.ntotal} entries")
    
    # Scenario 2: Agent encounters error and learns solution
    print("\n🔧 Scenario 2: Learning from error resolution")
    print("-" * 40)
    
    error_task = "Fix the file reading issue in the tool executor"
    error_response = """
    I identified the issue and successfully resolved it. The problem was caused by
    improper error handling when files don't exist. I discovered that adding proper
    exception handling with specific FileNotFoundError catches resolves the issue.
    
    The solution involved updating the tool_executor.py file to include:
    - Specific exception types for different error conditions
    - Graceful fallback behavior for missing files
    - Improved error messages for debugging
    
    FINAL ANSWER: Error handling improved with specific exception handling for file operations.
    """
    
    error_tool_outputs = [
        "Tool: read_file | Args: {'file_path': 'server/agent/tool_executor.py'} | Result: Found problematic error handling in execute_tool function",
        "Tool: write_file | Args: {'file_path': 'server/agent/tool_executor.py'} | Result: Updated with improved exception handling"
    ]
    
    error_insights = learning_identifier.analyze_interaction(
        task_description=error_task,
        doer_response=error_response,
        tool_outputs=error_tool_outputs,
        context={"original_task": "Fix file reading errors"}
    )
    
    print(f"🔍 Error Resolution Analysis:")
    for insight in error_insights:
        print(f"   - \"{insight.insight_text[:60]}...\" (confidence: {insight.confidence:.2f})")
    
    # Store error resolution insights
    if error_insights:
        for insight in error_insights[:1]:
            knowledge_text = (
                f"Error Resolution Context: {error_task}\n"
                f"Solution: {insight.insight_text}\n"
                f"Type: Error handling improvement"
            )
            rag_manager.add_knowledge(
                text=knowledge_text,
                source="error_resolution_learning"
            )
        rag_manager.save()
    
    # Scenario 3: Using learned knowledge for new tasks
    print("\n🚀 Scenario 3: Applying learned knowledge to new tasks")
    print("-" * 40)
    
    # Build a prompt for a new task that should benefit from learned knowledge
    context_builder = ContextBuilder()
    
    new_task = "Review the agent orchestration code for potential improvements"
    enhanced_prompt = await context_builder.build_system_prompt_planner(new_task)
    
    # Check if memories were injected
    if "agent orchestration" in enhanced_prompt.lower() or "planner-doer" in enhanced_prompt.lower():
        print("✅ Success! Relevant memories were injected into the prompt")
        
        # Show what memories were retrieved
        relevant_memories = rag_manager.retrieve_knowledge(new_task, top_k=3)
        print("📚 Retrieved memories:")
        for i, memory in enumerate(relevant_memories, 1):
            print(f"   {i}. Source: {memory['source']}")
            print(f"      Content: {memory['text'][:100]}...")
    else:
        print("⚠️  Memory injection may need tuning")
    
    # Scenario 4: Memory retrieval for different types of queries
    print("\n🔍 Scenario 4: Testing memory retrieval for various queries")
    print("-" * 40)
    
    test_queries = [
        "error handling in file operations",
        "agent system architecture patterns", 
        "tool executor implementation",
        "orchestration and coordination logic"
    ]
    
    for query in test_queries:
        results = rag_manager.retrieve_knowledge(query, top_k=2)
        print(f"\n📖 Query: \"{query}\"")
        if results:
            for j, result in enumerate(results, 1):
                score = result.get('relevance_score', 'N/A')
                print(f"   {j}. [{score:.2f}] {result['text'][:80]}...")
        else:
            print("   No relevant memories found")
    
    # Show final statistics
    print("\n📊 Final Learning Statistics")
    print("-" * 40)
    print(f"Total knowledge entries: {rag_manager.ntotal}")
    print(f"Learning system: {'Enhanced RAG' if hasattr(rag_manager._manager, 'keyword_index') else 'FAISS-based'}")
    
    print("\n🎉 Learning System Demonstration Complete!")
    print("The agent can now use this knowledge to make better decisions in future tasks.")


if __name__ == "__main__":
    asyncio.run(demonstrate_learning_workflow())