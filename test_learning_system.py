#!/usr/bin/env python3
"""
Test script for the learning identification and memory integration system.

This script tests the complete learning workflow:
1. Learning identification from agent interactions
2. Memory storage via RAG
3. Memory retrieval and injection into prompts
"""

import asyncio
import logging
from server.agent.learning_identifier import learning_identifier
from server.agent.rag_manager import rag_manager
from client.context_builder import ContextBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_learning_workflow():
    """Test the complete learning workflow."""
    print("=" * 60)
    print("Testing Learning Identification and Memory Integration")
    print("=" * 60)
    
    # Initial state
    print(f"Initial RAG entries: {rag_manager.ntotal}")
    
    # Simulate an agent interaction
    task_description = "Analyze the repository structure to understand the codebase"
    doer_response = """I successfully read several files and discovered that the system uses Flask for the web interface. 
    The main.py file contains the primary application setup with route handlers. 
    The server/agent directory implements the agent orchestration system with separate doer and planner components.
    Error handling is implemented throughout the tool execution pipeline."""
    
    tool_outputs = [
        "Tool: list_directory | Args: {'path': '.'} | Result: Found main.py, server/, client/, utils/ directories",
        "Tool: read_file | Args: {'file_path': 'main.py'} | Result: Flask application with route definitions and error handling",
        "Tool: read_file | Args: {'file_path': 'server/agent/organizer.py'} | Result: Agent orchestration logic with planner-doer pattern"
    ]
    
    context = {"original_task": "Understand the system architecture"}
    
    # Test learning identification
    print("\n1. Testing Learning Identification:")
    insights = learning_identifier.analyze_interaction(
        task_description=task_description,
        doer_response=doer_response,
        tool_outputs=tool_outputs,
        context=context
    )
    
    print(f"Identified {len(insights)} learning insights:")
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. \"{insight.insight_text[:80]}...\" (confidence: {insight.confidence:.2f})")
    
    # Test learning trigger
    should_learn = learning_identifier.should_trigger_learning(
        insights=insights,
        task_success=True,
        novel_information_discovered=True
    )
    print(f"\nShould trigger learning: {should_learn}")
    
    # Test memory storage
    if should_learn and insights:
        print("\n2. Testing Memory Storage:")
        for insight in insights[:2]:  # Store top 2 insights
            knowledge_text = (
                f"Context: {task_description}\n"
                f"Insight: {insight.insight_text}\n"
                f"Keywords: {', '.join(insight.keywords)}\n"
                f"Related to: {context['original_task']}"
            )
            
            rag_manager.add_knowledge(
                text=knowledge_text,
                source=f"learning_test_{insight.source}"
            )
            print(f"  Stored: {insight.insight_text[:60]}...")
        
        rag_manager.save()
        print(f"  Total entries after learning: {rag_manager.ntotal}")
    
    # Test memory retrieval
    print("\n3. Testing Memory Retrieval:")
    test_queries = [
        "Flask web application",
        "agent system architecture", 
        "error handling implementation"
    ]
    
    for query in test_queries:
        results = rag_manager.retrieve_knowledge(query, top_k=2)
        print(f"  Query: \"{query}\" -> Found {len(results)} relevant memories")
        for j, result in enumerate(results):
            print(f"    {j+1}. {result['text'][:100]}... (source: {result['source']})")
    
    # Test context building with memory injection
    print("\n4. Testing Memory Injection in Prompts:")
    context_builder = ContextBuilder()
    
    # Test with a task that should retrieve relevant memories
    test_task = "Review the Flask application code for potential improvements"
    prompt = await context_builder.build_system_prompt_planner(test_task)
    
    # Check if memories were injected
    if "Flask" in prompt and "From learning_test" in prompt:
        print("  ✅ Memory successfully injected into planner prompt")
        print(f"  Sample memory context: {prompt[prompt.find('From learning_test'):prompt.find('From learning_test')+100]}...")
    else:
        print("  ❌ Memory injection may not be working correctly")
    
    print("\n" + "=" * 60)
    print("Learning System Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_learning_workflow())