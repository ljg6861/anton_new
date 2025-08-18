#!/usr/bin/env python3
"""
Test script to verify the deduplication fix works correctly.
"""

import asyncio
import logging
import time
import uuid
from server.agent.tool_learning_store import tool_learning_store, ToolOutcome

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_deduplication():
    """Test that duplicate learnings are properly handled"""
    
    print("🧪 Testing Tool Learning Deduplication")
    print("=" * 50)
    
    # Initialize conversation
    conversation_id = f"test_dedup_{int(time.time())}"
    tool_learning_store.start_conversation(conversation_id)
    print(f"📝 Started conversation: {conversation_id}")
    
    # Mock LLM analysis function that always returns the same pattern
    def mock_llm_analysis(prompt: str) -> str:
        return '''
        {
            "is_learnable": true,
            "failure_pattern": "read_file fails when file path is incorrect or file doesn't exist",
            "successful_alternative": "Use search_codebase to find files by content or name patterns before reading",
            "confidence": 0.85,
            "context_pattern": "When trying to access files but unsure of exact path",
            "key_insights": "Search-first approach is more robust than direct file access"
        }
        '''
    
    # Create multiple identical learning scenarios
    learnings_created = []
    
    for i in range(3):
        print(f"\n🔄 Creating learning scenario {i+1}...")
        
        # Record failure
        failure_id = str(uuid.uuid4())
        tool_learning_store.record_tool_execution(
            tool_name="read_file",
            arguments={"path": f"/wrong/path/file{i}.py"},
            result="❌ Error: File not found",
            outcome=ToolOutcome.FAILURE,
            execution_id=failure_id,
            error_details="File not found"
        )
        
        # Record success
        success_id = str(uuid.uuid4())
        tool_learning_store.record_tool_execution(
            tool_name="search_codebase",
            arguments={"query": f"file{i}", "file_pattern": "*.py"},
            result="Found files matching pattern",
            outcome=ToolOutcome.SUCCESS,
            execution_id=success_id
        )
        
        # Analyze pattern
        learning = tool_learning_store.analyze_failure_success_pattern(
            failure_id,
            success_id,
            mock_llm_analysis
        )
        
        if learning:
            learnings_created.append(learning)
            print(f"   ✅ Learning created: {learning.learning_id}")
        else:
            print(f"   ➡️ No new learning (duplicate detected)")
    
    print(f"\n📊 Total unique learnings created: {len(learnings_created)}")
    
    # Test querying - should return only unique learnings
    print("\n🔍 Testing learning retrieval...")
    retrieved_learnings = tool_learning_store.query_relevant_learnings(
        "read_file", 
        {"path": "test.py"},
        "user wants to read a file"
    )
    
    print(f"📚 Retrieved {len(retrieved_learnings)} unique learnings")
    
    # Print the retrieved learnings to verify no duplicates
    for i, learning in enumerate(retrieved_learnings):
        print(f"   {i+1}. Pattern: {learning.failure_pattern}")
        print(f"      Alternative: {learning.successful_alternative}")
        print(f"      Confidence: {learning.confidence}")
    
    # Verify deduplication worked
    expected_unique_count = 1  # Should only have 1 unique learning despite 3 attempts
    success = len(learnings_created) <= expected_unique_count and len(retrieved_learnings) == expected_unique_count
    
    print(f"\n🎯 Deduplication Test: {'✅ PASSED' if success else '❌ FAILED'}")
    if success:
        print("   • Duplicate learnings were properly detected and prevented")
        print("   • Query returns only unique learnings without duplicates")
    else:
        print(f"   • Expected 1 unique learning, got {len(learnings_created)} created and {len(retrieved_learnings)} retrieved")
    
    return success

async def main():
    """Main test function"""
    print("🚀 Tool Learning Deduplication Test")
    print("Testing the fix for duplicate learning patterns...")
    
    success = await test_deduplication()
    
    print(f"\n🎉 Test completed: {'✅ SUCCESS' if success else '❌ FAILURE'}")
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
