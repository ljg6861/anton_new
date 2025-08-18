#!/usr/bin/env python3
"""
Test script to demonstrate the tool learning system with git scenario.
This script simulates the failure scenario where git_add fails and git_commit with add_all=True succeeds.
"""

import asyncio
import logging
import time
import uuid
from server.agent.tool_learning_store import tool_learning_store, ToolOutcome
from server.agent.tools.tool_manager import tool_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def simulate_git_failure_scenario():
    """
    Simulate the git scenario:
    1. git_add fails with non-existent file
    2. git_commit with add_all=True succeeds 
    3. System learns the pattern
    """
    print("🧪 Starting Git Tool Learning Test")
    print("=" * 50)
    
    # Initialize conversation
    conversation_id = f"test_git_{int(time.time())}"
    tool_learning_store.start_conversation(conversation_id)
    print(f"📝 Started conversation: {conversation_id}")
    
    # Step 1: Simulate git_add failure
    print("\n🔴 Step 1: Simulating git_add failure...")
    
    failure_execution_id = str(uuid.uuid4())
    failure_args = {"files": ["nonexistent_file.txt"]}
    failure_result = "❌ Error: pathspec 'nonexistent_file.txt' did not match any files"
    
    tool_learning_store.record_tool_execution(
        tool_name="git_add",
        arguments=failure_args,
        result=failure_result,
        outcome=ToolOutcome.FAILURE,
        execution_id=failure_execution_id,
        error_details="File not found in repository"
    )
    
    print(f"   ❌ git_add failed: {failure_result}")
    print(f"   📋 Recorded failure with ID: {failure_execution_id}")
    
    # Step 2: Simulate git_commit with add_all=True success
    print("\n🟢 Step 2: Simulating git_commit with add_all=True success...")
    
    success_execution_id = str(uuid.uuid4())
    success_args = {"message": "Auto commit", "add_all": True}
    success_result = "✅ Success: [main abc1234] Auto commit\\n 1 file changed, 5 insertions(+)"
    
    tool_learning_store.record_tool_execution(
        tool_name="git_commit",
        arguments=success_args,
        result=success_result,
        outcome=ToolOutcome.SUCCESS,
        execution_id=success_execution_id
    )
    
    print(f"   ✅ git_commit succeeded: {success_result}")
    print(f"   📋 Recorded success with ID: {success_execution_id}")
    
    # Step 3: Simulate LLM learning analysis
    print("\n🧠 Step 3: Performing learning analysis...")
    
    def mock_llm_analysis(prompt):
        """Mock LLM analysis that recognizes the git pattern"""
        print(f"   🤖 LLM Analysis Prompt Length: {len(prompt)} characters")
        return '''
        {
            "is_learnable": true,
            "failure_pattern": "git_add fails when specific files don't exist or have wrong paths",
            "successful_alternative": "Use git_commit with add_all=true to stage and commit all changes in one step",
            "confidence": 0.9,
            "context_pattern": "When trying to stage files for commit and individual file paths might be incorrect",
            "key_insights": "git_commit with add_all=true is more robust than separate git_add commands as it handles all modified files automatically without requiring exact file paths"
        }
        '''
    
    learning = tool_learning_store.analyze_failure_success_pattern(
        failure_execution_id,
        success_execution_id,
        mock_llm_analysis
    )
    
    if learning:
        print(f"   🎓 Learning created: {learning.learning_id}")
        print(f"   📖 Pattern: {learning.failure_pattern}")
        print(f"   💡 Alternative: {learning.successful_alternative}")
        print(f"   🎯 Confidence: {learning.confidence}")
    else:
        print("   ❌ No learning was created")
        return
    
    # Step 4: Test learning retrieval
    print("\n🔍 Step 4: Testing learning retrieval...")
    
    # Query for learnings when about to use git_add
    relevant_learnings = tool_learning_store.query_relevant_learnings(
        tool_name="git_add",
        arguments={"files": ["some_file.py"]},
        context="User wants to commit changes"
    )
    
    print(f"   📚 Found {len(relevant_learnings)} relevant learnings for git_add")
    
    for i, learning in enumerate(relevant_learnings):
        print(f"   {i+1}. Pattern: {learning.failure_pattern}")
        print(f"      Alternative: {learning.successful_alternative}")
        print(f"      Confidence: {learning.confidence}")
        print(f"      Tools involved: {learning.tool_names_involved}")
    
    # Step 5: Simulate prevention scenario
    print("\n🛡️ Step 5: Simulating prevention scenario...")
    
    if relevant_learnings and relevant_learnings[0].confidence > 0.8:
        print("   🚨 High-confidence learning found!")
        print("   💭 LLM would be advised to use git_commit with add_all=True instead")
        print("   ✅ Failure prevented through learning!")
    else:
        print("   ⚠️ No high-confidence learnings found")
    
    print("\n" + "=" * 50)
    print("🎉 Git Tool Learning Test Complete!")
    print("\n📊 Summary:")
    print(f"   • Recorded 1 failure (git_add)")
    print(f"   • Recorded 1 success (git_commit)")
    print(f"   • Created 1 learning pattern")
    print(f"   • Learning can prevent future failures")
    
    return learning

async def test_actual_git_tools():
    """Test with actual git tools if available"""
    print("\n🧪 Testing with Actual Git Tools")
    print("=" * 50)
    
    try:
        # Test git_status (should work)
        print("Testing git_status...")
        status_result = tool_manager.run_tool("git_status", {})
        print(f"git_status result: {status_result[:100]}...")
        
        # Test git_add with non-existent file (should fail)
        print("\nTesting git_add with non-existent file...")
        add_result = tool_manager.run_tool("git_add", {"files": ["nonexistent_test_file.txt"]})
        print(f"git_add result: {add_result[:100]}...")
        
        # Test git_commit with add_all (might succeed if there are changes)
        print("\nTesting git_commit with add_all...")
        commit_result = tool_manager.run_tool("git_commit", {"message": "Test commit", "add_all": True})
        print(f"git_commit result: {commit_result[:100]}...")
        
    except Exception as e:
        print(f"Error testing actual git tools: {e}")
        logger.error("Git tool test failed", exc_info=True)

async def main():
    """Main test function"""
    print("🚀 Tool Learning System Test Suite")
    print("=" * 50)
    
    # Run the simulated scenario
    learning = await simulate_git_failure_scenario()
    
    # Test with actual tools if available
    await test_actual_git_tools()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
