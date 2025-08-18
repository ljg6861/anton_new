#!/usr/bin/env python3
"""
Test the corrective action system - when tools fail, should immediately suggest alternatives
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from agent.tool_learning_store import ToolLearningStore, ToolOutcome
from agent.tool_executor import ToolExecutor
import tempfile
import json

def test_corrective_action_system():
    """Test that tool failures trigger immediate corrective action"""
    
    # Set up temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        store = ToolLearningStore(db_path)
        executor = ToolExecutor()
        
        print("🧪 Testing Corrective Action System")
        print("=" * 50)
        
        # STEP 1: Seed some learning data (failures followed by successes)
        print("\n📚 STEP 1: Seeding learning data...")
        
        # Simulate a pattern: git clone fails, git pull works
        conversation_id = "conv_corrective_test"
        
        # First failure
        exec_id_1, _ = store.record_tool_execution(
            tool_name="git",
            arguments={"command": "clone", "url": "https://github.com/user/repo.git"},
            result="Error: Repository already exists",
            outcome=ToolOutcome.FAILURE,
            conversation_id=conversation_id
        )
        
        # Then success with different approach
        exec_id_2, _ = store.record_tool_execution(
            tool_name="git", 
            arguments={"command": "pull", "branch": "main"},
            result="Successfully updated repository",
            outcome=ToolOutcome.SUCCESS,
            conversation_id=conversation_id
        )
        
        # Simulate LLM analysis linking them
        def mock_llm_analysis(failure_exec_id, success_exec_id):
            return store.add_learned_pattern(
                failure_execution_id=failure_exec_id,
                success_execution_id=success_exec_id,
                pattern_description="When git clone fails due to existing repository, use git pull instead",
                successful_alternative="git pull",
                confidence=0.95,
                context_similarity=["git", "repository", "update", "existing"]
            )
        
        # Add the learning
        learning_id = mock_llm_analysis(exec_id_1, exec_id_2)
        print(f"✅ Added learning pattern: {learning_id}")
        
        # STEP 2: Test corrective action triggering
        print("\n🔧 STEP 2: Testing corrective action on new failure...")
        
        # Simulate the same type of failure
        failed_args = {"command": "clone", "url": "https://github.com/user/another-repo.git"}
        
        # This should trigger corrective action
        exec_id_3, suggested_alternatives = store.record_tool_execution(
            tool_name="git",
            arguments=failed_args,
            result="Error: Repository already exists",
            outcome=ToolOutcome.FAILURE,
            conversation_id="conv_new_failure"
        )
        
        print(f"📋 Execution ID: {exec_id_3}")
        print(f"🤖 Suggested alternatives: {len(suggested_alternatives)}")
        
        if suggested_alternatives:
            print("\n💡 CORRECTIVE ACTION TRIGGERED:")
            for i, alt in enumerate(suggested_alternatives, 1):
                print(f"   {i}. {alt.successful_alternative} (confidence: {alt.confidence:.1%})")
                print(f"      Pattern: {alt.pattern_description}")
                print()
        else:
            print("❌ No corrective action triggered!")
            return False
            
        # STEP 3: Test tool executor integration
        print("🛠️  STEP 3: Testing tool executor integration...")
        
        # Mock tool call that will fail
        tool_calls = [{
            "type": "function",
            "function": {
                "name": "git",
                "arguments": json.dumps({"command": "clone", "url": "https://github.com/user/test-repo.git"})
            }
        }]
        
        # Mock git tool that fails
        def mock_git_tool(**kwargs):
            raise Exception("Repository already exists")
        
        # Temporarily add mock tool
        original_tools = executor.tools.copy()
        executor.tools["git"] = mock_git_tool
        
        # Mock LLM callback
        def mock_llm_callback(failure_id, success_id):
            pass
        
        messages = []
        
        try:
            # Process the tool call - should trigger corrective action
            result = executor.process_tool_calls(
                tool_calls=tool_calls,
                tool_learning_store=store,
                llm_analysis_callback=mock_llm_callback,
                messages=messages
            )
            
            print(f"📄 Tool result contains corrective action: {'CORRECTIVE ACTION SUGGESTED' in str(result)}")
            
            # Check if corrective message was added
            corrective_message_added = any(
                msg.get("role") == "system" and "TOOL FAILURE RECOVERY" in msg.get("content", "")
                for msg in messages
            )
            
            print(f"💬 System message with alternatives added: {corrective_message_added}")
            
            if "CORRECTIVE ACTION SUGGESTED" in str(result) and corrective_message_added:
                print("✅ CORRECTIVE ACTION SYSTEM WORKING!")
                return True
            else:
                print("❌ Corrective action not properly integrated")
                return False
                
        finally:
            # Restore original tools
            executor.tools = original_tools
            
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass

if __name__ == "__main__":
    success = test_corrective_action_system()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 CORRECTIVE ACTION SYSTEM TEST PASSED!")
        print("✅ Tool failures now trigger immediate alternative suggestions")
    else:
        print("❌ CORRECTIVE ACTION SYSTEM TEST FAILED!")
        sys.exit(1)
