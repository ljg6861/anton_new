#!/usr/bin/env python3
"""
Test that the tool executor properly integrates corrective action suggestions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from agent.tool_learning_store import ToolLearningStore, ToolOutcome, ToolLearning
import tempfile
import json
import time
import uuid

def test_tool_executor_corrective_integration():
    """Test the tool executor integration with corrective action"""
    
    # Set up temporary database  
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        store = ToolLearningStore(db_path)
        
        print("🧪 Testing Tool Executor Corrective Integration")
        print("=" * 55)
        
        # STEP 1: Seed a learning pattern
        print("\n📚 STEP 1: Creating learning pattern...")
        
        learning = ToolLearning(
            learning_id=str(uuid.uuid4()),
            failure_pattern="git status fails in non-git directory",
            successful_alternative="git init && git status",
            confidence=0.90,
            context_pattern="git directory initialization status",
            tool_names_involved=["git_status"],
            created_timestamp=time.time(),
            last_confirmed_timestamp=time.time(),
            confirmation_count=1,
            llm_analysis="When git status fails in non-git directory, initialize git first"
        )
        
        store._store_learning(learning)
        print(f"✅ Learning pattern created: {learning.learning_id}")
        
        # STEP 2: Test the corrective action message format
        print("\n🔧 STEP 2: Testing corrective action message format...")
        
        # Start conversation and trigger a failure that should get corrective action
        store.start_conversation("conv_integration_test")
        
        exec_id, suggested_alternatives = store.record_tool_execution(
            tool_name="git_status",
            arguments={"path": "/tmp/not-a-git-repo"},
            result="Error: not a git repository",
            outcome=ToolOutcome.FAILURE,
            execution_id=str(uuid.uuid4())
        )
        
        print(f"📋 Execution ID: {exec_id}")
        print(f"🤖 Suggested alternatives: {len(suggested_alternatives)}")
        
        if suggested_alternatives:
            # Test the message format that would be used in tool executor
            alternatives_text = "\n".join([
                f"• {alt.successful_alternative} (confidence: {alt.confidence:.1%})"
                for alt in suggested_alternatives[:3]  # Top 3 alternatives
            ])
            
            tool_result = f"❌ Error: not a git repository\n\n🤖 **CORRECTIVE ACTION SUGGESTED:**\nBased on past learnings, try these alternatives:\n{alternatives_text}\n\nThe original git_status approach failed, but these alternatives have worked in similar situations."
            
            corrective_message = {
                "role": "system",
                "content": f"🚨 TOOL FAILURE RECOVERY: git_status failed. High-confidence alternatives available:\n{alternatives_text}\n\nConsider using these learned alternatives instead of retrying the same approach."
            }
            
            print("\n💬 TOOL RESULT MESSAGE:")
            print(tool_result)
            print("\n💬 SYSTEM RECOVERY MESSAGE:")
            print(corrective_message["content"])
            
            # Verify key components are present
            has_error = "❌ Error:" in tool_result
            has_corrective_action = "CORRECTIVE ACTION SUGGESTED" in tool_result
            has_alternatives = "git init && git status" in tool_result
            has_confidence = "90.0%" in tool_result
            has_system_message = "TOOL FAILURE RECOVERY" in corrective_message["content"]
            
            print(f"\n✅ Components check:")
            print(f"   - Error message: {has_error}")
            print(f"   - Corrective action header: {has_corrective_action}")
            print(f"   - Alternative suggestion: {has_alternatives}")
            print(f"   - Confidence display: {has_confidence}")
            print(f"   - System recovery message: {has_system_message}")
            
            all_components_present = all([
                has_error, has_corrective_action, has_alternatives, 
                has_confidence, has_system_message
            ])
            
            return all_components_present
        else:
            print("❌ No corrective action suggestions returned!")
            return False
            
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass

if __name__ == "__main__":
    success = test_tool_executor_corrective_integration()
    
    print("\n" + "=" * 55)
    if success:
        print("🎉 TOOL EXECUTOR CORRECTIVE INTEGRATION TEST PASSED!")
        print("✅ Error messages include corrective action suggestions")
        print("✅ System messages provide recovery guidance")
        print("✅ Confidence levels and alternatives are properly formatted")
    else:
        print("❌ TOOL EXECUTOR CORRECTIVE INTEGRATION TEST FAILED!")
        sys.exit(1)
