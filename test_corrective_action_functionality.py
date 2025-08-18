#!/usr/bin/env python3
"""
Test the corrective action system - direct testing of the store functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from agent.tool_learning_store import ToolLearningStore, ToolOutcome, ToolLearning
import tempfile
import json

def test_corrective_action_functionality():
    """Test that the store returns corrective action suggestions correctly"""
    
    # Set up temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        store = ToolLearningStore(db_path)
        
        print("🧪 Testing Corrective Action Functionality")
        print("=" * 50)
        
        # STEP 1: Seed learning data
        print("\n📚 STEP 1: Creating learning pattern...")
        
        # Set conversation context
        store.start_conversation("conv_test")
        
        # Record a failure
        exec_id_1, _ = store.record_tool_execution(
            tool_name="git",
            arguments={"command": "clone", "url": "https://github.com/user/repo.git"},
            result="Error: Repository already exists",
            outcome=ToolOutcome.FAILURE,
            execution_id="exec_1"
        )
        
        # Record a success
        exec_id_2, _ = store.record_tool_execution(
            tool_name="git", 
            arguments={"command": "pull", "branch": "main"},
            result="Successfully updated repository",
            outcome=ToolOutcome.SUCCESS,
            execution_id="exec_2"
        )
        
        # Create learning pattern manually
        import time
        import uuid
        
        learning = ToolLearning(
            learning_id=str(uuid.uuid4()),
            failure_pattern="git clone fails with 'Repository already exists'",
            successful_alternative="git pull",
            confidence=0.95,
            context_pattern="git repository update existing",
            tool_names_involved=["git"],
            created_timestamp=time.time(),
            last_confirmed_timestamp=time.time(),
            confirmation_count=1,
            llm_analysis="When git clone fails due to existing repository, use git pull instead"
        )
        
        store._store_learning(learning)
        learning_id = learning.learning_id
        print(f"✅ Learning pattern created: {learning_id}")
        
        # STEP 2: Test immediate corrective action
        print("\n🔧 STEP 2: Testing corrective action on similar failure...")
        
        # Start new conversation for the test
        store.start_conversation("conv_new")
        
        # Record a similar failure - should trigger corrective action
        exec_id_3, suggested_alternatives = store.record_tool_execution(
            tool_name="git",
            arguments={"command": "clone", "url": "https://github.com/user/another-repo.git"},
            result="Error: Repository already exists",
            outcome=ToolOutcome.FAILURE,
            execution_id="exec_3"
        )
        
        print(f"📋 New failure execution ID: {exec_id_3}")
        print(f"🤖 Number of suggested alternatives: {len(suggested_alternatives)}")
        
        if suggested_alternatives:
            print("\n💡 CORRECTIVE ACTION SUGGESTIONS:")
            for i, alt in enumerate(suggested_alternatives, 1):
                print(f"   {i}. Alternative: {alt.successful_alternative}")
                print(f"      Confidence: {alt.confidence:.1%}")
                print(f"      Pattern: {alt.failure_pattern}")
                print()
        else:
            print("❌ No corrective action suggestions returned!")
            return False
            
        # STEP 3: Verify the suggestions are high-confidence
        high_confidence_count = sum(1 for alt in suggested_alternatives if alt.confidence >= 0.8)
        print(f"🎯 High-confidence suggestions (≥80%): {high_confidence_count}")
        
        # STEP 4: Test that success doesn't trigger corrective action
        print("\n✅ STEP 4: Testing that success doesn't trigger corrective action...")
        
        store.start_conversation("conv_success")
        
        exec_id_4, success_alternatives = store.record_tool_execution(
            tool_name="git",
            arguments={"command": "status"},
            result="On branch main, nothing to commit",
            outcome=ToolOutcome.SUCCESS,
            execution_id="exec_4"
        )
        
        print(f"📋 Success execution ID: {exec_id_4}")
        print(f"🤖 Alternatives for success: {len(success_alternatives)}")
        
        # STEP 5: Test pattern matching works correctly
        print("\n🔍 STEP 5: Testing pattern matching specificity...")
        
        store.start_conversation("conv_unrelated")
        
        # Different tool should not trigger git alternatives
        exec_id_5, unrelated_alternatives = store.record_tool_execution(
            tool_name="file_manager",
            arguments={"action": "delete", "path": "/tmp/test.txt"},
            result="Error: File not found",
            outcome=ToolOutcome.FAILURE,
            execution_id="exec_5"
        )
        
        print(f"📋 Unrelated failure execution ID: {exec_id_5}")
        print(f"🤖 Alternatives for unrelated failure: {len(unrelated_alternatives)}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY:")
        print(f"   ✅ Learning pattern created: {learning_id is not None}")
        print(f"   ✅ Similar failure triggered suggestions: {len(suggested_alternatives) > 0}")
        print(f"   ✅ High-confidence suggestions available: {high_confidence_count > 0}")
        print(f"   ✅ Success didn't trigger suggestions: {len(success_alternatives) == 0}")
        print(f"   ✅ Unrelated failures filtered: {len(unrelated_alternatives) == 0}")
        
        # Overall success
        all_passed = (
            learning_id is not None and
            len(suggested_alternatives) > 0 and
            high_confidence_count > 0 and
            len(success_alternatives) == 0 and
            len(unrelated_alternatives) == 0
        )
        
        return all_passed
        
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass

if __name__ == "__main__":
    success = test_corrective_action_functionality()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 CORRECTIVE ACTION FUNCTIONALITY TEST PASSED!")
        print("✅ Tool failures trigger immediate alternative suggestions")
        print("✅ High-confidence patterns are identified correctly")
        print("✅ System filters suggestions appropriately")
    else:
        print("❌ CORRECTIVE ACTION FUNCTIONALITY TEST FAILED!")
        sys.exit(1)
