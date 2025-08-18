#!/usr/bin/env python3
"""
Comprehensive integration test for the tool learning system.
This demonstrates the complete flow including:
1. Tool failure recording
2. LLM-driven learning analysis  
3. Learning storage
4. Learning retrieval and application
5. Prevention of repeated failures
"""

import asyncio
import logging
import time
import uuid
import json
from typing import Dict, Any

from server.agent.tool_learning_store import tool_learning_store, ToolOutcome
from server.agent.tool_executor import process_tool_calls, _analyze_tool_result
from server.agent.tools.tool_manager import tool_manager
from server.agent.config import TOOL_CALL_REGEX

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockLLMAnalysis:
    """Mock LLM that provides realistic learning analysis"""
    
    def __init__(self):
        self.learning_patterns = {
            ("git_add", "git_commit"): {
                "is_learnable": True,
                "failure_pattern": "git_add fails when specific files don't exist, have wrong paths, or there are permission issues",
                "successful_alternative": "Use git_commit with add_all=true to stage and commit all changes atomically without specifying individual files",
                "confidence": 0.95,
                "context_pattern": "When committing changes and individual file paths might be problematic",
                "key_insights": "git_commit with add_all=true is more robust and handles edge cases automatically"
            },
            ("read_file", "search_codebase"): {
                "is_learnable": True,
                "failure_pattern": "read_file fails when file path is incorrect or file doesn't exist",
                "successful_alternative": "Use search_codebase to find files by content or name patterns before reading",
                "confidence": 0.85,
                "context_pattern": "When trying to access files but unsure of exact path",
                "key_insights": "Search-first approach is more robust than direct file access with uncertain paths"
            }
        }
    
    def analyze(self, prompt: str) -> str:
        """Analyze the prompt and return appropriate learning response"""
        # Extract tool names from the prompt to determine which pattern to use
        if "git_add" in prompt and "git_commit" in prompt:
            pattern = self.learning_patterns[("git_add", "git_commit")]
        elif "read_file" in prompt and "search_codebase" in prompt:
            pattern = self.learning_patterns[("read_file", "search_codebase")]
        else:
            # Default "not learnable" response
            pattern = {
                "is_learnable": False,
                "reason": "Pattern doesn't match known learnable scenarios"
            }
        
        return json.dumps(pattern, indent=2)

async def simulate_realistic_failure_scenarios():
    """Simulate realistic tool failure scenarios that could happen in practice"""
    
    print("🧪 COMPREHENSIVE TOOL LEARNING INTEGRATION TEST")
    print("=" * 60)
    
    mock_llm = MockLLMAnalysis()
    results = []
    
    # Scenario 1: Git workflow failure
    print("\n📚 SCENARIO 1: Git Workflow Learning")
    print("-" * 40)
    
    scenario1_result = await test_git_learning_scenario(mock_llm)
    results.append(("Git Workflow", scenario1_result))
    
    # Scenario 2: File access failure  
    print("\n📚 SCENARIO 2: File Access Learning")
    print("-" * 40)
    
    scenario2_result = await test_file_access_scenario(mock_llm)
    results.append(("File Access", scenario2_result))
    
    # Scenario 3: Learning prevention
    print("\n📚 SCENARIO 3: Learning Application & Prevention")
    print("-" * 40)
    
    prevention_result = await test_learning_prevention()
    results.append(("Prevention", prevention_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    for scenario_name, result in results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"{status} {scenario_name}: {result['message']}")
    
    overall_success = all(result["success"] for _, result in results)
    print(f"\n🎯 OVERALL RESULT: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return overall_success

async def test_git_learning_scenario(mock_llm: MockLLMAnalysis) -> Dict[str, Any]:
    """Test the git add -> git commit learning scenario"""
    
    try:
        # Start new conversation
        conversation_id = f"integration_git_{int(time.time())}"
        tool_learning_store.start_conversation(conversation_id)
        print(f"🔄 Started conversation: {conversation_id}")
        
        # Step 1: Record git_add failure
        print("1️⃣ Recording git_add failure...")
        failure_id = str(uuid.uuid4())
        
        tool_learning_store.record_tool_execution(
            tool_name="git_add",
            arguments={"files": ["src/nonexistent.py", "docs/missing.md"]},
            result="❌ Error executing command: git add src/nonexistent.py docs/missing.md\nReturn Code: 128\nStderr: fatal: pathspec 'src/nonexistent.py' did not match any files",
            outcome=ToolOutcome.FAILURE,
            execution_id=failure_id,
            error_details="Files not found in repository"
        )
        print(f"   ❌ Recorded failure: {failure_id}")
        
        # Step 2: Record git_commit success
        print("2️⃣ Recording git_commit success...")
        success_id = str(uuid.uuid4())
        
        tool_learning_store.record_tool_execution(
            tool_name="git_commit",
            arguments={"message": "Fix integration issues", "add_all": True},
            result="✅ Success:\n[main f7e8d9a] Fix integration issues\n 3 files changed, 45 insertions(+), 12 deletions(-)",
            outcome=ToolOutcome.SUCCESS,
            execution_id=success_id
        )
        print(f"   ✅ Recorded success: {success_id}")
        
        # Step 3: Trigger learning analysis
        print("3️⃣ Analyzing failure-success pattern...")
        learning = tool_learning_store.analyze_failure_success_pattern(
            failure_id,
            success_id,
            mock_llm.analyze
        )
        
        if not learning:
            return {"success": False, "message": "Learning analysis failed"}
        
        print(f"   🎓 Created learning: {learning.learning_id}")
        print(f"   📊 Confidence: {learning.confidence}")
        
        # Step 4: Verify learning storage
        retrieved_learnings = tool_learning_store.query_relevant_learnings(
            "git_add", 
            {"files": ["test.py"]},
            "committing changes"
        )
        
        if not retrieved_learnings or retrieved_learnings[0].learning_id != learning.learning_id:
            return {"success": False, "message": "Learning storage or retrieval failed"}
        
        print(f"   ✅ Learning stored and retrievable")
        
        return {
            "success": True, 
            "message": f"Git learning scenario completed successfully (confidence: {learning.confidence})",
            "learning_id": learning.learning_id
        }
        
    except Exception as e:
        logger.error(f"Git learning scenario failed: {e}", exc_info=True)
        return {"success": False, "message": f"Exception: {str(e)}"}

async def test_file_access_scenario(mock_llm: MockLLMAnalysis) -> Dict[str, Any]:
    """Test file access failure -> search success learning scenario"""
    
    try:
        # Start new conversation  
        conversation_id = f"integration_file_{int(time.time())}"
        tool_learning_store.start_conversation(conversation_id)
        
        # Record read_file failure
        failure_id = str(uuid.uuid4())
        tool_learning_store.record_tool_execution(
            tool_name="read_file",
            arguments={"file_path": "/wrong/path/config.py"},
            result="❌ Error: File not found: /wrong/path/config.py",
            outcome=ToolOutcome.FAILURE,
            execution_id=failure_id,
            error_details="Incorrect file path"
        )
        
        # Record search_codebase success
        success_id = str(uuid.uuid4())
        tool_learning_store.record_tool_execution(
            tool_name="search_codebase",
            arguments={"query": "config", "file_pattern": "*.py"},
            result="Found 3 files: server/config.py, client/config.py, tests/test_config.py",
            outcome=ToolOutcome.SUCCESS,
            execution_id=success_id
        )
        
        # Analyze pattern
        learning = tool_learning_store.analyze_failure_success_pattern(
            failure_id,
            success_id,
            mock_llm.analyze
        )
        
        if learning and learning.confidence > 0.8:
            return {
                "success": True,
                "message": f"File access learning created (confidence: {learning.confidence})"
            }
        else:
            return {"success": False, "message": "File access learning failed or low confidence"}
            
    except Exception as e:
        return {"success": False, "message": f"Exception: {str(e)}"}

async def test_learning_prevention() -> Dict[str, Any]:
    """Test that learnings can prevent future failures"""
    
    try:
        # Query for git learnings
        git_learnings = tool_learning_store.query_relevant_learnings(
            "git_add",
            {"files": ["app.py"]},
            "user wants to commit changes to repository"
        )
        
        if not git_learnings:
            return {"success": False, "message": "No git learnings found for prevention test"}
        
        high_confidence_learnings = [l for l in git_learnings if l.confidence > 0.9]
        
        if high_confidence_learnings:
            learning = high_confidence_learnings[0]
            print(f"🛡️ High-confidence learning found:")
            print(f"   Pattern: {learning.failure_pattern}")
            print(f"   Alternative: {learning.successful_alternative}")
            print(f"   Confidence: {learning.confidence}")
            print(f"   ✅ Would prevent future git_add failures!")
            
            return {
                "success": True,
                "message": f"Prevention system working - {len(high_confidence_learnings)} high-confidence learnings available"
            }
        else:
            return {"success": False, "message": "No high-confidence learnings available for prevention"}
            
    except Exception as e:
        return {"success": False, "message": f"Exception: {str(e)}"}

async def test_tool_executor_integration():
    """Test the enhanced tool executor with failure detection"""
    
    print("\n🔧 TESTING ENHANCED TOOL EXECUTOR")
    print("-" * 40)
    
    # Test the _analyze_tool_result function
    test_cases = [
        ("✅ Success: Operation completed", True, "Should detect success"),
        ("❌ Error: Command failed", False, "Should detect error"),
        ("fatal: not a git repository", False, "Should detect git error"),
        ("{'result': 'data', 'status': 'ok'}", True, "Should detect structured success"),
        ("Command completed successfully", True, "Should detect success keyword"),
    ]
    
    print("Testing result analysis...")
    all_passed = True
    
    for result, expected_success, description in test_cases:
        actual_success, error_details = _analyze_tool_result(result)
        passed = actual_success == expected_success
        all_passed &= passed
        
        status = "✅" if passed else "❌"
        print(f"   {status} {description}")
        if not passed:
            print(f"      Expected: {expected_success}, Got: {actual_success}")
    
    return {
        "success": all_passed,
        "message": f"Tool executor analysis: {'passed' if all_passed else 'failed'}"
    }

async def main():
    """Main integration test"""
    print("🚀 STARTING COMPREHENSIVE TOOL LEARNING INTEGRATION TEST")
    print("This test validates the complete tool learning pipeline...")
    
    # Test core scenarios
    success = await simulate_realistic_failure_scenarios()
    
    # Test tool executor integration
    executor_result = await test_tool_executor_integration()
    
    print(f"\n🔧 Tool Executor Test: {'✅ PASSED' if executor_result['success'] else '❌ FAILED'}")
    
    overall_success = success and executor_result["success"]
    
    print("\n" + "=" * 60)
    print("🎯 FINAL INTEGRATION TEST RESULT")
    print("=" * 60)
    
    if overall_success:
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("\n🎉 The tool learning system is working correctly:")
        print("   • Tool failures are immediately recorded with proper outcome detection")
        print("   • LLM analysis correctly identifies learnable patterns")  
        print("   • Learnings are stored and retrieved effectively")
        print("   • High-confidence learnings can prevent future failures")
        print("   • The system integrates properly with the tool executor")
    else:
        print("❌ SOME INTEGRATION TESTS FAILED!")
        print("   Please check the test output above for details.")
    
    return overall_success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
