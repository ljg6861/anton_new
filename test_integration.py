"""
Integration test to demonstrate knowledge flow between planner, doer, and evaluator components.
This shows that the key issues from the problem statement have been resolved.
"""
import sys
from unittest.mock import MagicMock

# Mock dependencies to avoid import issues
sys.modules['server.agent.rag_manager'] = MagicMock()
sys.modules['ollama'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['vllm.third_party.pynvml'] = MagicMock()
sys.modules['client.context_builder'] = MagicMock()
sys.modules['metrics'] = MagicMock()
sys.modules['server.agent.config'] = MagicMock()
sys.modules['server.agent.knowledge_handler'] = MagicMock()
sys.modules['server.agent.message_handler'] = MagicMock()
sys.modules['server.agent.prompts'] = MagicMock()
sys.modules['server.model_server'] = MagicMock()

from server.agent.knowledge_store import KnowledgeStore, ContextType, ImportanceLevel


def test_knowledge_flow_integration():
    """
    Test that demonstrates all problem statement issues are resolved:
    
    1. Context store tracks explored files and code content consistently across tool interactions
    2. Structured knowledge transfer between planner and doer components  
    3. Doer discoveries are automatically available to Planner for future planning
    4. Evaluator contributes to persistent knowledge state
    5. Context elements are prioritized and weighted by importance
    """
    print("🧪 Testing Knowledge Flow Integration...")
    
    # Initialize knowledge store
    knowledge_store = KnowledgeStore()
    
    # ISSUE 1: Context store tracks explored files consistently across tool interactions
    print("\n1️⃣  Testing consistent context tracking across tool interactions...")
    
    # Simulate doer tool execution (file reading)
    knowledge_store.update_from_tool_execution(
        "read_file", 
        {"file_path": "/src/main.py"}, 
        "def main():\n    print('Hello world')"
    )
    
    # Simulate doer tool execution (directory listing)
    knowledge_store.update_from_tool_execution(
        "list_directory",
        {"path": "/src"},
        "main.py\nconfig.py\nutils.py"
    )
    
    # Verify context is tracked
    assert "/src/main.py" in knowledge_store.explored_files
    assert "/src" in knowledge_store.explored_files
    assert knowledge_store.code_content["/src/main.py"] == "def main():\n    print('Hello world')"
    print("   ✅ Context consistently tracked across tool interactions")
    
    
    # ISSUE 2: Structured knowledge transfer between planner and doer
    print("\n2️⃣  Testing structured knowledge transfer between planner and doer...")
    
    # Simulate planner adding insights
    knowledge_store.add_planner_insight(
        "Need to examine configuration files to understand the system architecture",
        ImportanceLevel.HIGH
    )
    
    # Simulate doer making a discovery
    knowledge_store.update_from_tool_execution(
        "read_file",
        {"file_path": "/config/database.yaml"},
        "database:\n  host: localhost\n  port: 5432"
    )
    
    # Generate context summary for planner (demonstrates knowledge transfer)
    context_summary = knowledge_store.build_context_summary()
    
    # Verify planner insights and doer discoveries are both included
    assert "configuration" in context_summary.lower() or "config" in context_summary.lower()
    assert "database.yaml" in context_summary or "/config/database.yaml" in context_summary
    print("   ✅ Structured knowledge transfer between planner and doer working")
    
    
    # ISSUE 3: Doer discoveries automatically available to Planner for future planning
    print("\n3️⃣  Testing automatic availability of doer discoveries to planner...")
    
    # Get prioritized context (what planner would receive)
    prioritized_context = knowledge_store.get_prioritized_context(max_items=10)
    
    # Check that doer's file discoveries are available
    file_discoveries = [item for item in prioritized_context if item.context_type == ContextType.FILE_CONTENT]
    assert len(file_discoveries) >= 2  # main.py and database.yaml
    
    # Check that planner insights are also preserved
    planner_insights = [item for item in prioritized_context if item.context_type == ContextType.PLANNER_INSIGHT]
    assert len(planner_insights) >= 1
    print("   ✅ Doer discoveries automatically available to planner for future planning")
    
    
    # ISSUE 4: Evaluator contributes to persistent knowledge state
    print("\n4️⃣  Testing evaluator contribution to persistent knowledge...")
    
    # Simulate evaluator feedback
    knowledge_store.add_evaluator_feedback(
        "SUCCESS: Database configuration found and validated",
        ImportanceLevel.HIGH
    )
    
    knowledge_store.add_evaluator_feedback(
        "FAILURE: Missing environment variables for production setup",
        ImportanceLevel.CRITICAL
    )
    
    # Verify evaluator feedback is captured
    evaluator_feedback = [item for item in knowledge_store.context_items 
                         if item.context_type == ContextType.EVALUATOR_FEEDBACK]
    assert len(evaluator_feedback) == 2
    
    # Verify feedback is included in prioritized context
    prioritized = knowledge_store.get_prioritized_context(max_items=15)
    feedback_in_prioritized = [item for item in prioritized 
                              if item.context_type == ContextType.EVALUATOR_FEEDBACK]
    assert len(feedback_in_prioritized) >= 1
    print("   ✅ Evaluator contributes to persistent knowledge state")
    
    
    # ISSUE 5: Context elements are prioritized and weighted by importance
    print("\n5️⃣  Testing context prioritization and importance weighting...")
    
    # Add items with different importance levels
    knowledge_store.add_context("Low priority info", ContextType.TASK_PROGRESS, ImportanceLevel.LOW, "test")
    knowledge_store.add_context("Critical system error", ContextType.EVALUATOR_FEEDBACK, ImportanceLevel.CRITICAL, "test")
    knowledge_store.add_context("Medium importance note", ContextType.PLANNER_INSIGHT, ImportanceLevel.MEDIUM, "test")
    
    # Get prioritized context
    prioritized = knowledge_store.get_prioritized_context(max_items=20)
    
    # Verify that higher importance items come first
    importance_scores = []
    for item in prioritized[:5]:  # Check first 5 items
        importance_scores.append(item.importance.value)
    
    # Should be generally descending (allowing for recency effects)
    critical_items = [score for score in importance_scores if score == ImportanceLevel.CRITICAL.value]
    low_items = [score for score in importance_scores if score == ImportanceLevel.LOW.value]
    
    # Critical items should appear before low items in prioritized list
    if critical_items and low_items:
        critical_pos = importance_scores.index(ImportanceLevel.CRITICAL.value)
        low_pos = next((i for i, score in enumerate(importance_scores) if score == ImportanceLevel.LOW.value), len(importance_scores))
        assert critical_pos < low_pos, "Critical items should be prioritized over low importance items"
    
    print("   ✅ Context elements properly prioritized and weighted by importance")
    
    
    # ADDITIONAL: Test backward compatibility
    print("\n🔄 Testing backward compatibility...")
    
    legacy_format = knowledge_store.get_legacy_context_store()
    assert "explored_files" in legacy_format
    assert "code_content" in legacy_format  
    assert "task_progress" in legacy_format
    print("   ✅ Backward compatibility maintained")
    
    
    print(f"\n🎉 All tests passed! Knowledge management system successfully addresses all problem statement issues.")
    print(f"   📊 Total context items: {len(knowledge_store.context_items)}")
    print(f"   📁 Files explored: {len(knowledge_store.explored_files)}")
    print(f"   🗂️  File contents cached: {len(knowledge_store.code_content)}")
    
    return True


if __name__ == "__main__":
    try:
        test_knowledge_flow_integration()
        print("\n✅ Integration test PASSED - All problem statement issues resolved!")
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        raise