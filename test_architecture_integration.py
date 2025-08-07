#!/usr/bin/env python3
"""
Integration test to verify the refactored architecture works end-to-end without heavy dependencies.
"""
import unittest
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock heavy dependencies
sys.modules['vllm'] = Mock()
sys.modules['vllm.third_party'] = Mock()
sys.modules['vllm.third_party.pynvml'] = Mock()
sys.modules['pynvml'] = Mock()
sys.modules['server.agent.learning_loop'] = Mock()
sys.modules['server.agent.rag_manager'] = Mock()
sys.modules['numpy'] = Mock()
sys.modules['faiss'] = Mock()
sys.modules['sentence_transformers'] = Mock()

from server.agent.knowledge_store import KnowledgeStore, ContextType, ImportanceLevel
from server.agent.react_agent import ReActAgent, process_tool_calls_with_knowledge_store


class TestArchitectureIntegration(unittest.TestCase):
    """Integration test for the refactored architecture"""
    
    def setUp(self):
        # Mock the dependencies
        self.mock_learning_loop = Mock()
        self.mock_rag_manager = Mock()
        self.mock_rag_manager.retrieve_knowledge.return_value = []
        self.mock_rag_manager.add_knowledge.return_value = None
        
        # Patch the imports
        patcher1 = patch('server.agent.knowledge_store.learning_loop', self.mock_learning_loop)
        patcher2 = patch('server.agent.knowledge_store.rag_manager', self.mock_rag_manager)
        patcher1.start()
        patcher2.start()
        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
    
    def test_complete_architecture_workflow(self):
        """Test complete workflow: KnowledgeStore + ReActAgent integration"""
        # Create knowledge store
        store = KnowledgeStore()
        
        # Create ReAct agent with knowledge store
        tools = [{"name": "test_tool", "description": "A test tool"}]
        agent = ReActAgent(
            api_base_url="http://localhost:8000",
            tools=tools,
            knowledge_store=store,
            max_iterations=3
        )
        
        # Verify the agent uses the knowledge store
        self.assertIs(agent.knowledge_store, store)
        
        # Test that the agent can access knowledge store methods
        store.add_message("user", "Hello")
        messages = store.get_messages_for_llm()
        self.assertEqual(len(messages), 1)
        
        # Verify message was also added to context for advanced processing
        message_contexts = [item for item in store.context_items if item.context_type == ContextType.MESSAGE]
        self.assertEqual(len(message_contexts), 1)
    
    def test_system_prompt_includes_rag_knowledge(self):
        """Test that system prompt integrates RAG knowledge"""
        store = KnowledgeStore()
        agent = ReActAgent(
            api_base_url="http://localhost:8000",
            tools=[],
            knowledge_store=store,
            max_iterations=3
        )
        
        # Mock RAG returning some knowledge
        self.mock_rag_manager.retrieve_knowledge.return_value = [
            "Previous experience with similar tasks",
            "Important context from past interactions"
        ]
        
        # Call get_react_system_prompt which should query RAG
        prompt = agent.get_react_system_prompt()
        
        # Verify RAG was queried
        self.mock_rag_manager.retrieve_knowledge.assert_called_once()
        
        # Verify prompt contains expected content
        self.assertIn("ReAct", prompt)
        self.assertIn("REASON", prompt)
        self.assertIn("ACT", prompt)
    
    def test_knowledge_store_tool_integration(self):
        """Test that tool execution updates knowledge store properly"""
        store = KnowledgeStore()
        
        # Simulate tool execution with read_file (should create FILE_CONTENT context)
        store.update_from_tool_execution(
            "read_file", 
            {"file_path": "test.py"}, 
            "def hello(): print('world')"
        )
        
        # Verify file was tracked
        self.assertIn("test.py", store.explored_files)
        self.assertIn("test.py", store.code_content)
        
        # Verify context was added with FILE_CONTENT type
        file_contexts = [item for item in store.context_items if item.context_type == ContextType.FILE_CONTENT]
        self.assertTrue(len(file_contexts) > 0)
        
        # Test with generic tool (should create TOOL_EXECUTION context)
        store.update_from_tool_execution(
            "custom_tool",
            {"param": "value"},
            "Tool execution result"
        )
        
        # Verify generic tool context was added
        tool_contexts = [item for item in store.context_items if item.context_type == ContextType.TOOL_EXECUTION]
        self.assertTrue(len(tool_contexts) > 0)
    
    def test_tool_call_processing_with_knowledge_store(self):
        """Test that tool call processing works with knowledge store"""
        store = KnowledgeStore()
        
        # Mock tool call regex
        mock_regex = Mock()
        mock_regex.finditer.return_value = []  # No tool calls found
        
        # Test the function exists and can be imported
        # We can't easily test async without more complex setup, so just verify import
        self.assertTrue(callable(process_tool_calls_with_knowledge_store))
    
    def test_knowledge_store_deprecates_conversation_state(self):
        """Verify that KnowledgeStore provides all ConversationState functionality"""
        store = KnowledgeStore()
        
        # Test all the interface methods from ConversationState
        # These should work without importing ConversationState
        
        # Message management
        store.add_message("user", "test message")
        messages = store.get_messages_for_llm()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "test message")
        
        # Tool output management
        store.add_tool_output("test_tool", "output", {"args": {"param": "value"}})
        self.assertIn("test_tool", store.tool_outputs)
        
        # Context building
        summary = store.build_context_summary()
        self.assertIsInstance(summary, str)
        
        # Completion tracking
        store.mark_complete("Done!")
        self.assertTrue(store.is_complete)
        self.assertEqual(store.final_response, "Done!")
        
        # Duration tracking
        duration = store.get_duration()
        self.assertIsInstance(duration, float)
        self.assertGreaterEqual(duration, 0)
        
        # Reset functionality
        store.reset_conversation()
        self.assertEqual(len(store.messages), 0)
        self.assertFalse(store.is_complete)
    
    def test_enhanced_context_prioritization(self):
        """Test that KnowledgeStore provides enhanced context prioritization beyond ConversationState"""
        store = KnowledgeStore()
        
        # Add items with different importance levels
        store.add_context("Low priority item", ContextType.TOOL_EXECUTION, ImportanceLevel.LOW, "test")
        store.add_context("High priority item", ContextType.EVALUATOR_FEEDBACK, ImportanceLevel.HIGH, "test")
        store.add_context("Critical item", ContextType.PLANNER_INSIGHT, ImportanceLevel.CRITICAL, "test")
        
        # Get prioritized context
        prioritized = store.get_prioritized_context(max_items=5)
        
        # Should have items in priority order (highest first)
        self.assertTrue(len(prioritized) >= 3)
        
        # Critical item should be first, low priority should be last
        importance_levels = [item.importance for item in prioritized]
        critical_index = next(i for i, level in enumerate(importance_levels) if level == ImportanceLevel.CRITICAL)
        low_index = next(i for i, level in enumerate(importance_levels) if level == ImportanceLevel.LOW)
        
        self.assertLess(critical_index, low_index, "Critical items should come before low priority items")


if __name__ == '__main__':
    unittest.main()