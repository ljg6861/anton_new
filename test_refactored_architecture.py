#!/usr/bin/env python3
"""
Test the refactored architecture that uses KnowledgeStore exclusively and removes ConversationState dependency.
"""
import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the heavy dependencies to avoid import errors
sys.modules['server.agent.learning_loop'] = Mock()
sys.modules['server.agent.rag_manager'] = Mock()
sys.modules['numpy'] = Mock()
sys.modules['faiss'] = Mock()
sys.modules['sentence_transformers'] = Mock()

# Now import our modules
from server.agent.knowledge_store import KnowledgeStore, ContextType, ImportanceLevel


class TestRefactoredArchitecture(unittest.TestCase):
    """Test the refactored architecture with KnowledgeStore as the central state manager"""
    
    def setUp(self):
        # Mock the learning_loop and rag_manager to avoid dependencies
        self.mock_learning_loop = Mock()
        self.mock_rag_manager = Mock()
        self.mock_rag_manager.retrieve_knowledge.return_value = []
        self.mock_rag_manager.add_knowledge.return_value = None
        
        # Patch the imports in knowledge_store
        patcher1 = patch('server.agent.knowledge_store.learning_loop', self.mock_learning_loop)
        patcher2 = patch('server.agent.knowledge_store.rag_manager', self.mock_rag_manager)
        patcher1.start()
        patcher2.start()
        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
    
    def test_knowledge_store_conversation_management(self):
        """Test that KnowledgeStore can handle conversation state"""
        store = KnowledgeStore()
        
        # Test message management
        store.add_message("user", "Hello, can you help me?")
        store.add_message("assistant", "Of course! How can I help you?")
        
        messages = store.get_messages_for_llm()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Hello, can you help me?")
        
        # Test tool output management
        store.add_tool_output("file_reader", "File contents here", {"args": {"path": "test.py"}})
        self.assertIn("file_reader", store.tool_outputs)
        
        # Test completion marking
        store.mark_complete("Task completed successfully")
        self.assertTrue(store.is_complete)
        self.assertEqual(store.final_response, "Task completed successfully")
    
    def test_knowledge_store_context_integration(self):
        """Test that conversation messages are properly integrated into context system"""
        store = KnowledgeStore()
        
        # Add messages and verify they become context items
        store.add_message("user", "What is the capital of France?")
        
        # Check that message was added to context items
        message_contexts = [item for item in store.context_items if item.context_type == ContextType.MESSAGE]
        self.assertEqual(len(message_contexts), 1)
        self.assertEqual(message_contexts[0].content, "What is the capital of France?")
        self.assertEqual(message_contexts[0].importance, ImportanceLevel.MEDIUM)
    
    def test_react_agent_initialization_with_knowledge_store(self):
        """Test that ReActAgent properly initializes with KnowledgeStore"""
        store = KnowledgeStore()
        tools = [{"name": "test_tool", "description": "A test tool"}]
        
        # Import here to avoid early import issues
        from server.agent.react_agent import ReActAgent
        
        agent = ReActAgent(
            api_base_url="http://localhost:8000",
            tools=tools,
            knowledge_store=store,
            max_iterations=5
        )
        
        self.assertIs(agent.knowledge_store, store)
        self.assertEqual(agent.tools, tools)
        self.assertEqual(agent.max_iterations, 5)
    
    def test_knowledge_store_rag_integration(self):
        """Test that KnowledgeStore properly integrates with RAG for knowledge retrieval"""
        store = KnowledgeStore()
        
        # Test query method exists and returns list
        results = store.query_relevant_knowledge("test query", max_results=3)
        self.assertIsInstance(results, list)
        # Verify rag_manager was called
        self.mock_rag_manager.retrieve_knowledge.assert_called_once_with("test query", top_k=3)
    
    def test_conversation_reset(self):
        """Test that conversation state can be reset while preserving learned context"""
        store = KnowledgeStore()
        
        # Add some conversation data
        store.add_message("user", "Hello")
        store.add_tool_output("test_tool", "result")
        store.mark_complete("Done")
        
        # Add some context that should persist
        store.add_context("Important knowledge", ContextType.PLANNER_INSIGHT, ImportanceLevel.HIGH, "test")
        
        initial_context_count = len(store.context_items)
        
        # Reset conversation
        store.reset_conversation()
        
        # Check conversation state is reset
        self.assertEqual(len(store.messages), 0)
        self.assertEqual(len(store.tool_outputs), 0)
        self.assertFalse(store.is_complete)
        self.assertEqual(store.final_response, "")
        
        # Check that learned context is preserved
        self.assertEqual(len(store.context_items), initial_context_count)
    
    def test_build_context_summary_integration(self):
        """Test that context summary includes both old and new functionality"""
        store = KnowledgeStore()
        
        # Add various types of data
        store.add_message("user", "Test message")
        store.add_context("Important file analysis", ContextType.FILE_CONTENT, ImportanceLevel.HIGH, "analyzer")
        store.explored_files.add("test.py")
        store.task_progress.append("Analyzed code structure")
        
        summary = store.build_context_summary()
        
        # Verify summary contains expected elements
        self.assertIn("test.py", summary)
        self.assertIn("Analyzed code structure", summary)
        self.assertIn("Key insights", summary)
    
    def test_knowledge_store_replaces_conversation_state_interface(self):
        """Test that KnowledgeStore provides all the interface methods that ConversationState had"""
        store = KnowledgeStore()
        
        # Test all the methods that ConversationState had
        self.assertTrue(hasattr(store, 'add_message'))
        self.assertTrue(hasattr(store, 'add_tool_output'))
        self.assertTrue(hasattr(store, 'get_messages_for_llm'))
        self.assertTrue(hasattr(store, 'mark_complete'))
        self.assertTrue(hasattr(store, 'get_duration'))
        self.assertTrue(hasattr(store, 'build_context_summary'))
        
        # Test that these methods work as expected
        store.add_message("user", "test")
        self.assertEqual(len(store.messages), 1)
        
        messages = store.get_messages_for_llm()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "test")
        
        duration = store.get_duration()
        self.assertIsInstance(duration, float)
        self.assertGreaterEqual(duration, 0)


if __name__ == '__main__':
    unittest.main()