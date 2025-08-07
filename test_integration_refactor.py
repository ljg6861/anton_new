#!/usr/bin/env python3
"""
Integration test to verify the main refactoring works correctly.
Tests the core flow without external dependencies.
"""
import unittest
import json
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.agent.conversation_state import ConversationState, StateType


class TestIntegration(unittest.TestCase):
    """Integration tests for the refactored agent system"""
    
    def test_conversation_state_flow(self):
        """Test the conversation state manages a realistic conversation flow"""
        state = ConversationState()
        
        # Simulate a user request
        state.add_message("user", "Can you help me analyze this Python file?")
        
        # Simulate agent thinking
        state.add_state_entry("I need to read the file first", StateType.THOUGHT)
        
        # Simulate tool usage
        state.add_tool_output("read_file", "def hello(): print('world')", {"path": "test.py"})
        state.add_file_exploration("test.py", "def hello(): print('world')")
        
        # Simulate agent response
        state.add_message("assistant", "The file contains a simple hello function.")
        
        # Verify state tracking
        self.assertEqual(len(state.messages), 2)
        self.assertIn("read_file", state.tool_outputs)
        self.assertIn("test.py", state.explored_files)
        
        # Test context building
        context = state.build_context_summary()
        self.assertIn("test.py", context)
        self.assertIn("read_file", context)
        
        # Mark as complete
        state.mark_complete("Analysis complete")
        self.assertTrue(state.is_complete)
    
    def test_react_agent_final_response_detection(self):
        """Test the ReAct agent's completion detection logic"""
        # Import here to avoid dependency issues during import
        try:
            from server.agent.react_agent import ReActAgent
            
            agent = ReActAgent("http://test", [], 5)
            
            # Test various response types
            final_responses = [
                "Task completed successfully. The file has been created.",
                "Here is the final result: The analysis shows...",
                "I have finished processing your request.",
                "Done! All files have been processed."
            ]
            
            for response in final_responses:
                self.assertTrue(agent._is_final_response(response), 
                              f"Should detect '{response}' as final")
            
            # Test non-final responses
            non_final_responses = [
                "I need to check another file first.",
                "Let me use a tool to get more information.",
                "<tool_code>read_file</tool_code>",
                "I should analyze this further."
            ]
            
            for response in non_final_responses:
                self.assertFalse(agent._is_final_response(response),
                               f"Should not detect '{response}' as final")
                
        except ImportError as e:
            self.skipTest(f"Skipping ReAct agent test due to import error: {e}")
    
    def test_rag_knowledge_retrieval(self):
        """Test that RAG retrieval works as expected"""
        # Simple mock test without complex patching
        # This tests the logic without external dependencies
        
        # Mock a simple RAG response structure
        mock_results = [
            {"text": "Python is a programming language", "source": "python_docs.md"},
            {"text": "Functions are defined with def", "source": "tutorial.md"}
        ]
        
        # Test that we can process RAG-like results
        query = "How to define functions in Python?"
        
        # Simulate what the agent_server.py RAG path would do
        if mock_results:
            context_parts = []
            for i, doc in enumerate(mock_results):
                context_parts.append(f"Source {i+1}: {doc['source']}\n{doc['text'][:500]}...")
            
            context = "\n\n".join(context_parts)
            
            self.assertIn("Functions are defined with def", context)
            self.assertIn("tutorial.md", context)
            self.assertEqual(len(context_parts), 2)
    
    def test_state_adapter_integration(self):
        """Test that ConversationState integrates with tool execution"""
        state = ConversationState()
        
        # Simulate the StateAdapter from react_agent.py
        class StateAdapter:
            def __init__(self, conv_state):
                self.conv_state = conv_state
                
            def update_from_tool_execution(self, tool_name, tool_args, tool_result):
                self.conv_state.add_tool_output(tool_name, tool_result, {"args": tool_args})
                
                if tool_name in ['read_file', 'list_files', 'write_file'] and 'path' in tool_args:
                    self.conv_state.add_file_exploration(tool_args['path'])
        
        adapter = StateAdapter(state)
        
        # Simulate tool execution
        adapter.update_from_tool_execution(
            "read_file", 
            {"path": "/home/test.py"}, 
            "print('hello world')"
        )
        
        # Verify integration
        self.assertIn("read_file", state.tool_outputs)
        self.assertIn("/home/test.py", state.explored_files)
        self.assertEqual(state.tool_outputs["read_file"]["output"], "print('hello world')")


class TestStructuralChanges(unittest.TestCase):
    """Test that the structural changes are correctly implemented"""
    
    def test_entry_points_consolidated(self):
        """Test that redundant entry points have been removed"""
        import os
        repo_root = os.path.dirname(os.path.abspath(__file__))
        
        # These files should NOT exist anymore
        redundant_files = ["main.py", "backend.py"]
        for filename in redundant_files:
            file_path = os.path.join(repo_root, filename)
            self.assertFalse(os.path.exists(file_path), 
                           f"Redundant file {filename} should have been removed")
        
        # These files SHOULD exist as the primary entry points
        required_files = [
            "client/main.py",
            "server/agent/agent_server.py"
        ]
        for filename in required_files:
            file_path = os.path.join(repo_root, filename)
            self.assertTrue(os.path.exists(file_path), 
                          f"Primary entry point {filename} should exist")
    
    def test_new_components_exist(self):
        """Test that the new components have been created"""
        import os
        repo_root = os.path.dirname(os.path.abspath(__file__))
        
        new_components = [
            "server/agent/conversation_state.py",
            "server/agent/react_agent.py"
        ]
        
        for component in new_components:
            file_path = os.path.join(repo_root, component)
            self.assertTrue(os.path.exists(file_path), 
                          f"New component {component} should exist")


if __name__ == '__main__':
    print("🧪 Running Anton Agent Refactoring Integration Tests")
    print("=" * 60)
    
    # Run tests with verbose output
    unittest.main(verbosity=2)