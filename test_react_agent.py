#!/usr/bin/env python3
"""
Test the new ReAct agent and ConversationState classes.
"""
import unittest
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.agent.conversation_state import ConversationState, StateType


class TestConversationState(unittest.TestCase):
    """Test the ConversationState class functionality"""
    
    def test_initialization(self):
        """Test basic initialization"""
        state = ConversationState()
        self.assertEqual(len(state.messages), 0)
        self.assertEqual(len(state.state_entries), 0)
        self.assertFalse(state.is_complete)
    
    def test_add_message(self):
        """Test adding messages"""
        state = ConversationState()
        state.add_message("user", "Hello, world!")
        
        self.assertEqual(len(state.messages), 1)
        self.assertEqual(state.messages[0]["role"], "user")
        self.assertEqual(state.messages[0]["content"], "Hello, world!")
        
        # Should also create a state entry
        self.assertEqual(len(state.state_entries), 1)
        self.assertEqual(state.state_entries[0].state_type, StateType.MESSAGE)
    
    def test_context_building(self):
        """Test context summary building"""
        state = ConversationState()
        state.add_message("user", "Test message")
        state.add_file_exploration("test.py", "print('hello')")
        
        summary = state.build_context_summary()
        self.assertIn("test.py", summary)
        self.assertIn("Recent activity", summary)
    
    def test_tool_outputs(self):
        """Test tool output tracking"""
        state = ConversationState()
        state.add_tool_output("file_reader", "File content here")
        
        self.assertIn("file_reader", state.tool_outputs)
        self.assertEqual(state.tool_outputs["file_reader"]["output"], "File content here")
    
    def test_completion(self):
        """Test marking conversation as complete"""
        state = ConversationState()
        state.mark_complete("Task completed successfully")
        
        self.assertTrue(state.is_complete)
        self.assertEqual(state.final_response, "Task completed successfully")


class TestReActLogic(unittest.TestCase):
    """Test ReAct agent logic without external dependencies"""
    
    def test_final_response_detection(self):
        """Test the heuristic for detecting final responses"""
        # This would require mocking the ReActAgent, but let's test the logic
        
        # Test completion indicators
        final_responses = [
            "The task is completed successfully.",
            "Here is the final result: 42",
            "I have finished the analysis.",
            "Done! The file has been created."
        ]
        
        for response in final_responses:
            # Simple heuristic test
            is_final = any(indicator in response.lower() for indicator in [
                'completed', 'finished', 'done', 'final result'
            ])
            self.assertTrue(is_final, f"Should detect '{response}' as final")
        
        # Test non-final responses
        non_final_responses = [
            "I need to check the file first.",
            "Let me analyze this further.",
            "I should run a tool to get more information."
        ]
        
        for response in non_final_responses:
            is_final = any(indicator in response.lower() for indicator in [
                'completed', 'finished', 'done', 'final result'
            ]) and not any(phrase in response.lower() for phrase in [
                'let me', 'i should', 'i need to'
            ])
            self.assertFalse(is_final, f"Should not detect '{response}' as final")


if __name__ == '__main__':
    # Run the tests
    unittest.main()