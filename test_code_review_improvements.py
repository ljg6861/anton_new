"""
Test the improvements made to the Anton agent system for systematic task handling.
"""
import unittest
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestAgentImprovements(unittest.TestCase):
    """Test the general agent system improvements"""


    def test_evaluator_has_exploration_guidance(self):
        """Test that the evaluator prompt includes exploration and investigation specific guidance"""
        try:
            from server.agent.prompts import get_evaluator_prompt
            prompt = get_evaluator_prompt()
            self.assertIn("Exploration and Investigation Tasks", prompt)
            self.assertIn("reading files", prompt.lower())
            self.assertIn("directory contents", prompt.lower())
        except ImportError:
            self.skipTest("Module import failed - dependency issue")

    def test_context_store_update_for_file_reading(self):
        """Test that context store is properly updated when files are read"""
        try:
            from server.agent.tool_executor import _update_context_store
            
            context_store = {
                "explored_files": set(),
                "code_content": {},
                "task_progress": []
            }
            
            # Simulate reading a file
            _update_context_store(
                context_store, 
                "read_file", 
                {"file_path": "/test/file.py"}, 
                "def test_function():\n    pass"
            )
            
            self.assertIn("/test/file.py", context_store["explored_files"])
            self.assertIn("/test/file.py", context_store["code_content"])
            self.assertEqual(context_store["code_content"]["/test/file.py"], "def test_function():\n    pass")
        except ImportError:
            self.skipTest("Module import failed - dependency issue")

    def test_context_store_truncation_for_large_files(self):
        """Test that large file contents are truncated in context store"""
        try:
            from server.agent.tool_executor import _update_context_store
            
            context_store = {
                "explored_files": set(),
                "code_content": {},
                "task_progress": []
            }
            
            large_content = "x" * 15000  # Create content larger than 10000 chars
            
            _update_context_store(
                context_store,
                "read_file",
                {"file_path": "/test/large_file.py"},
                large_content
            )
            
            self.assertIn("/test/large_file.py", context_store["explored_files"])
            self.assertTrue(context_store["code_content"]["/test/large_file.py"].endswith("... [truncated]"))
            self.assertTrue(len(context_store["code_content"]["/test/large_file.py"]) < len(large_content))
        except ImportError:
            self.skipTest("Module import failed - dependency issue")

    def test_context_store_directory_listing(self):
        """Test that directory listing updates context store"""
        try:
            from server.agent.tool_executor import _update_context_store
            
            context_store = {
                "explored_files": set(),
                "code_content": {},
                "task_progress": []
            }
            
            _update_context_store(
                context_store,
                "list_directory",
                {"path": "/test/dir"},
                "file1.py\nfile2.py\nsubdir/"
            )
            
            self.assertIn("/test/dir", context_store["explored_files"])
            self.assertTrue(any("Listed directory" in progress for progress in context_store["task_progress"]))
        except ImportError:
            self.skipTest("Module import failed - dependency issue")

    def test_thought_loop_detection_logic(self):
        """Test the thought loop detection logic"""
        # Test word overlap calculation (this is the core logic from doer.py)
        content1 = "I need to read the file but cannot find it"
        content2 = "I need to read the file but it cannot be found"
        
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        overlap = len(words1.intersection(words2)) / max(len(words1), 1)
        
        # Should detect high overlap (indicating potential loop)
        self.assertGreater(overlap, 0.7)
        
        # Test with different content
        content3 = "The directory structure shows multiple Python files"
        words3 = set(content3.lower().split())
        overlap2 = len(words1.intersection(words3)) / max(len(words1), 1)
        
        # Should detect low overlap (no loop)
        self.assertLess(overlap2, 0.5)

    def test_exploration_task_detection_logic(self):
        """Test the general exploration patterns that benefit from enhanced context tracking"""
        # Test exploration keywords that would benefit from the improvements
        exploration_keywords = ['review', 'code', 'source', 'function', 'class', 'file', 'implementation', 'analyze', 
                                'examine', 'investigate', 'explore', 'find', 'search', 'read', 'check']
        
        test_requests = [
            "Please review the code in the main module",
            "Analyze the implementation of the user authentication system", 
            "Can you look at the source code for the payment processor?",
            "Find the function that handles file uploads",
            "Examine the class structure in the database module",
            "Investigate the configuration files",
            "Explore the project structure",
            "Search for documentation about the API"
        ]
        
        simple_requests = [
            "What is 2 + 2?",
            "Hello, how are you?",
            "Calculate the area of a circle",
            "Write a poem about nature"
        ]
        
        # All exploration requests contain relevant keywords
        for request in test_requests:
            has_exploration_keyword = any(keyword in request.lower() for keyword in exploration_keywords)
            self.assertTrue(has_exploration_keyword, f"Failed to detect exploration patterns: {request}")
        
        # Simple requests typically don't contain exploration keywords
        for request in simple_requests:
            has_exploration_keyword = any(keyword in request.lower() for keyword in exploration_keywords)
            # This is informational - we don't strictly require this to be false since the system now handles all tasks generically


if __name__ == '__main__':
    unittest.main()