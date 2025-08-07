"""
Tests for the KnowledgeStore functionality
"""
import unittest
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Mock the rag_manager import to avoid dependency issues during testing
sys.modules['server.agent.rag_manager'] = MagicMock()

from server.agent.knowledge_store import KnowledgeStore, ContextType, ImportanceLevel, ContextItem


class TestKnowledgeStore(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.knowledge_store = KnowledgeStore()
    
    def test_add_context_basic(self):
        """Test basic context addition"""
        content = "This is test content"
        self.knowledge_store.add_context(
            content=content,
            context_type=ContextType.FILE_CONTENT,
            importance=ImportanceLevel.MEDIUM,
            source="test"
        )
        
        self.assertEqual(len(self.knowledge_store.context_items), 1)
        item = self.knowledge_store.context_items[0]
        self.assertEqual(item.content, content)
        self.assertEqual(item.context_type, ContextType.FILE_CONTENT)
        self.assertEqual(item.importance, ImportanceLevel.MEDIUM)
    
    def test_file_content_tracking(self):
        """Test that file content is properly tracked"""
        file_path = "/path/to/test.py"
        content = "def test(): pass"
        
        self.knowledge_store.add_context(
            content=content,
            context_type=ContextType.FILE_CONTENT,
            importance=ImportanceLevel.MEDIUM,
            source="test",
            metadata={"file_path": file_path}
        )
        
        # Check backward compatibility fields
        self.assertIn(file_path, self.knowledge_store.explored_files)
        self.assertEqual(self.knowledge_store.code_content[file_path], content)
    
    def test_tool_execution_update_file(self):
        """Test updating from file reading tool execution"""
        """Test updating from directory listing tool execution"""
        tool_args = {"path": "/test/dir"}
        result = "file1.py\nfile2.py\nsubdir/"
        
        self.knowledge_store.update_from_tool_execution("list_directory", tool_args, result)
        
        # Should create a context item
        self.assertEqual(len(self.knowledge_store.context_items), 1)
        item = self.knowledge_store.context_items[0]
        self.assertEqual(item.context_type, ContextType.DIRECTORY_LISTING)
        self.assertEqual(item.content, result)
        
        # Should add directory to explored files
        self.assertIn("/test/dir", self.knowledge_store.explored_files)
    
    def test_tool_execution_update_file(self):
        """Test updating from file reading tool execution"""
        tool_args = {"file_path": "/test/file.py"}
        result = "print('hello world')"
        
        self.knowledge_store.update_from_tool_execution("read_file", tool_args, result)
        
        # Should create a context item
        self.assertEqual(len(self.knowledge_store.context_items), 1)
        item = self.knowledge_store.context_items[0]
        self.assertEqual(item.context_type, ContextType.FILE_CONTENT)
        self.assertEqual(item.content, result)
        
        # Should update legacy fields
        self.assertIn("/test/file.py", self.knowledge_store.explored_files)
        self.assertEqual(self.knowledge_store.code_content["/test/file.py"], result)
    
    def test_context_prioritization(self):
        """Test that context items are properly prioritized"""
        # Add items with different importance levels
        self.knowledge_store.add_context("Low importance", ContextType.TASK_PROGRESS, ImportanceLevel.LOW, "test1")
        self.knowledge_store.add_context("Critical item", ContextType.EVALUATOR_FEEDBACK, ImportanceLevel.CRITICAL, "test2")
        self.knowledge_store.add_context("Medium item", ContextType.FILE_CONTENT, ImportanceLevel.MEDIUM, "test3")
        self.knowledge_store.add_context("High importance", ContextType.PLANNER_INSIGHT, ImportanceLevel.HIGH, "test4")
        
        # Get prioritized context
        prioritized = self.knowledge_store.get_prioritized_context(max_items=4)
        
        # Should be ordered by importance (critical first)
        self.assertEqual(prioritized[0].importance, ImportanceLevel.CRITICAL)
        self.assertEqual(prioritized[1].importance, ImportanceLevel.HIGH)
        self.assertEqual(prioritized[2].importance, ImportanceLevel.MEDIUM)
        self.assertEqual(prioritized[3].importance, ImportanceLevel.LOW)
    
    def test_context_summary_generation(self):
        """Test context summary generation"""
        # Add some context
        self.knowledge_store.add_context(
            "test content", 
            ContextType.FILE_CONTENT, 
            ImportanceLevel.HIGH, 
            "test",
            metadata={"file_path": "/test.py"}
        )
        self.knowledge_store.add_context(
            "task completed",
            ContextType.TASK_PROGRESS,
            ImportanceLevel.MEDIUM,
            "test"
        )
        
        summary = self.knowledge_store.build_context_summary()
        
        # Should contain explored files and progress
        self.assertIn("Explored files", summary)
        self.assertIn("/test.py", summary)
        self.assertIn("Progress so far", summary)
        self.assertIn("task completed", summary)
    
    def test_evaluator_feedback_storage(self):
        """Test that evaluator feedback is properly stored with high importance"""
        feedback = "FAILURE: Task not completed correctly"
        
        self.knowledge_store.add_evaluator_feedback(feedback)
        
        # Should create high importance context item
        self.assertEqual(len(self.knowledge_store.context_items), 1)
        item = self.knowledge_store.context_items[0]
        self.assertEqual(item.context_type, ContextType.EVALUATOR_FEEDBACK)
        self.assertEqual(item.importance, ImportanceLevel.HIGH)
        self.assertEqual(item.content, feedback)
    
    def test_planner_insight_storage(self):
        """Test that planner insights are stored"""
        insight = "Need to examine configuration files first"
        
        self.knowledge_store.add_planner_insight(insight, ImportanceLevel.MEDIUM)
        
        # Should create context item
        self.assertEqual(len(self.knowledge_store.context_items), 1)
        item = self.knowledge_store.context_items[0]
        self.assertEqual(item.context_type, ContextType.PLANNER_INSIGHT)
        self.assertEqual(item.importance, ImportanceLevel.MEDIUM)
        self.assertEqual(item.content, insight)
    
    def test_legacy_context_store_compatibility(self):
        """Test backward compatibility with legacy context store format"""
        # Add some data
        self.knowledge_store.add_context(
            "file content",
            ContextType.FILE_CONTENT,
            ImportanceLevel.MEDIUM,
            "test",
            metadata={"file_path": "/test.py"}
        )
        self.knowledge_store.add_context(
            "progress update",
            ContextType.TASK_PROGRESS,
            ImportanceLevel.LOW,
            "test"
        )
        
        # Get legacy format
        legacy = self.knowledge_store.get_legacy_context_store()
        
        # Should have the expected structure
        self.assertIn("explored_files", legacy)
        self.assertIn("code_content", legacy)
        self.assertIn("task_progress", legacy)
        
        self.assertIn("/test.py", legacy["explored_files"])
        self.assertEqual(legacy["code_content"]["/test.py"], "file content")
        self.assertIn("progress update", legacy["task_progress"])
    
    @patch('server.agent.knowledge_store.rag_manager')
    def test_rag_integration(self, mock_rag):
        """Test integration with RAG manager"""
        # Configure mock
        mock_rag.retrieve_knowledge.return_value = [
            {"text": "Previous knowledge from RAG"}
        ]
        
        # Add high importance item (should trigger RAG storage)
        self.knowledge_store.add_context(
            "Critical discovery",
            ContextType.FILE_CONTENT,
            ImportanceLevel.CRITICAL,
            "test"
        )
        
        # Should have called RAG add_knowledge
        mock_rag.add_knowledge.assert_called_once()
        
        # Test knowledge querying
        results = self.knowledge_store.query_relevant_knowledge("test query")
        
        # Should query RAG
        mock_rag.retrieve_knowledge.assert_called_once_with("test query", top_k=5)
        self.assertIn("Previous knowledge from RAG", results)


if __name__ == '__main__':
    unittest.main()