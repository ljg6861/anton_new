"""
Comprehensive tests for the enhanced tool management system.
"""

import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.agent.tools.base_tool import BaseTool, ToolMetadata, ToolCapability
from server.agent.tools.tool_loader import ToolLoader
from server.agent.tools.tool_manager import ToolManager, ToolConflictResolver
from server.agent.tools.legacy_wrapper import LegacyToolWrapper


class MockLegacyTool:
    """Mock legacy tool for testing."""
    
    def __init__(self, name="mock_tool"):
        self.function = {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Mock tool {name}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_arg": {"type": "string", "description": "Test argument"}
                    },
                    "required": ["test_arg"]
                }
            }
        }
    
    def run(self, arguments):
        return f"Mock result for {arguments.get('test_arg', 'unknown')}"


class MockBaseTool(BaseTool):
    """Mock BaseTool implementation for testing."""
    
    def __init__(self, name="mock_base_tool", version="1.0.0"):
        self._name = name
        self._version = version
        super().__init__()
    
    def get_metadata(self):
        return ToolMetadata(
            name=self._name,
            version=self._version,
            description=f"Mock base tool {self._name}",
            capabilities=[ToolCapability.CODE_EXECUTION],
            author="Test Author"
        )
    
    def get_function_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": f"Mock base tool {self._name}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Input parameter"}
                    },
                    "required": ["input"]
                }
            }
        }
    
    def run(self, arguments):
        return f"BaseTool result for {arguments.get('input', 'unknown')}"


class TestBaseTool(unittest.TestCase):
    """Test the BaseTool interface."""
    
    def test_tool_creation(self):
        """Test creating a basic tool."""
        tool = MockBaseTool("test_tool", "1.2.3")
        
        self.assertEqual(tool.metadata.name, "test_tool")
        self.assertEqual(tool.metadata.version, "1.2.3")
        self.assertEqual(tool.metadata.description, "Mock base tool test_tool")
        self.assertEqual(tool.metadata.capabilities, [ToolCapability.CODE_EXECUTION])
    
    def test_function_schema(self):
        """Test function schema generation."""
        tool = MockBaseTool("schema_test")
        schema = tool.function
        
        self.assertEqual(schema["function"]["name"], "schema_test")
        self.assertIn("parameters", schema["function"])
    
    def test_tool_execution(self):
        """Test tool execution."""
        tool = MockBaseTool("exec_test")
        result = tool.run({"input": "test_input"})
        
        self.assertEqual(result, "BaseTool result for test_input")
    
    def test_argument_validation(self):
        """Test argument validation."""
        tool = MockBaseTool("validation_test")
        
        # Valid arguments
        self.assertTrue(tool.validate_arguments({"input": "test"}))
        
        # Missing required argument
        self.assertFalse(tool.validate_arguments({}))
    
    def test_version_compatibility(self):
        """Test version compatibility checking."""
        tool = MockBaseTool("version_test", "1.2.3")
        
        # Compatible versions
        self.assertTrue(tool.is_compatible_version("1.2.0"))
        self.assertTrue(tool.is_compatible_version("1.0.0"))
        
        # Incompatible versions
        self.assertFalse(tool.is_compatible_version("2.0.0"))
        self.assertFalse(tool.is_compatible_version("1.3.0"))


class TestLegacyWrapper(unittest.TestCase):
    """Test the legacy tool wrapper."""
    
    def test_wrapper_creation(self):
        """Test wrapping a legacy tool."""
        legacy_tool = MockLegacyTool("legacy_test")
        wrapped = LegacyToolWrapper(legacy_tool, [ToolCapability.FILE_SYSTEM])
        
        self.assertEqual(wrapped.metadata.name, "legacy_test")
        self.assertEqual(wrapped.metadata.version, "1.0.0")
        self.assertIn(ToolCapability.FILE_SYSTEM, wrapped.metadata.capabilities)
    
    def test_wrapper_execution(self):
        """Test executing through the wrapper."""
        legacy_tool = MockLegacyTool("execution_test")
        wrapped = LegacyToolWrapper(legacy_tool)
        
        result = wrapped.run({"test_arg": "wrapped_input"})
        self.assertEqual(result, "Mock result for wrapped_input")


class TestToolConflictResolver(unittest.TestCase):
    """Test tool conflict resolution."""
    
    def test_no_conflicts(self):
        """Test when there are no naming conflicts."""
        tools = {
            "tool1": MockBaseTool("tool1"),
            "tool2": MockBaseTool("tool2")
        }
        
        resolved = ToolConflictResolver.resolve_conflicts(tools)
        self.assertEqual(len(resolved), 2)
        self.assertIn("tool1", resolved)
        self.assertIn("tool2", resolved)
    
    def test_version_conflict_resolution(self):
        """Test resolving version conflicts."""
        tools = {
            "tool_v1_0_0": MockBaseTool("tool", "1.0.0"),
            "tool_v2_0_0": MockBaseTool("tool", "2.0.0"),
            "tool_v1_5_0": MockBaseTool("tool", "1.5.0")
        }
        
        resolved = ToolConflictResolver.resolve_conflicts(tools)
        
        # Should keep the highest version with base name
        self.assertIn("tool", resolved)
        self.assertEqual(resolved["tool"].metadata.version, "2.0.0")
        
        # Should have versioned names for others
        self.assertTrue(any("v1_" in name for name in resolved.keys()))


class TestToolManager(unittest.TestCase):
    """Test the enhanced tool manager."""
    
    def setUp(self):
        """Set up test environment."""
        self.manager = ToolManager(auto_discover=False)  # Don't auto-discover for tests
    
    def test_tool_registration(self):
        """Test registering tools."""
        tool = MockBaseTool("register_test")
        self.manager.register(tool)
        
        self.assertTrue(self.manager.has_tool("register_test"))
        self.assertEqual(self.manager.get_tool_count(), 1)
    
    def test_tool_execution(self):
        """Test executing registered tools."""
        tool = MockBaseTool("execute_test")
        self.manager.register(tool)
        
        result = self.manager.run_tool("execute_test", {"input": "test_execution"})
        self.assertEqual(result, "BaseTool result for test_execution")
    
    def test_missing_tool_execution(self):
        """Test executing non-existent tools."""
        result = self.manager.run_tool("nonexistent", {})
        self.assertIn("not found", result)
    
    def test_tool_metadata_caching(self):
        """Test metadata caching."""
        tool = MockBaseTool("metadata_test", "2.1.0")
        self.manager.register(tool)
        
        metadata = self.manager.get_tool_metadata("metadata_test")
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["version"], "2.1.0")
    
    def test_capability_filtering(self):
        """Test filtering tools by capability."""
        tool1 = MockBaseTool("cap_test1")  # Has CODE_EXECUTION
        tool2 = LegacyToolWrapper(MockLegacyTool("cap_test2"), [ToolCapability.FILE_SYSTEM])
        
        self.manager.register(tool1)
        self.manager.register(tool2)
        
        code_tools = self.manager.get_tools_by_capability(ToolCapability.CODE_EXECUTION)
        file_tools = self.manager.get_tools_by_capability(ToolCapability.FILE_SYSTEM)
        
        self.assertIn("cap_test1", code_tools)
        self.assertIn("cap_test2", file_tools)


class TestToolLoader(unittest.TestCase):
    """Test the tool loader functionality."""
    
    def test_tool_file_listing(self):
        """Test listing tool files."""
        loader = ToolLoader()
        files = loader.list_tool_files()
        
        # Should exclude system files
        self.assertNotIn('__init__', files)
        self.assertNotIn('tool_loader', files)
        self.assertNotIn('tool_manager', files)
        self.assertNotIn('base_tool', files)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestBaseTool,
        TestLegacyWrapper,
        TestToolConflictResolver,
        TestToolManager,
        TestToolLoader
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("🧪 Running Enhanced Tool Management System Tests...")
    print("=" * 60)
    
    result = run_tests()
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        for test, error in result.failures + result.errors:
            print(f"   - {test}: {error.split('\\n')[0]}")
    
    print(f"📊 Tests run: {result.testsRun}")
    print(f"📊 Failures: {len(result.failures)}")
    print(f"📊 Errors: {len(result.errors)}")