"""
Compatibility wrapper for legacy tools to work with the new tool system.
"""

from typing import Dict, Any
from server.agent.tools.base_tool import BaseTool, ToolMetadata, ToolCapability


class LegacyToolWrapper(BaseTool):
    """
    Wrapper class to make legacy tools compatible with the new BaseTool interface.
    """
    
    def __init__(self, legacy_tool_instance: Any, capabilities: list = None):
        """
        Initialize the wrapper with a legacy tool instance.
        
        Args:
            legacy_tool_instance: The legacy tool instance to wrap
            capabilities: List of capabilities for this tool
        """
        self.legacy_tool = legacy_tool_instance
        self._capabilities = capabilities or []
        super().__init__()
    
    def get_metadata(self) -> ToolMetadata:
        """Extract metadata from the legacy tool."""
        function_schema = getattr(self.legacy_tool, 'function', {})
        function_info = function_schema.get('function', {})
        
        name = function_info.get('name', self.legacy_tool.__class__.__name__.lower())
        description = function_info.get('description', f'Legacy tool: {name}')
        
        return ToolMetadata(
            name=name,
            version="1.0.0",  # Default version for legacy tools
            description=description,
            capabilities=self._capabilities,
            author="Legacy",
            tags=["legacy"]
        )
    
    def get_function_schema(self) -> Dict[str, Any]:
        """Return the legacy tool's function schema."""
        return getattr(self.legacy_tool, 'function', {})
    
    def run(self, arguments: Dict[str, Any]) -> str:
        """Delegate to the legacy tool's run method."""
        return self.legacy_tool.run(arguments)


def create_legacy_tool_wrappers() -> Dict[str, BaseTool]:
    """
    Create wrapped versions of all legacy tools.
    Handles missing dependencies gracefully.
    
    Returns:
        Dictionary mapping tool names to wrapped tool instances
    """
    wrapped_tools = {}
    
    # Try to import and wrap each tool, handling missing dependencies
    tool_configs = [
        ('coding', 'ExecutePythonCode', [ToolCapability.CODE_EXECUTION]),
        ('file_management', 'WriteFileTool', [ToolCapability.FILE_SYSTEM]),
        ('file_management', 'ReadFileTool', [ToolCapability.FILE_SYSTEM]),
        ('file_management', 'ListDirectoryTool', [ToolCapability.FILE_SYSTEM]),
        ('web_search', 'WebSearchTool', [ToolCapability.WEB_SEARCH, ToolCapability.EXTERNAL_API]),
        ('tool_creation.tool_creator', 'ToolCreator', [ToolCapability.TOOL_CREATION]),
    ]
    
    for module_path, class_name, capabilities in tool_configs:
        try:
            module = __import__(f'server.agent.tools.{module_path}', fromlist=[class_name])
            tool_class = getattr(module, class_name)
            tool_instance = tool_class()
            
            # Extract tool name from function schema
            tool_name = tool_instance.function.get('function', {}).get('name', class_name.lower())
            
            wrapped_tools[tool_name] = LegacyToolWrapper(tool_instance, capabilities)
            print(f"🔧 Wrapped legacy tool: {tool_name}")
            
        except ImportError as e:
            missing_dep = str(e).split("'")[1] if "'" in str(e) else "unknown"
            print(f"ℹ️  Skipping {class_name} due to missing dependency: {missing_dep}")
        except Exception as e:
            print(f"⚠️  Failed to wrap {class_name}: {e}")
    
    # Try to wrap Git tools
    try:
        from server.agent.tools.git import GitManagementSkill
        git_skill = GitManagementSkill()
        git_tools = git_skill.get_tools()
        
        for git_tool in git_tools:
            tool_name = git_tool.function.get('function', {}).get('name', 'unknown_git_tool')
            wrapped_tools[tool_name] = LegacyToolWrapper(
                git_tool,
                [ToolCapability.GIT_OPERATIONS, ToolCapability.FILE_SYSTEM]
            )
            print(f"🔧 Wrapped git tool: {tool_name}")
    
    except ImportError as e:
        print(f"ℹ️  Skipping Git tools due to missing dependency: {e}")
    except Exception as e:
        print(f"⚠️  Failed to wrap Git tools: {e}")
    
    return wrapped_tools