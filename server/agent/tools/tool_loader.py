import os
import importlib
import inspect
from typing import List, Dict, Any, Optional, Type
from pathlib import Path

from server.agent.tools.base_tool import BaseTool, ToolCapability


class ToolLoader:
    """
    Enhanced tool loader that dynamically discovers and loads tools from the filesystem.
    Supports automatic discovery, versioning, and conflict resolution.
    """
    
    def __init__(self, tools_directory: Optional[str] = None):
        """
        Initialize the tool loader.
        
        Args:
            tools_directory: Path to the tools directory. If None, uses the current directory.
        """
        if tools_directory is None:
            self.tools_directory = Path(__file__).parent
        else:
            self.tools_directory = Path(tools_directory)
    
    def list_tool_files(self) -> List[str]:
        """Returns a list of available tool filenames from the tools directory."""
        tool_files = [
            f.stem for f in self.tools_directory.glob("*.py")
            if f.name not in ['__init__.py', 'tool_loader.py', 'tool_manager.py', 'tool_defs.py', 'base_tool.py']
        ]
        return tool_files
    
    def discover_tools(self) -> Dict[str, Type[BaseTool]]:
        """
        Dynamically discover all tools in the tools directory.
        
        Returns:
            Dictionary mapping tool names to their classes
        """
        discovered_tools = {}
        tool_files = self.list_tool_files()
        
        for tool_file in tool_files:
            try:
                tools_in_file = self._load_tools_from_file(tool_file)
                discovered_tools.update(tools_in_file)
            except Exception as e:
                print(f"⚠️  Warning: Failed to load tools from {tool_file}.py: {e}")
                continue
        
        return discovered_tools
    
    def _load_tools_from_file(self, module_name: str) -> Dict[str, Type[BaseTool]]:
        """
        Load all tool classes from a specific module file.
        
        Args:
            module_name: Name of the module to load (without .py extension)
            
        Returns:
            Dictionary mapping tool names to their classes
        """
        tools = {}
        
        try:
            # Import the module dynamically
            full_module_name = f"server.agent.tools.{module_name}"
            module = importlib.import_module(full_module_name)
            
            # Find all classes that inherit from BaseTool or have the expected interface
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_tool_class(obj) and obj.__module__ == full_module_name:
                    # Try to instantiate and get the tool name
                    try:
                        tool_instance = obj()
                        tool_name = self._get_tool_name(tool_instance)
                        tools[tool_name] = obj
                        print(f"🔧 Discovered tool: {tool_name} from {module_name}.py")
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to instantiate {name} from {module_name}.py: {e}")
                        continue
        
        except ImportError as e:
            # Don't warn for missing optional dependencies
            if "No module named" in str(e):
                missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
                print(f"ℹ️  Info: Skipping {module_name}.py due to missing dependency: {missing_module}")
            else:
                print(f"⚠️  Warning: Could not import module {module_name}: {e}")
        
        return tools
    
    def _is_tool_class(self, cls: Type) -> bool:
        """
        Check if a class is a valid tool class.
        
        Args:
            cls: Class to check
            
        Returns:
            True if the class is a valid tool, False otherwise
        """
        # Check if it inherits from BaseTool
        if issubclass(cls, BaseTool):
            return True
        
        # Check if it has the legacy tool interface (function attribute and run method)
        if hasattr(cls, 'function') and hasattr(cls, 'run'):
            return True
        
        return False
    
    def _get_tool_name(self, tool_instance: Any) -> str:
        """
        Extract the tool name from a tool instance.
        
        Args:
            tool_instance: Instance of a tool
            
        Returns:
            The tool's name
        """
        # For BaseTool instances, use metadata
        if isinstance(tool_instance, BaseTool):
            return tool_instance.metadata.name
        
        # For legacy tools, extract from function schema
        if hasattr(tool_instance, 'function'):
            function_schema = tool_instance.function
            if isinstance(function_schema, dict):
                return function_schema.get("function", {}).get("name", "unknown_tool")
        
        # Fallback to class name
        return tool_instance.__class__.__name__.lower()
    
    def create_tool_instances(self) -> Dict[str, Any]:
        """
        Discover and instantiate all available tools.
        
        Returns:
            Dictionary mapping tool names to their instances
        """
        tool_classes = self.discover_tools()
        tool_instances = {}
        
        for tool_name, tool_class in tool_classes.items():
            try:
                instance = tool_class()
                tool_instances[tool_name] = instance
            except Exception as e:
                print(f"⚠️  Warning: Failed to instantiate {tool_name}: {e}")
                continue
        
        return tool_instances
    
    def get_tools_by_capability(self, capability: ToolCapability) -> Dict[str, Any]:
        """
        Get all tools that have a specific capability.
        
        Args:
            capability: The capability to filter by
            
        Returns:
            Dictionary of tools with the specified capability
        """
        all_tools = self.create_tool_instances()
        filtered_tools = {}
        
        for name, tool in all_tools.items():
            if isinstance(tool, BaseTool):
                if capability in tool.metadata.capabilities:
                    filtered_tools[name] = tool
            # For legacy tools, we can't filter by capability
        
        return filtered_tools
    
    def validate_tool_compatibility(self, tool_name: str, required_version: str) -> bool:
        """
        Check if a tool meets version requirements.
        
        Args:
            tool_name: Name of the tool to check
            required_version: Required version string
            
        Returns:
            True if the tool meets requirements, False otherwise
        """
        tools = self.create_tool_instances()
        tool = tools.get(tool_name)
        
        if not tool:
            return False
        
        if isinstance(tool, BaseTool):
            return tool.is_compatible_version(required_version)
        
        # Legacy tools are assumed compatible
        return True