# tools/tool_manager.py

import json
from typing import Dict, Any, List, Optional
from collections import defaultdict
import re

from server.agent.tools.base_tool import BaseTool, ToolCapability
from server.agent.tools.tool_loader import ToolLoader


class ToolConflictResolver:
    """Handles conflicts when multiple tools have the same name."""
    
    @staticmethod
    def resolve_conflicts(tools: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve naming conflicts between tools.
        
        Args:
            tools: Dictionary of tool name to tool instance
            
        Returns:
            Dictionary with resolved tool names
        """
        # Group tools by base name (without version)
        grouped_tools = defaultdict(list)
        for name, tool in tools.items():
            base_name = ToolConflictResolver._get_base_name(name)
            grouped_tools[base_name].append((name, tool))
        
        resolved_tools = {}
        
        for base_name, tool_list in grouped_tools.items():
            if len(tool_list) == 1:
                # No conflict, use original name
                name, tool = tool_list[0]
                resolved_tools[base_name] = tool
            else:
                # Conflict detected, resolve by version
                resolved = ToolConflictResolver._resolve_version_conflict(tool_list)
                resolved_tools.update(resolved)
        
        return resolved_tools
    
    @staticmethod
    def _get_base_name(tool_name: str) -> str:
        """Extract base name from versioned tool name."""
        # Remove version suffix if present (e.g., "tool_v1_2_0" -> "tool")
        return re.sub(r'_v\d+(_\d+)*$', '', tool_name)
    
    @staticmethod
    def _resolve_version_conflict(tool_list: List[tuple]) -> Dict[str, Any]:
        """
        Resolve conflicts between multiple versions of the same tool.
        Keeps the highest version and adds versioned names for others.
        """
        resolved = {}
        
        # Sort by version (newest first)
        versioned_tools = []
        legacy_tools = []
        
        for name, tool in tool_list:
            if isinstance(tool, BaseTool):
                versioned_tools.append((name, tool))
            else:
                legacy_tools.append((name, tool))
        
        # Sort versioned tools by version
        versioned_tools.sort(key=lambda x: x[1].metadata.version, reverse=True)
        
        # Add the highest version with base name
        if versioned_tools:
            base_name = ToolConflictResolver._get_base_name(versioned_tools[0][0])
            resolved[base_name] = versioned_tools[0][1]
            
            # Add other versions with explicit version names
            for name, tool in versioned_tools[1:]:
                resolved[tool.get_full_name()] = tool
        
        # Add legacy tools with their original names
        for name, tool in legacy_tools:
            resolved[name] = tool
        
        return resolved


class ToolManager:
    """
    Enhanced tool manager with dynamic discovery, versioning, and conflict resolution.
    """

    def __init__(self, auto_discover: bool = True):
        """
        Initialize the tool manager.
        
        Args:
            auto_discover: Whether to automatically discover tools on initialization
        """
        self.tools: Dict[str, Any] = {}
        self.tool_loader = ToolLoader()
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        
        print("✅ Enhanced ToolManager initialized.")
        
        if auto_discover:
            self.discover_and_register_tools()

    def discover_and_register_tools(self):
        """Discover and register all available tools."""
        print("🔍 Discovering tools...")
        
        # Get all tool instances
        discovered_tools = self.tool_loader.create_tool_instances()
        
        # Resolve conflicts
        resolved_tools = ToolConflictResolver.resolve_conflicts(discovered_tools)
        
        # Register all resolved tools
        for name, tool in resolved_tools.items():
            self._register_tool(name, tool)
        
        print(f"✅ Tool discovery complete. {len(self.tools)} tools registered.")

    def register(self, tool_instance: Any, name: Optional[str] = None):
        """
        Register a single tool instance.
        
        Args:
            tool_instance: The tool instance to register
            name: Optional custom name for the tool
        """
        if name is None:
            name = self._extract_tool_name(tool_instance)
        
        self._register_tool(name, tool_instance)

    def _register_tool(self, name: str, tool_instance: Any):
        """Internal method to register a tool with the given name."""
        # Check for conflicts
        if name in self.tools:
            existing_tool = self.tools[name]
            if isinstance(tool_instance, BaseTool) and isinstance(existing_tool, BaseTool):
                # Compare versions
                if tool_instance.metadata.version > existing_tool.metadata.version:
                    print(f"🔄 Upgrading tool '{name}' from v{existing_tool.metadata.version} to v{tool_instance.metadata.version}")
                else:
                    print(f"⚠️  Skipping '{name}' v{tool_instance.metadata.version} (current: v{existing_tool.metadata.version})")
                    return
        
        self.tools[name] = tool_instance
        self._cache_tool_metadata(name, tool_instance)
        print(f"🔧 Tool '{name}' registered.")

    def _extract_tool_name(self, tool_instance: Any) -> str:
        """Extract the name from a tool instance."""
        if isinstance(tool_instance, BaseTool):
            return tool_instance.metadata.name
        
        # Legacy tool extraction
        if hasattr(tool_instance, 'function'):
            function_schema = tool_instance.function
            if isinstance(function_schema, dict):
                return function_schema.get("function", {}).get("name", "unknown_tool")
        
        return tool_instance.__class__.__name__.lower()

    def _cache_tool_metadata(self, name: str, tool_instance: Any):
        """Cache metadata for faster retrieval."""
        metadata = {}
        
        if isinstance(tool_instance, BaseTool):
            metadata = {
                "name": tool_instance.metadata.name,
                "version": tool_instance.metadata.version,
                "description": tool_instance.metadata.description,
                "capabilities": [cap.value for cap in tool_instance.metadata.capabilities],
                "author": tool_instance.metadata.author,
                "dependencies": tool_instance.metadata.dependencies,
                "tags": tool_instance.metadata.tags
            }
        else:
            # Legacy tool metadata
            if hasattr(tool_instance, 'function'):
                function_schema = tool_instance.function
                if isinstance(function_schema, dict):
                    metadata = {
                        "name": function_schema.get("function", {}).get("name", name),
                        "description": function_schema.get("function", {}).get("description", ""),
                        "version": "legacy",
                        "capabilities": [],
                        "legacy": True
                    }
        
        self._metadata_cache[name] = metadata

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns the JSON schemas of all registered tools."""
        return [tool.function for tool in self.tools.values()]

    def get_tools_by_capability(self, capability: ToolCapability) -> List[str]:
        matching_tools = []
        
        for name, metadata in self._metadata_cache.items():
            if capability.value in metadata.get("capabilities", []):
                matching_tools.append(name)
        
        return matching_tools

    def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        return self._metadata_cache.get(tool_name)

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all registered tools."""
        return self._metadata_cache.copy()

    def run_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        if tool_name not in self.tools:
            # Try to find a close match
            similar_tools = [name for name in self.tools.keys() if tool_name.lower() in name.lower()]
            if similar_tools:
                suggestion = f" Did you mean: {', '.join(similar_tools[:3])}?"
            else:
                suggestion = f" Available tools: {', '.join(list(self.tools.keys())[:5])}"
            raise Exception(f"Tool '{tool_name}' not found.{suggestion}")

        tool_instance = self.tools[tool_name]
        
        # Validate arguments if possible
        if isinstance(tool_instance, BaseTool):
            if not tool_instance.validate_arguments(tool_args):
                raise Exception(f"Invalid arguments for tool '{tool_name}'")
        
        # Let exceptions bubble up naturally - they indicate real tool failures
        return tool_instance.run(json.loads(tool_args))

    def reload_tools(self):
        """Reload all tools from the filesystem."""
        print("🔄 Reloading tools...")
        self.tools.clear()
        self._metadata_cache.clear()
        self.discover_and_register_tools()

    def get_tool_count(self) -> int:
        """Get the number of registered tools."""
        return len(self.tools)

    def get_tool_names(self) -> List[str]:
        """Get a list of all registered tool names."""
        return list(self.tools.keys())

    def get_tools_by_names(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """Get metadata for specific tools by name."""
        return [self.tools[name].function for name in tool_names if name in self.tools]

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self.tools


# Create a single, global instance that the entire application will share.
tool_manager = ToolManager()