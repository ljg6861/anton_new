"""
Formats tools for inclusion in prompts
"""
from typing import List, Dict

from .memory_manager import MemoryManager


class ToolFormatter:
    """Formats tools for inclusion in prompts"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager
    
    def format_tools_compact(self, tools: List[Dict]) -> str:
        """Format tools in a very compact way for tight token budget"""
        if not tools:
            return "No tools available"
        
        # Only show tool names and brief descriptions
        tool_names = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")[:50]  # Truncate descriptions
            tool_names.append(f"{name}: {desc}")
        
        # Limit to top 5 most essential tools
        essential_tools = tool_names[:5]
        return "; ".join(essential_tools)
    
    def _format_single_tool(self, tool: Dict) -> str:
        """Format a single tool for display"""
        if not isinstance(tool, dict):
            return str(tool)
        
        name = self._extract_tool_name(tool)
        description = tool.get('description', 'No description')
        parameters = tool.get('parameters', '{}')
        
        # Limit description length to avoid bloat
        short_desc = self._truncate_description(description)
        
        return f"- {name}: {short_desc}, Parameters: {parameters}"
    
    def _extract_tool_name(self, tool: Dict) -> str:
        """Extract tool name from tool dict"""
        name = tool.get('name', 'unknown')
        if name == 'unknown':
            tool_func = tool.get('function', {})
            name = tool_func.get('name', 'unknown')
        return name
    
    def _truncate_description(self, description: str) -> str:
        """Truncate description if too long"""
        return description[:100] + "..." if len(description) > 100 else description
