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
        """Format tools in a compact way to reduce prompt bloat"""
        if not tools:
            return "No tools available"
        
        tool_summaries = []
        for tool in tools:
            summary = self._format_single_tool(tool)
            tool_summaries.append(summary)
        
        tools_text = "; ".join(tool_summaries)
        budget = self.memory.budget.system_tools_budget // 3
        
        return self.memory.truncate_to_budget(tools_text, budget)
    
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
