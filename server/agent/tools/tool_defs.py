"""
Dynamic tool definitions that automatically discover and load tools.
Replaces the hardcoded STATIC_TOOLS array with intelligent discovery.
"""

from typing import List, Any
from server.agent.tools.legacy_wrapper import create_legacy_tool_wrappers
from server.agent.tools.tool_loader import ToolLoader


def get_all_tools() -> List[Any]:
    """
    Dynamically discover and return all available tools.
    
    Returns:
        List of all tool instances (both new and legacy)
    """
    all_tools = []
    
    # Get legacy tools wrapped for compatibility
    legacy_tools = create_legacy_tool_wrappers()
    all_tools.extend(legacy_tools.values())
    
    # Discover any new tools that inherit from BaseTool
    tool_loader = ToolLoader()
    discovered_tools = tool_loader.create_tool_instances()
    
    # Filter out legacy tools we've already wrapped
    legacy_tool_names = set(legacy_tools.keys())
    for name, tool in discovered_tools.items():
        if name not in legacy_tool_names:
            all_tools.append(tool)
    
    return all_tools


def get_tools_by_pattern(pattern: str) -> List[Any]:
    """
    Get tools matching a specific pattern or type.
    
    Args:
        pattern: Pattern to match against tool names or capabilities
        
    Returns:
        List of matching tool instances
    """
    all_tools = get_all_tools()
    matching_tools = []
    
    for tool in all_tools:
        tool_name = getattr(tool, 'metadata', {}).get('name', 'unknown')
        if pattern.lower() in tool_name.lower():
            matching_tools.append(tool)
    
    return matching_tools


def reload_tools() -> List[Any]:
    """
    Force reload of all tools from the filesystem.
    
    Returns:
        Refreshed list of all tool instances
    """
    # Clear any cached imports
    import importlib
    from server.agent.tools import tool_loader, legacy_wrapper
    
    try:
        importlib.reload(tool_loader)
        importlib.reload(legacy_wrapper)
    except:
        pass  # Ignore reload errors
    
    return get_all_tools()


# For backward compatibility, provide the STATIC_TOOLS array
# but now it's dynamically generated
STATIC_TOOLS = get_all_tools()


# Print discovery information
if __name__ == "__main__":
    tools = get_all_tools()
    print(f"🔍 Discovered {len(tools)} tools:")
    for tool in tools:
        if hasattr(tool, 'metadata'):
            print(f"  - {tool.metadata.name} v{tool.metadata.version}")
        else:
            print(f"  - {tool.__class__.__name__} (legacy)")
