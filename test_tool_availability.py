#!/usr/bin/env python3
"""
Test script to check if web_search tool is properly registered
"""

from server.agent.tools.tool_manager import tool_manager

def test_tool_availability():
    print("🔍 Checking tool availability...")
    
    # Get all tool names
    all_tools = tool_manager.get_tool_names()
    print(f"Available tools: {all_tools}")
    
    # Check if web_search is available
    has_web_search = tool_manager.has_tool("web_search")
    print(f"Has web_search: {has_web_search}")
    
    # Get web search tool schema if available
    if has_web_search:
        web_search_tools = tool_manager.get_tools_by_names(["web_search"])
        print(f"Web search tool schema: {web_search_tools}")
        
        # Test running the tool
        try:
            result = tool_manager.run_tool("web_search", '{"query": "test"}')
            print(f"Web search test result (first 200 chars): {str(result)[:200]}...")
        except Exception as e:
            print(f"Error running web search: {e}")
    
    # Check other tools we expect
    for tool_name in ["search_codebase", "fetch_web_content", "read_file"]:
        has_tool = tool_manager.has_tool(tool_name)
        print(f"Has {tool_name}: {has_tool}")

if __name__ == "__main__":
    test_tool_availability()
