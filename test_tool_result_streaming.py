#!/usr/bin/env python3
"""
Test tool result streaming improvements - ensuring actual results are shown instead of JSON requests.
"""

import asyncio
import json
import logging
from unittest.mock import Mock, AsyncMock

# Setup test environment
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from server.agent.tool_executor import process_tool_calls
from server.agent.knowledge_store import KnowledgeStore
import re

# Mock tool manager for testing
class MockToolManager:
    def run_tool(self, tool_name, tool_args):
        if tool_name == "test_tool":
            return f"Tool executed successfully with args: {tool_args}"
        elif tool_name == "failing_tool":
            raise Exception("Tool failed")
        return "Unknown tool"

# Mock the tool manager module
sys.modules['server.agent.tools.tool_manager'] = Mock()
sys.modules['server.agent.tools.tool_manager'].tool_manager = MockToolManager()

async def test_tool_result_streaming():
    """Test that actual tool results are streamed to UI, not just JSON requests"""
    print("🧪 Testing tool result streaming improvements...")
    
    # Setup
    logger = Mock()
    knowledge_store = Mock()
    knowledge_store.update_from_tool_execution = Mock()
    
    messages = []
    tool_results_captured = []
    
    async def capture_tool_result(tool_result_summary):
        tool_results_captured.append(tool_result_summary)
    
    # Test content with tool call
    response_buffer = """
    I need to use a tool.
    
    <tool_code>
    {"name": "test_tool", "arguments": {"param1": "value1"}}
    </tool_code>
    
    Let me process that.
    """
    
    # Tool call regex (simplified version of what's used in config)
    tool_call_regex = re.compile(r'<tool_code>\s*(\{.*?\})\s*</tool_code>', re.DOTALL)
    
    # Execute
    result = await process_tool_calls(
        response_buffer,
        tool_call_regex,
        messages,
        logger,
        knowledge_store,
        capture_tool_result
    )
    
    # Verify tool was called
    assert result == True, "Should return True when tool calls are made"
    
    # Verify messages were updated with proper role
    assert len(messages) == 1, f"Expected 1 message, got {len(messages)}"
    assert messages[0]["role"] == "system", f"Expected system role, got {messages[0]['role']}"
    assert "OBSERVATION:" in messages[0]["content"], "Message should use OBSERVATION format"
    
    # Verify tool result was captured for UI
    assert len(tool_results_captured) == 1, f"Expected 1 tool result, got {len(tool_results_captured)}"
    
    tool_result = tool_results_captured[0]
    assert tool_result["name"] == "test_tool", f"Expected test_tool, got {tool_result['name']}"
    assert tool_result["status"] == "success", f"Expected success, got {tool_result['status']}"
    assert "Tool executed successfully" in tool_result["brief_result"], "Result should contain actual tool output"
    assert tool_result["arguments"] == {"param1": "value1"}, "Arguments should be preserved"
    
    print("✅ Tool result streaming test passed!")

async def test_single_tool_execution():
    """Test that only one tool is executed per turn to avoid dependencies"""
    print("🧪 Testing single tool per turn execution...")
    
    # Setup
    logger = Mock()
    knowledge_store = Mock()
    knowledge_store.update_from_tool_execution = Mock()
    
    messages = []
    tool_results_captured = []
    
    async def capture_tool_result(tool_result_summary):
        tool_results_captured.append(tool_result_summary)
    
    # Test content with multiple tool calls
    response_buffer = """
    I need to use multiple tools.
    
    <tool_code>
    {"name": "test_tool", "arguments": {"param1": "value1"}}
    </tool_code>
    
    <tool_code>
    {"name": "test_tool", "arguments": {"param2": "value2"}}
    </tool_code>
    
    Let me process those.
    """
    
    # Tool call regex
    tool_call_regex = re.compile(r'<tool_code>\s*(\{.*?\})\s*</tool_code>', re.DOTALL)
    
    # Execute
    result = await process_tool_calls(
        response_buffer,
        tool_call_regex,
        messages,
        logger,
        knowledge_store,
        capture_tool_result
    )
    
    # Verify only one tool was executed
    assert result == True, "Should return True when tool calls are made"
    assert len(tool_results_captured) == 1, f"Expected only 1 tool result (single tool per turn), got {len(tool_results_captured)}"
    
    # Verify the first tool was executed
    tool_result = tool_results_captured[0]
    assert tool_result["arguments"] == {"param1": "value1"}, "First tool should have been executed"
    
    print("✅ Single tool per turn test passed!")

async def test_error_handling():
    """Test proper error handling and streaming"""
    print("🧪 Testing error handling in tool execution...")
    
    # Setup
    logger = Mock()
    knowledge_store = Mock()
    knowledge_store.update_from_tool_execution = Mock()
    
    messages = []
    tool_results_captured = []
    
    async def capture_tool_result(tool_result_summary):
        tool_results_captured.append(tool_result_summary)
    
    # Test content with failing tool call
    response_buffer = """
    <tool_code>
    {"name": "failing_tool", "arguments": {}}
    </tool_code>
    """
    
    # Tool call regex
    tool_call_regex = re.compile(r'<tool_code>\s*(\{.*?\})\s*</tool_code>', re.DOTALL)
    
    # Execute
    result = await process_tool_calls(
        response_buffer,
        tool_call_regex,
        messages,
        logger,
        knowledge_store,
        capture_tool_result
    )
    
    # Verify error was handled properly
    assert result == True, "Should return True even when tool fails"
    assert len(tool_results_captured) == 1, f"Expected 1 error result, got {len(tool_results_captured)}"
    
    tool_result = tool_results_captured[0]
    assert tool_result["status"] == "error", f"Expected error status, got {tool_result['status']}"
    assert "Error:" in tool_result["brief_result"], "Error should be in result"
    
    print("✅ Error handling test passed!")

async def main():
    """Run all tests"""
    print("🚀 Running tool result streaming tests...\n")
    
    await test_tool_result_streaming()
    print()
    await test_single_tool_execution()
    print()
    await test_error_handling()
    
    print("\n🎯 All tool result streaming tests passed! ✅")

if __name__ == "__main__":
    asyncio.run(main())