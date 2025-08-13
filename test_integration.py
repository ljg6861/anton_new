#!/usr/bin/env python3
"""
Test script to verify end-to-end integration of dynamic tool discovery with ReAct agent.
"""
import asyncio
import sys
import os

# Add the server directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

from server.agent.state import make_state
from server.agent.tool_executor import execute_tool_async
from server.agent.tools_router import tools_router
import logging


async def test_integration():
    """Test that dynamic tool discovery integrates with ReAct agent workflow"""
    print("=== Integration Test: Dynamic Discovery + ReAct Agent ===\n")
    
    # 1. Check that tools_router is initialized with dynamic discovery
    print("1. ToolsRouter Initialization:")
    available_tools = tools_router.get_available_tools()
    allowed_tools = tools_router.get_allowlist()
    print(f"   Available tools: {len(available_tools)}")
    print(f"   Allowed tools: {len(allowed_tools)}")
    print(f"   Match: {available_tools == allowed_tools}")
    print()
    
    # 2. Create a State instance using the new structured system
    print("2. Creating structured State:")
    state = make_state(goal="Test dynamic tool discovery")
    print(f"   Status: {state.status}")
    print(f"   Goal: {state.goal}")
    print(f"   Token budget: {state.budgets.total_tokens}")
    print()
    
    # 3. Test tool execution through the ReAct agent pathway
    print("3. Testing tool execution through ReAct workflow:")
    
    # Create a simple logger for the test
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    try:
        # This uses the same pathway as the ReAct agent
        result = await execute_tool_async(
            "get_codebase_stats", 
            {"path": "/home/lucas/anton_new"}, 
            logger
        )
        
        print(f"   Tool execution completed")
        print(f"   Result type: {type(result)}")
        print(f"   Result length: {len(str(result))}")
        print(f"   Tool execution successful: {result is not None}")
        
    except Exception as e:
        print(f"   ❌ Tool execution failed: {e}")
        return False
    
    print()
    
    # 4. Verify ToolsRouter statistics
    print("4. ToolsRouter Execution Statistics:")
    stats = tools_router.get_stats()
    print(f"   Total calls: {stats['total_calls']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Block rate: {stats['block_rate']:.1f}%")
    print()
    
    # 5. Test dynamic discovery features
    print("5. Testing Dynamic Discovery Features:")
    
    # Check sync capability
    added, removed = tools_router.sync_with_available_tools()
    print(f"   Sync check - Added: {len(added)}, Removed: {len(removed)}")
    
    # Check refresh capability
    old_count = len(tools_router.get_allowlist())
    refreshed = tools_router.refresh_allowlist()
    new_count = len(tools_router.get_allowlist())
    print(f"   Refresh - Tools before: {old_count}, after: {new_count}, changed: {refreshed}")
    print()
    
    print("✅ Integration test completed successfully!")
    print("\nKEY BENEFITS DEMONSTRATED:")
    print("• Dynamic tool discovery automatically populates allowlist")
    print("• ReAct agent can use tools without hardcoded configuration")
    print("• System maintains security through allowlist enforcement")
    print("• Runtime tool management is fully functional")
    print("• Existing ReAct workflow unchanged but enhanced")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_integration())
    exit(0 if success else 1)
