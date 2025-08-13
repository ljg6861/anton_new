#!/usr/bin/env python3
"""
Demo script showing the new ToolsRouter with allowlists, timeouts, and retries.
Demonstrates safety and observabili    print("DYNAMIC DISCOVERY BENEFITS:")
    print("🔄 Auto-sync: Tools are automatically discovered from tools/ directory")
    print("🔒 Security: Only actual available tools are allowed")
    print("⚡ Flexibility: New tools are automatically available")
    print("🧹 Clean: No stale tool references in allowlist")
    print("🔧 Maintainable: No manual allowlist management required")
    print("📊 Observable: Track available vs allowed tools")
    print("🎯 Accurate: Allowlist always matches reality")res.
"""

import asyncio
import json
from server.agent.tools_router import tools_router, ExecutionStatus


async def demo_tools_router():
    """Demonstrate ToolsRouter capabilities with dynamic tool discovery"""
    print("=== Demo: ToolsRouter with Dynamic Tool Discovery ===\n")
    
    # Show dynamically discovered allowlist
    print("1. Dynamically discovered allowlist:")
    allowlist = sorted(tools_router.allowlist)
    print(f"   {len(allowlist)} tools discovered from tools/ directory")
    print(f"   Sample tools: {', '.join(allowlist[:10])}...")
    if len(allowlist) > 10:
        print(f"   And {len(allowlist) - 10} more...")
    print()
    
    # Show available vs allowed tools
    print("2. Available vs Allowed tools:")
    available_tools = tools_router.get_available_tools()
    allowed_tools = tools_router.get_allowlist()
    print(f"   Available tools: {len(available_tools)}")
    print(f"   Allowed tools: {len(allowed_tools)}")
    print(f"   Match: {available_tools == allowed_tools}")
    print()
    
    # Test successful tool call
    print("3. Testing dynamically allowed tool:")
    if "list_directory" in allowlist:
        result = await tools_router.call("list_directory", {"path": "."})
        print(f"   Tool: list_directory")
        print(f"   Status: {result.status.value}")
        print(f"   Execution time: {result.execution_time_ms:.1f}ms")
        print(f"   Attempts: {result.attempts}")
        if result.ok:
            print(f"   Result: {str(result.result)[:100]}...")
        else:
            print(f"   Error: {result.error_message}")
    else:
        print("   list_directory not available, trying first available tool...")
        if allowlist:
            first_tool = allowlist[0]
            result = await tools_router.call(first_tool, {})
            print(f"   Tool: {first_tool}")
            print(f"   Status: {result.status.value}")
            if not result.ok:
                print(f"   Error: {result.error_message}")
    print()
    
    # Test blocked tool call (should not exist in discovered tools)
    print("4. Testing blocked tool (not in discovered tools):")
    result = await tools_router.call("fake_dangerous_tool", {"do_bad_stuff": True})
    print(f"   Status: {result.status.value}")
    print(f"   Error: {result.error_message}")
    print()
    
    # Test allowlist refresh
    print("5. Testing allowlist refresh:")
    old_count = len(tools_router.allowlist)
    tools_router.refresh_allowlist()
    new_count = len(tools_router.allowlist)
    print(f"   Tools before refresh: {old_count}")
    print(f"   Tools after refresh: {new_count}")
    print(f"   Allowlist updated: {old_count != new_count}")
    print()
    
    # Test sync with available tools
    print("6. Testing sync with available tools:")
    sync_result = tools_router.sync_with_available_tools()
    print(f"   Added tools: {sync_result['added']}")
    print(f"   Removed tools: {sync_result['removed']}")
    print(f"   Sync needed: {bool(sync_result['added'] or sync_result['removed'])}")
    print()
    
    # Show execution statistics
    print("7. Execution statistics:")
    stats = tools_router.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2%}")
        else:
            print(f"   {key}: {value}")
    print()
    
    # Test allowlist modification with dynamic tools
    print("8. Testing allowlist modification:")
    original_size = len(tools_router.allowlist)
    
    # Try removing and re-adding a real tool
    if "read_file" in tools_router.allowlist:
        tools_router.remove_from_allowlist(["read_file"])
        removed_size = len(tools_router.allowlist)
        print(f"   Removed read_file: {original_size} -> {removed_size}")
        
        # Test that it's now blocked
        result = await tools_router.call("read_file", {"file_path": "test.txt"})
        print(f"   read_file now blocked: {result.status.value}")
        
        # Add it back
        tools_router.add_to_allowlist(["read_file"])
        final_size = len(tools_router.allowlist)
        print(f"   Re-added read_file: {removed_size} -> {final_size}")
    else:
        print("   read_file not available for modification test")
    print()


def demo_comparison():
    """Show the difference between old and new tool execution"""
    print("\n=== Comparison: Old vs New Tool Execution ===\n")
    
    print("OLD APPROACH (direct tool_manager):")
    print("- Direct calls to tool_manager.run_tool()")
    print("- No allowlist checking")
    print("- No timeout protection")
    print("- No automatic retries")
    print("- Limited error handling")
    print("- No execution metrics")
    print("- Manual tool management")
    print("- Static tool configuration")
    print()
    
    print("NEW APPROACH (ToolsRouter with Dynamic Discovery):")
    print("✓ Centralized tool execution through tools_router.call()")
    print("✓ Dynamic allowlist from tools/ directory discovery")
    print("✓ Automatic tool registration and availability checking")
    print("✓ Configurable allowlist for security")
    print("✓ Timeout protection with configurable limits")
    print("✓ Automatic retry with exponential backoff")
    print("✓ Comprehensive error handling and status reporting")
    print("✓ Execution metrics and observability")
    print("✓ Runtime tool refresh and synchronization")
    print("✓ Easy to extend with new safety features")
    print()
    
    print("DYNAMIC DISCOVERY BENEFITS:")
    print("� Auto-sync: Tools are automatically discovered from tools/ directory")
    print("🔒 Security: Only actual available tools are allowed")
    print("⚡ Flexibility: New tools are automatically available")
    print("🧹 Clean: No stale tool references in allowlist")
    print("� Maintainable: No manual allowlist management required")
    print("� Observable: Track available vs allowed tools")
    print("🎯 Accurate: Allowlist always matches reality")


async def demo_integration():
    """Show how ToolsRouter integrates with the existing system"""
    print("\n=== Integration with Existing System ===\n")
    
    print("INTEGRATION POINTS:")
    print("1. tool_executor.py: Updated to use tools_router.call()")
    print("2. state_ops.py: Enhanced to track execution metadata")
    print("3. react_agent.py: Enhanced tool result callbacks")
    print("4. Backward compatibility: execute_tool_async() still works")
    print()
    
    print("EXECUTION FLOW:")
    print("ReAct Agent → tool_executor.process_tool_calls()")
    print("            → tools_router.call(name, args, timeout)")
    print("            → _low_level_dispatch() → tool_manager.run_tool()")
    print("            → ExecutionResult with metadata")
    print("            → Enhanced state tracking")
    print()
    
    print("NEW FEATURES AVAILABLE:")
    print("- Per-tool timeout configuration")
    print("- Execution attempt tracking")
    print("- Failure reason classification")
    print("- Performance monitoring")
    print("- Security policy enforcement")


if __name__ == "__main__":
    async def main():
        await demo_tools_router()
        demo_comparison()
        await demo_integration()
    
    asyncio.run(main())
