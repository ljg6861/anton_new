#!/usr/bin/env python3
"""
Demo script showing the new ToolsRouter with allowlists, timeouts, and retries.
Demonstrates safety and observability features.
"""

import asyncio
import json
from server.agent.tools_router import tools_router, ExecutionStatus


async def demo_tools_router():
    """Demonstrate ToolsRouter capabilities"""
    print("=== Demo: ToolsRouter with Safety Features ===\n")
    
    # Show initial allowlist
    print("1. Initial allowlist:")
    allowlist = sorted(tools_router.allowlist)
    print(f"   {len(allowlist)} allowed tools: {', '.join(allowlist[:10])}...")
    print()
    
    # Test successful tool call
    print("2. Testing allowed tool (list_directory):")
    result = await tools_router.call("list_directory", {"path": "."})
    print(f"   Status: {result.status.value}")
    print(f"   Execution time: {result.execution_time_ms:.1f}ms")
    print(f"   Attempts: {result.attempts}")
    if result.ok:
        print(f"   Result: {str(result.result)[:100]}...")
    else:
        print(f"   Error: {result.error_message}")
    print()
    
    # Test blocked tool call
    print("3. Testing blocked tool (fake_dangerous_tool):")
    result = await tools_router.call("fake_dangerous_tool", {"do_bad_stuff": True})
    print(f"   Status: {result.status.value}")
    print(f"   Error: {result.error_message}")
    print()
    
    # Test timeout (using a tool that might take long)
    print("4. Testing timeout with very short timeout:")
    result = await tools_router.call("search_web", {"query": "test"}, timeout_ms=1)  # 1ms timeout
    print(f"   Status: {result.status.value}")
    print(f"   Execution time: {result.execution_time_ms:.1f}ms")
    if not result.ok:
        print(f"   Error: {result.error_message}")
    print()
    
    # Test nonexistent tool (should trigger retries)
    print("5. Testing nonexistent tool (should retry and fail):")
    result = await tools_router.call("nonexistent_tool", {"param": "value"})
    print(f"   Status: {result.status.value}")
    print(f"   Attempts: {result.attempts}")
    print(f"   Error: {result.error_message}")
    print()
    
    # Show execution statistics
    print("6. Execution statistics:")
    stats = tools_router.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2%}")
        else:
            print(f"   {key}: {value}")
    print()
    
    # Test allowlist modification
    print("7. Testing allowlist modification:")
    original_size = len(tools_router.allowlist)
    tools_router.add_to_allowlist(["custom_tool_1", "custom_tool_2"])
    new_size = len(tools_router.allowlist)
    print(f"   Added 2 tools, allowlist size: {original_size} -> {new_size}")
    
    # Test if new tool is allowed
    print(f"   custom_tool_1 allowed: {tools_router.is_allowed('custom_tool_1')}")
    
    # Remove the test tools
    tools_router.remove_from_allowlist(["custom_tool_1", "custom_tool_2"])
    final_size = len(tools_router.allowlist)
    print(f"   Removed test tools, final size: {final_size}")
    print()
    
    # Test a more realistic scenario
    print("8. Realistic scenario - reading a file:")
    result = await tools_router.call("read_file", {
        "file_path": "/home/lucas/anton_new/server/agent/tools_router.py",
        "start_line": 1,
        "end_line": 10
    })
    print(f"   Status: {result.status.value}")
    print(f"   Execution time: {result.execution_time_ms:.1f}ms")
    if result.ok:
        lines = str(result.result).split('\n')
        print(f"   Read {len(lines)} lines successfully")
        print(f"   First line: {lines[0] if lines else 'N/A'}")
    else:
        print(f"   Error: {result.error_message}")
    print()


def demo_comparison():
    """Show the difference between old and new tool execution"""
    print("=== Comparison: Old vs New Tool Execution ===\n")
    
    print("OLD APPROACH (direct tool_manager):")
    print("- Direct calls to tool_manager.run_tool()")
    print("- No allowlist checking")
    print("- No timeout protection")
    print("- No automatic retries")
    print("- Limited error handling")
    print("- No execution metrics")
    print("- Difficult to add safety features")
    print()
    
    print("NEW APPROACH (ToolsRouter):")
    print("✓ Centralized tool execution through tools_router.call()")
    print("✓ Configurable allowlist for security")
    print("✓ Timeout protection with configurable limits")
    print("✓ Automatic retry with exponential backoff")
    print("✓ Comprehensive error handling and status reporting")
    print("✓ Execution metrics and observability")
    print("✓ Easy to extend with new safety features")
    print()
    
    print("BENEFITS:")
    print("🔒 Security: Allowlist prevents execution of dangerous tools")
    print("⏱️  Reliability: Timeouts prevent hanging operations")
    print("🔄 Resilience: Retries handle transient failures")
    print("📊 Observability: Detailed execution metrics and timing")
    print("🛡️  Safety: Centralized control over all tool execution")
    print("🎯 Consistency: Same interface for all tools")


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
