#!/usr/bin/env python3

# Test script to verify tool registration works correctly

try:
    from server.agent.tools.google_calendar_tool import GoogleCalendarTool
    from server.agent.tools.manager import tool_manager
    print("Successfully imported GoogleCalendarTool and tool_manager")
    
    # Create tool instance
    tool_instance = GoogleCalendarTool()
    print("Created tool instance successfully")
    
    # Try to register the tool
    tool_manager.register(tool_instance)
    print("Tool registered successfully!")
    
except Exception as e:
    print(f"Error during registration: {e}")
    import traceback
    traceback.print_exc()