#!/usr/bin/env python3

# Test script to verify Google Calendar tool functionality

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from server.agent.tools.google_calendar_tool import GoogleCalendarTool
    print("✅ Successfully imported GoogleCalendarTool")
    
    # Create tool instance
    tool_instance = GoogleCalendarTool()
    print("✅ Created tool instance successfully")
    
    # Check tool metadata
    metadata = tool_instance.get_metadata()
    print(f"✅ Tool name: {metadata.name}")
    print(f"✅ Tool version: {metadata.version}")
    print(f"✅ Tool description: {metadata.description}")
    
    # Test basic tool functionality
    print("✅ Google Calendar tool is available and ready for use!")
    
except ImportError as e:
    print(f"❌ Failed to import GoogleCalendarTool: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()