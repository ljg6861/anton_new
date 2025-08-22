#!/usr/bin/env python3
"""
Test the enhanced chat route with tool calling capabilities
"""

import asyncio
import sys
import os

# Add the server directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from server.agent.agentic_flow.full_agentic_flow import _handle_chat_route

async def test_chat_with_tools():
    """Test that chat route can handle tool calls"""
    print("🧪 Testing enhanced chat route with tool capabilities...")
    
    # Test 1: Simple chat without tools (should respond directly)
    print("\n--- Test 1: Simple greeting (no tools needed) ---")
    messages = [
        {"role": "user", "content": "Hello! How are you today?"}
    ]
    
    response = ""
    try:
        async for chunk in _handle_chat_route(messages):
            response += chunk
        print(f"✅ Simple chat response: {response[:100]}...")
    except Exception as e:
        print(f"❌ Simple chat failed: {e}")
    
    # Test 2: Request that should trigger tool use
    print("\n--- Test 2: Request that should use tools ---")
    messages = [
        {"role": "user", "content": "Can you search for any files related to 'episodic memory' in this codebase?"}
    ]
    
    response = ""
    try:
        async for chunk in _handle_chat_route(messages):
            response += chunk
        print(f"✅ Tool-enabled chat response: {response[:200]}...")
    except Exception as e:
        print(f"❌ Tool-enabled chat failed: {e}")
    
    # Test 3: File reading request
    print("\n--- Test 3: File reading request ---")
    messages = [
        {"role": "user", "content": "Can you read the README.md file and tell me what this project is about?"}
    ]
    
    response = ""
    try:
        async for chunk in _handle_chat_route(messages):
            response += chunk
        print(f"✅ File reading chat response: {response[:200]}...")
    except Exception as e:
        print(f"❌ File reading chat failed: {e}")
    
    print("\n🎉 Chat route testing complete!")

if __name__ == "__main__":
    asyncio.run(test_chat_with_tools())
