#!/usr/bin/env python3
"""
Test the full agentic flow routing with the enhanced chat route
"""

import asyncio
import sys
import os

# Add the server directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from server.agent.agentic_flow.full_agentic_flow import execute_agentic_flow

async def test_routing_with_tools():
    """Test that the routing correctly identifies chat vs task and handles tools"""
    print("🧪 Testing agentic flow routing with enhanced chat...")
    
    # Test 1: Simple chat message (should route to enhanced chat)
    print("\n--- Test 1: Simple chat routing ---")
    messages = [
        {"role": "user", "content": "What's the weather like?"}
    ]
    
    response = ""
    try:
        async for chunk in execute_agentic_flow(messages):
            if not (chunk.startswith("<step>") or chunk.startswith("<step_content>")):
                response += chunk
        print(f"✅ Chat route response: {response[:150]}...")
    except Exception as e:
        print(f"❌ Chat routing failed: {e}")
    
    # Test 2: Chat message that should use tools
    print("\n--- Test 2: Chat with tools routing ---")
    messages = [
        {"role": "user", "content": "Can you tell me about the files in this project?"}
    ]
    
    response = ""
    try:
        async for chunk in execute_agentic_flow(messages):
            if not (chunk.startswith("<step>") or chunk.startswith("<step_content>")):
                response += chunk
        print(f"✅ Chat with tools response: {response[:150]}...")
    except Exception as e:
        print(f"❌ Chat with tools routing failed: {e}")
    
    print("\n🎉 Routing test complete!")

if __name__ == "__main__":
    asyncio.run(test_routing_with_tools())
