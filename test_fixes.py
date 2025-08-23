#!/usr/bin/env python3
"""
Test script to verify the fixes for:
1. Research capability for unknown topics like "zyns"
2. Streaming tokens in chat route
"""

import asyncio
import json
import sys
import httpx
from typing import List, Dict

async def test_chat_with_unknown_topic():
    """Test that the agent researches unknown topics before responding"""
    
    print("🧪 Testing chat route with unknown topic research...")
    
    # Test message about zyns (which the agent should research)
    test_messages = [
        {
            "role": "user", 
            "content": "I'm having withdrawal symptoms from stopping zyns. Can you help me understand what zyns are and advice for withdrawal?"
        }
    ]
    
    request_data = {
        "messages": test_messages,
        "temperature": 0.7
    }
    
    print(f"Sending request: {test_messages[0]['content']}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            print("📡 Connecting to agent server...")
            
            # Stream the response
            response_buffer = ""
            tool_usage_detected = False
            tokens_streamed = False
            
            async with client.stream(
                "POST", 
                "http://localhost:8001/v1/agent/chat", 
                json=request_data
            ) as response:
                
                if response.status_code != 200:
                    print(f"❌ Error: HTTP {response.status_code}")
                    return False
                
                print("📥 Receiving streamed response...")
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        response_buffer += chunk
                        
                        # Check for tool usage (research capability)
                        if "web_search" in chunk or "search_codebase" in chunk:
                            tool_usage_detected = True
                            print("✅ Tool usage detected - agent is researching!")
                        
                        # Check for token streaming
                        if chunk.startswith("<token>") and chunk.endswith("</token>"):
                            tokens_streamed = True
                        
                        # Print chunks for debugging
                        if len(chunk) < 100:
                            print(f"   Chunk: {chunk!r}")
            
            print(f"\n📊 Test Results:")
            print(f"   Response length: {len(response_buffer)} chars")
            print(f"   Tool usage detected: {tool_usage_detected}")
            print(f"   Tokens streamed: {tokens_streamed}")
            
            # Check if response mentions actual research about zyns
            response_lower = response_buffer.lower()
            if "zyn" in response_lower and "nicotine" in response_lower:
                print("✅ Response correctly identifies zyns as nicotine products!")
                research_success = True
            else:
                print("❌ Response doesn't properly identify what zyns are")
                research_success = False
            
            print(f"\n📝 Full response preview (first 500 chars):")
            print(response_buffer[:500])
            print("...")
            
            return tool_usage_detected and tokens_streamed and research_success
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

async def main():
    """Run the test suite"""
    print("🚀 Starting fix verification tests...\n")
    
    # Test 1: Chat with research capability
    test1_passed = await test_chat_with_unknown_topic()
    
    print(f"\n🏁 Test Results:")
    print(f"   Chat research & streaming: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    
    if test1_passed:
        print("\n🎉 All tests passed! Fixes are working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
