#!/usr/bin/env python3
"""
Test script to verify the agent uses research results instead of saying "I don't have access"
"""

import asyncio
import httpx

async def test_weather_research_usage():
    """Test that the agent uses web search results for weather questions"""
    
    print("🌦️  Testing weather research capability...")
    
    # Test message about weather (should research and provide results)
    test_messages = [
        {
            "role": "user", 
            "content": "What's the weather forecast for Ormond Beach at 5 PM today?"
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
            
            response_buffer = ""
            research_detected = False
            says_no_access = False
            
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
                        
                        # Check for research activity
                        if "web_search" in chunk:
                            research_detected = True
                            print("✅ Research detected - agent is searching for weather info!")
                        
                        # Check for problematic responses
                        if "don't have access" in chunk.lower() or "i don't have" in chunk.lower():
                            says_no_access = True
                            print("❌ Agent says it doesn't have access despite using tools!")
            
            print(f"\n📊 Test Results:")
            print(f"   Research detected: {research_detected}")
            print(f"   Says 'no access': {says_no_access}")
            
            # Check if response contains actual weather information
            response_lower = response_buffer.lower()
            has_useful_info = any(word in response_lower for word in [
                "temperature", "rain", "sunny", "cloudy", "degrees", "forecast", 
                "weather", "precipitation", "humidity", "wind"
            ])
            
            print(f"   Contains weather info: {has_useful_info}")
            
            print(f"\n📝 Response preview (last 500 chars):")
            # Show the end of the response where the final answer should be
            final_part = response_buffer[-500:] if len(response_buffer) > 500 else response_buffer
            # Clean up the preview by removing token tags
            clean_preview = final_part.replace("<token>", "").replace("</token>", "")
            print(clean_preview)
            
            # Success if research happened, agent doesn't claim no access, and provides useful info
            success = research_detected and not says_no_access and has_useful_info
            return success
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

async def main():
    """Run the weather research test"""
    print("🚀 Testing agent's ability to use research results...\n")
    
    test_passed = await test_weather_research_usage()
    
    print(f"\n🏁 Test Results:")
    print(f"   Weather research usage: {'✅ PASSED' if test_passed else '❌ FAILED'}")
    
    if test_passed:
        print("\n🎉 Agent correctly uses research results!")
        return 0
    else:
        print("\n💥 Agent still claims no access despite doing research.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
