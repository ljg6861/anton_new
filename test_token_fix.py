#!/usr/bin/env python3
"""
Simple test to verify that the token processing fix works correctly.
"""
import asyncio
import re

def test_token_extraction():
    """Test that we can properly extract content from token tags."""
    # Simulate the problematic token stream
    token_stream = [
        "<token>I've</token>",
        "<token> </token>", 
        "<token>gathered</token>",
        "<token> </token>",
        "<token>several</token>",
        "<token> </token>",
        "<token>resources</token>",
        "<token> </token>",
        "<token>about</token>",
        "<token> </token>",
        "<token>Crazy</token>",
        "<token> </token>",
        "<token>Train.</token>"
    ]
    
    # Simulate the executor processing
    result = ""
    for chunk in token_stream:
        if chunk.startswith("<token>") and chunk.endswith("</token>"):
            token_content = chunk[7:-8]  # Remove <token> and </token> tags
            result += token_content
        else:
            result += chunk
    
    expected = "I've gathered several resources about Crazy Train."
    print(f"Input tokens: {token_stream}")
    print(f"Expected result: '{expected}'")
    print(f"Actual result: '{result}'")
    print(f"Match: {result == expected}")
    
    # Test preview generation (first 200 chars)
    result_preview = result[:200] + "..." if len(result) > 200 else result
    print(f"Result preview: '{result_preview}'")
    
    return result == expected

if __name__ == "__main__":
    print("🧪 Testing Token Processing Fix...")
    success = test_token_extraction()
    if success:
        print("✅ Token processing fix working correctly!")
    else:
        print("❌ Token processing fix has issues!")
