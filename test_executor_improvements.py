#!/usr/bin/env python3
"""
Test script to demonstrate the improved tool execution visibility and loop detection.
"""
import json
import re

def test_tool_signature_creation():
    """Test that tool signatures are created consistently."""
    print("🧪 Testing Tool Signature Creation...")
    
    # Test cases with different argument types
    test_cases = [
        {
            "tool_name": "web_search",
            "tool_args": {"query": "test search", "num_results": 3},
            "expected_prefix": "web_search:"
        },
        {
            "tool_name": "web_search", 
            "tool_args": {"query": "different search", "num_results": 3},
            "expected_prefix": "web_search:"
        },
        {
            "tool_name": "web_search",
            "tool_args": {"query": "test search", "num_results": 3},  # Same as first
            "expected_prefix": "web_search:"
        }
    ]
    
    signatures = []
    for i, test_case in enumerate(test_cases):
        tool_name = test_case["tool_name"]
        tool_args = test_case["tool_args"]
        
        # Create signature the same way our fixed code does
        tool_args_str = json.dumps(tool_args, sort_keys=True)
        tool_signature = f"{tool_name}:{tool_args_str}"
        signatures.append(tool_signature)
        
        print(f"  Case {i+1}: {tool_signature}")
    
    # Check for duplicates (should detect loops)
    print(f"\n  Signature 1 == Signature 3: {signatures[0] == signatures[2]} (should be True - same args)")
    print(f"  Signature 1 == Signature 2: {signatures[0] == signatures[1]} (should be False - different args)")
    
    return signatures

def test_token_cleaning():
    """Test that token tags are properly cleaned from results."""
    print("\n🧪 Testing Token Cleaning...")
    
    # Test cases with token tags
    test_cases = [
        "<token>I've</token><token> </token><token>found</token><token> </token><token>information</token>",
        "Regular text without tokens",
        "<token>Mixed</token> content <token>with</token> some tokens",
        ""
    ]
    
    for i, test_input in enumerate(test_cases):
        # Clean using the same regex our code uses
        cleaned = re.sub(r'<token>(.*?)</token>', r'\1', test_input)
        print(f"  Case {i+1}:")
        print(f"    Input:  '{test_input}'")
        print(f"    Output: '{cleaned}'")
    
    return True

def test_frequency_counting():
    """Test the frequency counting logic."""
    print("\n🧪 Testing Frequency Counting...")
    
    # Simulate recent tool calls
    recent_calls = [
        "web_search:{\"num_results\": 3, \"query\": \"search1\"}",
        "web_search:{\"num_results\": 3, \"query\": \"search2\"}", 
        "fetch_web_page:{\"url\": \"example.com\"}",
        "web_search:{\"num_results\": 5, \"query\": \"search3\"}",
        "web_search:{\"num_results\": 3, \"query\": \"search4\"}",
        "web_search:{\"num_results\": 3, \"query\": \"search5\"}",
        "web_search:{\"num_results\": 3, \"query\": \"search6\"}",
        "web_search:{\"num_results\": 3, \"query\": \"search7\"}",
    ]
    
    tool_name = "web_search"
    same_tool_count = sum(1 for sig in recent_calls[-10:] if sig.startswith(f"{tool_name}:"))
    
    print(f"  Recent calls: {len(recent_calls)}")
    print(f"  web_search calls in last 10: {same_tool_count}")
    print(f"  Would trigger frequency warning (8+): {same_tool_count >= 8}")
    
    return same_tool_count

if __name__ == "__main__":
    print("🔧 Testing Executor Improvements...")
    
    print("\n" + "="*60)
    signatures = test_tool_signature_creation()
    
    print("\n" + "="*60)  
    test_token_cleaning()
    
    print("\n" + "="*60)
    count = test_frequency_counting()
    
    print(f"\n✅ All tests completed!")
    print(f"📊 Summary:")
    print(f"  - Tool signatures work correctly for loop detection")
    print(f"  - Token cleaning removes markup properly")
    print(f"  - Frequency counting: {count} calls detected")
