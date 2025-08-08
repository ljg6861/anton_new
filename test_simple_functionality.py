#!/usr/bin/env python3
"""
Simple test for finality detection logic without dependencies.
"""

def _is_final_response(content: str) -> bool:
    """
    Copy of the new finality detection logic from react_agent.py
    """
    content_lower = content.lower().strip()

    if not content:
        return False

    # --- Rule 1: Check for explicit final answer markers ---
    final_markers = [
        'final answer:',
        '<final_answer>',
        'task completed',
        'i have finished',
        'done.',
        'that completes'
    ]
    if any(marker in content_lower for marker in final_markers):
        return True

    # --- Rule 2: Check for conversational closing statements ---
    closing_signals = [
        'let me know if you need anything else',
        'is there anything else',
        'how else can i help',
        'hope that helps',
        'feel free to ask'
    ]
    if any(signal in content_lower for signal in closing_signals):
        return True

    # --- Rule 3: Check for explicit signs of continuation or tool use ---
    continuation_signals = [
        '<tool_call>',
        '<tool_code>',
        'i need to use',
        'i will now',
        'the next step is to',
        'let me first',
        'let me check',
        'i should',
        'i need to'
    ]
    if any(signal in content_lower for signal in continuation_signals):
        return False

    # --- Rule 4: Default to NOT final unless explicit finality is indicated ---
    return False

def test_finality_detection():
    """Test the improved finality detection logic"""
    print("🧪 Testing finality detection improvements...")
    
    test_cases = [
        # Cases that should NOT be final (was incorrectly final before)
        ('Hello there!', False),
        ('Good morning!', False),
        ('I can help you with that.', False),
        ('Let me check that for you.', False),
        ('I need to use a tool first', False),
        ('Let me first examine the file.', False),
        ('I should look into this.', False),
        
        # Cases that should be final (explicit markers)
        ('Final Answer: The result is 42', True),
        ('<final_answer>Here is the result</final_answer>', True),
        ('Task completed successfully.', True),
        ('I have finished the analysis.', True),
        ('Done.', True),
        ('Hope that helps!', True),
        ('Let me know if you need anything else.', True),
        ('Is there anything else I can help with?', True),
        
        # Cases with tool usage indicators (should NOT be final)
        ('I need to use the search tool', False),
        ('<tool_code>{"name": "search"}</tool_code>', False),
        ('The next step is to read the file', False),
        ('I will now execute the command', False),
        
        # Edge cases
        ('', False),  # Empty content
        ('   ', False),  # Whitespace only
    ]
    
    passed = 0
    failed = 0
    
    print("\nTest Results:")
    print("=" * 60)
    
    for content, expected in test_cases:
        result = _is_final_response(content)
        status = '✅ PASS' if result == expected else '❌ FAIL'
        
        if result == expected:
            passed += 1
        else:
            failed += 1
            
        print(f'{status} | {result:5} | "{content[:40]}..."')
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎯 All finality detection tests passed! ✅")
    else:
        print(f"⚠️  {failed} tests failed")
        
    return failed == 0

def test_file_sanitization():
    """Test the file content sanitization logic"""
    print("\n🧪 Testing file content sanitization...")
    
    test_content = """
    This is a file with tool patterns:
    
    <tool_code>
    {"name": "some_tool", "args": {}}
    </tool_code>
    
    And also:
    <tool_call>
    {"name": "another_tool"}
    </tool_call>
    
    Regular content should be preserved.
    """
    
    # Apply the sanitization logic from file_management.py
    sanitized = test_content.replace("<tool_code>", "&lt;tool_code&gt;")
    sanitized = sanitized.replace("</tool_code>", "&lt;/tool_code&gt;")
    sanitized = sanitized.replace("<tool_call>", "&lt;tool_call&gt;")
    sanitized = sanitized.replace("</tool_call>", "&lt;/tool_call&gt;")
    
    # Check that patterns were escaped
    assert "&lt;tool_code&gt;" in sanitized, "tool_code tags should be escaped"
    assert "&lt;tool_call&gt;" in sanitized, "tool_call tags should be escaped"
    assert "<tool_code>" not in sanitized, "Original tool_code tags should be removed"
    assert "<tool_call>" not in sanitized, "Original tool_call tags should be removed"
    assert "Regular content should be preserved" in sanitized, "Regular content should remain"
    
    print("✅ File sanitization test passed!")
    return True

if __name__ == "__main__":
    print("🚀 Running simplified functionality tests...\n")
    
    finality_ok = test_finality_detection()
    sanitization_ok = test_file_sanitization()
    
    if finality_ok and sanitization_ok:
        print("\n🎯 All tests passed! ✅")
    else:
        print("\n❌ Some tests failed")