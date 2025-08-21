#!/usr/bin/env python3
"""
Test script for the multiply tool.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from server.agent.tools.multiply_tool import MultiplyTool, MultiplyInput

def test_multiply_tool():
    """Test that the multiply tool works correctly."""
    # Create tool instance
    tool = MultiplyTool()
    
    # Test case 1: Multiply two positive numbers
    input1 = MultiplyInput(a=5.0, b=3.0)
    result1 = tool(input1)
    print(f"5.0 * 3.0 = {result1}")
    assert result1 == 15.0
    
    # Test case 2: Multiply with decimals
    input2 = MultiplyInput(a=2.5, b=4.0)
    result2 = tool(input2)
    print(f"2.5 * 4.0 = {result2}")
    assert result2 == 10.0
    
    # Test case 3: Multiply with negative numbers
    input3 = MultiplyInput(a=-2.0, b=3.0)
    result3 = tool(input3)
    print(f"-2.0 * 3.0 = {result3}")
    assert result3 == -6.0
    
    print("All tests passed!")

if __name__ == "__main__":
    test_multiply_tool()