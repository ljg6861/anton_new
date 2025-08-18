#!/usr/bin/env python3
"""
Test the specific scenario from the log where duplicate learnings were shown in advisory.
"""

import asyncio
from server.agent.tool_learning_store import tool_learning_store

async def test_advisory_deduplication():
    """Test that the learning advisory doesn't show duplicates"""
    
    print("🧪 Testing Learning Advisory Deduplication")
    print("=" * 50)
    
    # Query for read_file learnings (which had duplicates in the log)
    learnings = tool_learning_store.query_relevant_learnings(
        "read_file",
        {"path": "some_file.py"},
        "user wants to read a file"
    )
    
    print(f"📚 Found {len(learnings)} learnings for read_file")
    
    # Check for duplicate patterns
    seen_patterns = set()
    duplicates_found = False
    
    for i, learning in enumerate(learnings):
        pattern_key = f"{learning.failure_pattern}|{learning.successful_alternative}"
        
        print(f"\n{i+1}. Pattern: {learning.failure_pattern}")
        print(f"   Alternative: {learning.successful_alternative}")
        print(f"   Confidence: {learning.confidence}")
        print(f"   Key: {pattern_key}")
        
        if pattern_key in seen_patterns:
            duplicates_found = True
            print(f"   ⚠️ DUPLICATE DETECTED!")
        else:
            seen_patterns.add(pattern_key)
    
    # Test what would be shown in advisory
    print(f"\n🎯 Advisory Test: {'❌ DUPLICATES FOUND' if duplicates_found else '✅ NO DUPLICATES'}")
    
    if not duplicates_found:
        print("   • Learning advisory will not show duplicate patterns")
        print("   • Each unique pattern appears only once")
    
    # Simulate the advisory message format
    advisory_lines = []
    for learning in learnings:
        advisory_line = f"⚠️ LEARNING: {learning.failure_pattern} → {learning.successful_alternative}"
        advisory_lines.append(advisory_line)
    
    print(f"\n📝 Simulated Advisory Message:")
    print("Tool Learning Advisory:")
    for line in advisory_lines:
        print(line)
    
    return not duplicates_found

async def main():
    """Main test function"""
    print("🚀 Testing Learning Advisory Deduplication Fix")
    print("This tests the specific issue from the agent log...")
    
    success = await test_advisory_deduplication()
    
    print(f"\n🎉 Test result: {'✅ FIXED' if success else '❌ STILL BROKEN'}")
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
