#!/usr/bin/env python3
"""
Demonstration script showing the Anton agent improvements for code review tasks.
This script demonstrates the key improvements without requiring external dependencies.
"""

import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demonstrate_code_review_detection():
    """Demonstrate how the system detects code review tasks"""
    print("=== Code Review Task Detection ===")
    
    code_review_keywords = ['review', 'code', 'source', 'function', 'class', 'file', 'implementation', 'analyze']
    
    test_tasks = [
        "Please review the code in the authentication module",
        "Help me understand how the payment system works",
        "What's the weather like today?",
        "Analyze the implementation of the user management system",
        "Can you check the source code for any security issues?"
    ]
    
    for task in test_tasks:
        is_code_review = any(keyword in task.lower() for keyword in code_review_keywords)
        status = "✓ CODE REVIEW" if is_code_review else "✗ GENERAL"
        print(f"{status}: {task}")
    print()

def demonstrate_context_store():
    """Demonstrate how the context store works"""
    print("=== Context Store Functionality ===")
    
    try:
        from server.agent.tool_executor import _update_context_store
        
        context_store = {
            "explored_files": set(),
            "code_content": {},
            "task_progress": []
        }
        
        print("Initial context store:", context_store)
        
        # Simulate reading a file
        _update_context_store(
            context_store,
            "read_file",
            {"file_path": "/app/server/agent/prompts.py"},
            "def get_planner_prompt():\n    return 'System prompt text...'"
        )
        
        print("After reading a file:")
        print(f"  Explored files: {list(context_store['explored_files'])}")
        print(f"  Code content keys: {list(context_store['code_content'].keys())}")
        print(f"  Progress: {context_store['task_progress']}")
        
        # Simulate listing a directory
        _update_context_store(
            context_store,
            "list_directory", 
            {"path": "/app/server/agent"},
            "prompts.py\norganizer.py\ndoer.py\ntool_executor.py"
        )
        
        print("After listing directory:")
        print(f"  Explored files: {list(context_store['explored_files'])}")
        print(f"  Progress: {context_store['task_progress']}")
        
    except ImportError as e:
        print(f"Cannot demonstrate context store due to import error: {e}")
    print()

def demonstrate_thought_loop_detection():
    """Demonstrate the thought loop detection logic"""
    print("=== Thought Loop Detection ===")
    
    responses = [
        "I need to read the configuration file but cannot find it",
        "I need to read the config file but it cannot be found", 
        "Let me try a different approach to locate the file",
        "I should check the directory structure first"
    ]
    
    print("Analyzing response similarity for loop detection:")
    
    for i, response in enumerate(responses):
        words = set(response.lower().split())
        
        if i > 0:
            prev_words = set(responses[i-1].lower().split())
            overlap = len(words.intersection(prev_words)) / max(len(words), 1)
            
            if overlap > 0.8:
                status = "🔄 POTENTIAL LOOP DETECTED"
            elif overlap > 0.5:
                status = "⚠️  HIGH SIMILARITY"
            else:
                status = "✓ DIFFERENT APPROACH"
                
            print(f"  Response {i+1}: {status} (overlap: {overlap:.2f})")
            print(f"    \"{response}\"")
        else:
            print(f"  Response {i+1}: ✓ BASELINE")
            print(f"    \"{response}\"")
    print()

def demonstrate_prompts():
    """Demonstrate the specialized prompts"""
    print("=== Specialized Prompts ===")
    
    try:
        from server.agent.prompts import get_code_review_planner_prompt, get_evaluator_prompt
        
        print("Code Review Planner Prompt (excerpt):")
        prompt = get_code_review_planner_prompt()
        # Show first 200 characters
        excerpt = prompt[:200] + "..." if len(prompt) > 200 else prompt
        print(f"  {excerpt}")
        print()
        
        print("Enhanced Evaluator Prompt (excerpt):")
        eval_prompt = get_evaluator_prompt()
        # Find the code review section
        if "Code Review Tasks" in eval_prompt:
            start = eval_prompt.find("**Special Handling for Code Review Tasks:**")
            end = eval_prompt.find("**Based on your analysis", start)
            if start != -1 and end != -1:
                code_review_section = eval_prompt[start:end].strip()
                print(f"  {code_review_section}")
        print()
        
    except ImportError as e:
        print(f"Cannot demonstrate prompts due to import error: {e}")

def main():
    """Run all demonstrations"""
    print("Anton Agent Code Review Improvements Demonstration")
    print("=" * 55)
    print()
    
    demonstrate_code_review_detection()
    demonstrate_context_store()
    demonstrate_thought_loop_detection()
    demonstrate_prompts()
    
    print("=== Summary of Improvements ===")
    print("✓ Enhanced context tracking between agent steps")
    print("✓ Automatic code review task detection")
    print("✓ Thought loop prevention to avoid circular reasoning")
    print("✓ Specialized prompts for systematic code exploration")
    print("✓ More lenient evaluator for multi-step progress")
    print("✓ Standardized FINAL ANSWER formatting")
    print()
    print("These improvements address the core issues that were causing")
    print("the agent to get stuck in loops during code review tasks.")

if __name__ == "__main__":
    main()