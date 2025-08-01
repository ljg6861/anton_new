#!/usr/bin/env python3
"""
Demonstration script showing the Anton agent improvements for systematic task handling.
This script demonstrates the key improvements without requiring external dependencies.
"""

import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demonstrate_exploration_patterns():
    """Demonstrate how the system benefits tasks with exploration patterns"""
    print("=== Exploration Pattern Recognition ===")
    
    exploration_keywords = ['review', 'code', 'source', 'function', 'class', 'file', 'implementation', 'analyze',
                           'examine', 'investigate', 'explore', 'find', 'search', 'read', 'check']
    
    test_tasks = [
        "Please review the code in the authentication module",
        "Help me understand how the payment system works", 
        "What's the weather like today?",
        "Analyze the implementation of the user management system",
        "Can you check the source code for any security issues?",
        "Investigate the configuration files in the project",
        "Explore the database schema",
        "Find all references to the user model"
    ]
    
    for task in test_tasks:
        has_exploration_pattern = any(keyword in task.lower() for keyword in exploration_keywords)
        status = "✓ EXPLORATION" if has_exploration_pattern else "○ SIMPLE"
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
    """Demonstrate the enhanced prompts"""
    print("=== Enhanced Prompt System ===")
    
    try:
        from server.agent.prompts import get_evaluator_prompt
        
        print("Enhanced Evaluator Prompt (excerpt):")
        eval_prompt = get_evaluator_prompt()
        # Find the exploration section
        if "Exploration and Investigation Tasks" in eval_prompt:
            start = eval_prompt.find("**Special Handling for Exploration and Investigation Tasks:**")
            end = eval_prompt.find("**Based on your analysis", start)
            if start != -1 and end != -1:
                exploration_section = eval_prompt[start:end].strip()
                print(f"  {exploration_section}")
        print()
        
    except ImportError as e:
        print(f"Cannot demonstrate prompts due to import error: {e}")

def main():
    """Run all demonstrations"""
    print("Anton Agent General Task Handling Improvements Demonstration")
    print("=" * 60)
    print()
    
    demonstrate_exploration_patterns()
    demonstrate_context_store()
    demonstrate_thought_loop_detection()
    demonstrate_prompts()
    
    print("=== Summary of Improvements ===")
    print("✓ Enhanced context tracking between agent steps")
    print("✓ Thought loop prevention to avoid circular reasoning") 
    print("✓ Improved evaluator for multi-step exploration progress")
    print("✓ Generalized systematic approach for all task types")
    print("✓ Standardized FINAL ANSWER formatting")
    print()
    print("These improvements help the agent handle any exploration or")
    print("investigation task more effectively, preventing loops and")
    print("maintaining context across multiple steps.")

if __name__ == "__main__":
    main()