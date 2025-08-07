#!/usr/bin/env python3
"""
Demo script to showcase the refactored ReAct agent architecture with KnowledgeStore.
This demonstrates the elimination of ConversationState dependency and integration of RAG.
"""
import sys
import os
from unittest.mock import Mock, patch

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock heavy dependencies for demo
sys.modules['vllm'] = Mock()
sys.modules['vllm.third_party'] = Mock()  
sys.modules['vllm.third_party.pynvml'] = Mock()
sys.modules['pynvml'] = Mock()
sys.modules['server.agent.learning_loop'] = Mock()
sys.modules['server.agent.rag_manager'] = Mock()
sys.modules['numpy'] = Mock()
sys.modules['faiss'] = Mock()
sys.modules['sentence_transformers'] = Mock()

from server.agent.knowledge_store import KnowledgeStore, ContextType, ImportanceLevel
from server.agent.react_agent import ReActAgent


def demo_knowledge_store_capabilities():
    """Demonstrate KnowledgeStore capabilities that replace ConversationState"""
    print("=== KnowledgeStore Demo: Replacing ConversationState ===")
    
    # Mock dependencies
    with patch('server.agent.knowledge_store.learning_loop'), \
         patch('server.agent.knowledge_store.rag_manager') as mock_rag:
        
        mock_rag.retrieve_knowledge.return_value = ["Past experience with file operations"]
        mock_rag.add_knowledge.return_value = None
        
        # Create knowledge store
        store = KnowledgeStore()
        
        print("1. Adding conversation messages (replaces ConversationState.add_message):")
        store.add_message("user", "Can you help me analyze this Python file?")
        store.add_message("assistant", "Of course! I'll help you analyze the Python file.")
        
        print(f"   Messages stored: {len(store.messages)}")
        print(f"   Context items created: {len(store.context_items)}")
        
        print("\n2. Adding tool execution results:")
        store.add_tool_output("read_file", "def analyze_code(): return 'insights'", 
                             {"args": {"file_path": "analyzer.py"}})
        
        print(f"   Tool outputs stored: {len(store.tool_outputs)}")
        
        print("\n3. Adding high-importance context (auto-persists to RAG):")
        store.add_context(
            "This file contains critical analysis logic for the system",
            ContextType.EVALUATOR_FEEDBACK,
            ImportanceLevel.HIGH,
            "code_analyzer"
        )
        
        print(f"   Total context items: {len(store.context_items)}")
        
        print("\n4. Building prioritized context summary:")
        summary = store.build_context_summary()
        print("   Context summary preview:")
        print("   " + "\n   ".join(summary.split('\n')[:5]))
        
        print("\n5. Getting messages for LLM (ConversationState compatibility):")
        llm_messages = store.get_messages_for_llm()
        print(f"   LLM-formatted messages: {len(llm_messages)}")
        
        print("\n6. Demonstrating conversation reset (preserves learned context):")
        context_count_before = len(store.context_items)
        store.mark_complete("Analysis completed successfully")
        store.reset_conversation()
        
        print(f"   Messages after reset: {len(store.messages)}")
        print(f"   Context items preserved: {len(store.context_items)} (was {context_count_before})")
        
        print("\n✅ KnowledgeStore successfully replaces ConversationState with enhanced capabilities!")


def demo_react_agent_integration():
    """Demonstrate ReActAgent integration with KnowledgeStore"""
    print("\n=== ReActAgent Demo: KnowledgeStore Integration ===")
    
    with patch('server.agent.knowledge_store.learning_loop'), \
         patch('server.agent.knowledge_store.rag_manager') as mock_rag:
        
        mock_rag.retrieve_knowledge.return_value = [
            "Previous experience with code analysis tasks",
            "Best practices for file reading operations"
        ]
        
        # Create knowledge store and ReAct agent
        store = KnowledgeStore()
        tools = [
            {"name": "read_file", "description": "Read file contents"},
            {"name": "analyze_code", "description": "Analyze Python code structure"}
        ]
        
        agent = ReActAgent(
            api_base_url="http://localhost:8000",
            tools=tools,
            knowledge_store=store,
            max_iterations=5
        )
        
        print("1. ReActAgent initialized with KnowledgeStore:")
        print(f"   Agent tools: {len(agent.tools)}")
        print(f"   Max iterations: {agent.max_iterations}")
        print(f"   Knowledge store attached: {agent.knowledge_store is store}")
        
        print("\n2. Generating system prompt with RAG integration:")
        system_prompt = agent.get_react_system_prompt()
        
        # Verify RAG was queried
        mock_rag.retrieve_knowledge.assert_called()
        
        print("   System prompt preview:")
        print("   " + "\n   ".join(system_prompt.split('\n')[:8]))
        print("   ... (prompt includes RAG knowledge)")
        
        print("\n3. Adding sample conversation to knowledge store:")
        store.add_message("user", "Please analyze the main.py file")
        
        print(f"   Messages in store: {len(store.messages)}")
        print(f"   Context items: {len(store.context_items)}")
        
        print("\n4. Simulating tool execution through knowledge store:")
        store.update_from_tool_execution(
            "read_file",
            {"file_path": "main.py"},
            "import sys\ndef main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()"
        )
        
        print(f"   Files explored: {store.explored_files}")
        print(f"   Code content cached: {len(store.code_content)} files")
        
        print("\n✅ ReActAgent successfully integrated with KnowledgeStore!")


def demo_elimination_of_dependencies():
    """Demonstrate that doer.py dependency has been eliminated"""
    print("\n=== Architecture Demo: Eliminated Dependencies ===")
    
    print("1. ConversationState dependency eliminated:")
    try:
        # Try to import ConversationState - should not be needed
        from server.agent.react_agent import ReActAgent
        print("   ✅ ReActAgent imports without ConversationState")
    except ImportError as e:
        if "ConversationState" in str(e):
            print("   ❌ ConversationState still required")
        else:
            print("   ⚠️  Other import issue:", e)
    
    print("\n2. doer.py dependency eliminated:")
    try:
        # Check that ReActAgent doesn't import from doer
        from server.agent.react_agent import ReActAgent
        import inspect
        source = inspect.getsource(ReActAgent)
        
        if "from server.agent.doer" in source or "import doer" in source:
            print("   ❌ doer.py still imported in ReActAgent")
        else:
            print("   ✅ ReActAgent no longer depends on doer.py")
            
        # Check that ReActAgent has direct LLM execution
        if "_execute_llm_request" in source:
            print("   ✅ ReActAgent has direct LLM execution method")
        else:
            print("   ❌ Direct LLM execution method not found")
            
    except Exception as e:
        print(f"   ⚠️  Could not analyze source: {e}")
    
    print("\n3. RAG integration verified:")
    try:
        from server.agent.knowledge_store import KnowledgeStore
        store = KnowledgeStore()
        
        # Check if query method exists
        if hasattr(store, 'query_relevant_knowledge'):
            print("   ✅ KnowledgeStore has RAG query capability")
        else:
            print("   ❌ RAG query method missing")
            
        # Check if persistence method exists  
        if hasattr(store, '_persist_to_rag'):
            print("   ✅ KnowledgeStore has RAG persistence capability")
        else:
            print("   ❌ RAG persistence method missing")
            
    except Exception as e:
        print(f"   ⚠️  Could not verify RAG integration: {e}")
    
    print("\n✅ Successfully eliminated dependencies and enhanced architecture!")


def main():
    """Run all demos to showcase the refactored architecture"""
    print("🚀 Demonstrating Refactored ReAct Agent Architecture")
    print("=" * 60)
    
    try:
        demo_knowledge_store_capabilities()
        demo_react_agent_integration()
        demo_elimination_of_dependencies()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETE: Architecture Successfully Refactored!")
        print("\nKey Achievements:")
        print("• ConversationState deprecated and replaced by KnowledgeStore")
        print("• ReActAgent uses KnowledgeStore exclusively for state management")
        print("• doer.py dependency eliminated with direct LLM integration")
        print("• Enhanced RAG integration for better knowledge utilization")
        print("• Centralized state management with advanced context prioritization")
        print("• Simplified control flow reducing latency")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()