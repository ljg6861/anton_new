#!/usr/bin/env python3
"""
Demonstration of the ReAct Agent Message Summarization Feature

This script demonstrates how the ReAct agent automatically manages memory
by summarizing message history when it exceeds 15k tokens.
"""

import asyncio
import logging
from server.agent.react.react_agent import ReActAgent
from server.agent.knowledge_store import KnowledgeStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demonstrate_summarization():
    """Demonstrate the message summarization feature with realistic conversation"""
    
    print("🤖 ReAct Agent Message Summarization Demo")
    print("=" * 50)
    
    # Initialize the agent
    knowledge_store = KnowledgeStore()
    agent = ReActAgent(
        api_base_url="http://localhost:8002",
        tools=[],
        knowledge_store=knowledge_store,
        max_iterations=5
    )
    
    # Create a realistic conversation that grows over time
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant specialized in data analysis and software development."},
        {"role": "user", "content": "I need help building a customer analytics dashboard."},
        {"role": "assistant", "content": "I'll help you create a comprehensive customer analytics dashboard. Let's start by understanding your requirements."}
    ]
    
    print(f"📊 Starting with {len(conversation)} messages")
    
    # Simulate a long conversation by adding realistic exchanges
    topics = [
        "data collection strategy",
        "database schema design", 
        "ETL pipeline architecture",
        "visualization requirements",
        "user authentication system",
        "performance optimization",
        "real-time data streaming",
        "machine learning integration",
        "reporting automation",
        "security considerations"
    ]
    
    for round_num in range(8):  # 8 rounds of conversation
        print(f"\n📈 Conversation Round {round_num + 1}")
        
        for i, topic in enumerate(topics):
            # Add detailed user request
            user_msg = f"""I need detailed guidance on {topic} for our customer analytics platform. 
            This should include best practices, implementation strategies, scalability considerations, 
            performance optimization techniques, security measures, and integration with existing systems. 
            Please provide specific recommendations for tools, frameworks, and architectural patterns 
            that would be most suitable for handling large-scale customer data processing and analysis."""
            
            # Add detailed assistant response
            assistant_msg = f"""For {topic}, I recommend implementing a comprehensive approach that considers 
            both immediate needs and long-term scalability. Here's a detailed analysis: First, we should evaluate 
            your current infrastructure and data volume requirements. Then, implement industry best practices 
            including proper data governance, security protocols, and performance monitoring. The solution should 
            incorporate modern frameworks and tools that provide flexibility and maintainability while ensuring 
            robust data processing capabilities and user-friendly interfaces."""
            
            # Add tool execution result
            tool_msg = f"""Successfully analyzed requirements for {topic}. Key findings: Architecture pattern 
            identified, performance benchmarks established, security framework defined, implementation roadmap 
            created. Recommendations include specific technology stack, database optimizations, caching strategies, 
            and monitoring solutions. Estimated implementation time and resource requirements calculated based on 
            project scope and complexity."""
            
            conversation.extend([
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
                {"role": "function", "name": f"analysis_tool_{i}", "content": tool_msg}
            ])
        
        # Check token count and demonstrate summarization
        current_tokens = agent.calculate_total_message_tokens(conversation)
        print(f"   Current tokens: {current_tokens:,}")
        print(f"   Message count: {len(conversation)}")
        
        if current_tokens > 15000:
            print(f"   🔄 Token limit exceeded! Starting summarization...")
            summarized = await agent.check_and_summarize_if_needed(conversation)
            new_tokens = agent.calculate_total_message_tokens(summarized)
            
            print(f"   ✅ Summarization complete:")
            print(f"      Before: {len(conversation)} messages, {current_tokens:,} tokens")
            print(f"      After:  {len(summarized)} messages, {new_tokens:,} tokens")
            print(f"      Reduction: {current_tokens - new_tokens:,} tokens ({((current_tokens - new_tokens) / current_tokens * 100):.1f}%)")
            
            # Show a sample of the summary
            for msg in summarized:
                if "CONVERSATION SUMMARY" in msg.get("content", ""):
                    print(f"\n   📋 Summary Preview:")
                    print(f"      {msg['content'][:200]}...")
                    break
            
            conversation = summarized
            break
        else:
            print(f"   ⏳ Under limit, continuing...")
    
    print(f"\n🎯 Final Results:")
    print(f"   Messages: {len(conversation)}")
    print(f"   Tokens: {agent.calculate_total_message_tokens(conversation):,}")
    print(f"   Status: Ready for continued conversation!")

if __name__ == "__main__":
    asyncio.run(demonstrate_summarization())
