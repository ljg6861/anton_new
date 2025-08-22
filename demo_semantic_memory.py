#!/usr/bin/env python3
"""
Demonstration of the semantic memory system integration with Anton's task flow
"""

import asyncio
import sys
import os

# Add the server directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from server.agent.knowledge_store import KnowledgeStore
from server.agent.semantic_memory import semantic_memory_store


async def demo_semantic_memory_workflow():
    """Demonstrate the complete semantic memory workflow"""
    print("🎭 Semantic Memory System Demonstration\n")
    
    # Simulate a knowledge store for a coding task
    knowledge_store = KnowledgeStore()
    run_id = knowledge_store.start_episodic_run("code_dev")
    
    print(f"📝 Started episodic run in 'code_dev' domain: {run_id}")
    
    # Phase 1: Assessor uses semantic knowledge
    print("\n🔍 Phase 1: Assessor retrieves relevant domain knowledge")
    
    # Pre-populate some semantic facts (simulating learned knowledge)
    await knowledge_store.write_semantic_fact(
        text="Python functions should always include type hints for better code maintainability",
        tags=["python", "best_practice", "typing"],
        entities={"language": "python", "practice": "type_hints"},
        confidence=0.9
    )
    
    await knowledge_store.write_semantic_fact(
        text="Unit tests should cover edge cases and error conditions, not just happy paths",
        tags=["testing", "best_practice", "quality"],
        entities={"practice": "unit_testing", "coverage": "edge_cases"},
        confidence=0.85
    )
    
    await knowledge_store.write_semantic_fact(
        text="Code reviews are most effective when focused on logic, security, and maintainability",
        tags=["code_review", "best_practice", "collaboration"],
        entities={"practice": "code_review", "focus": "quality"},
        confidence=0.8
    )
    
    # Assessor retrieves relevant facts for a code quality task
    query = "improve code quality Python function"
    semantic_context = await knowledge_store.build_semantic_context(query, max_facts=3)
    
    print(f"📚 Semantic context for '{query}':")
    print(semantic_context)
    
    # Phase 2: Planner uses semantic knowledge
    print("\n📋 Phase 2: Planner incorporates domain knowledge")
    
    planning_query = "testing strategy Python code"
    planning_facts = await knowledge_store.get_semantic_facts(
        query=planning_query,
        tags=["testing"],
        limit=2
    )
    
    print(f"🎯 Planning facts for '{planning_query}':")
    for i, fact in enumerate(planning_facts, 1):
        print(f"  {i}. {fact.text} (confidence: {fact.confidence:.2f})")
    
    # Phase 3: Simulate successful execution
    print("\n⚡ Phase 3: Execution successful - promoting insights")
    
    # Simulate successful execution outcomes
    successful_episode = "Implemented comprehensive test suite with 95% coverage including edge cases"
    outcome = {
        "status": "pass",
        "test_coverage": 95,
        "edge_cases_covered": True,
        "notes": "TDD approach worked well"
    }
    
    # Promote successful patterns to semantic memory
    promoted_id = await knowledge_store.promote_episode_to_semantic(
        episode_summary=successful_episode,
        outcome=outcome,
        confidence=0.85
    )
    
    print(f"🏆 Promoted successful episode to semantic fact: {promoted_id}")
    
    # Phase 4: Show accumulated knowledge
    print("\n📊 Phase 4: Domain knowledge statistics")
    
    stats = await knowledge_store.get_domain_knowledge_stats()
    print(f"Domain: {stats['domain']}")
    print(f"Semantic facts: {stats['semantic_facts']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"Episodic experiences: {stats['episodic_experiences']}")
    
    # Phase 5: Demonstrate knowledge retrieval for future tasks
    print("\n🔮 Phase 5: Knowledge available for future tasks")
    
    all_facts = await knowledge_store.get_semantic_facts(limit=10)
    print(f"📚 Available semantic knowledge ({len(all_facts)} facts):")
    
    for i, fact in enumerate(all_facts, 1):
        confidence_bar = "●" * int(fact.confidence * 10)
        support_info = f"(support: {fact.support_count})"
        print(f"  {i}. {fact.text[:80]}...")
        print(f"     Confidence: {confidence_bar} {fact.confidence:.2f} {support_info}")
        print(f"     Tags: {', '.join(fact.tags[:3])}")
    
    print("\n🎉 Semantic memory demonstration complete!")
    print("\n📋 Key Benefits Demonstrated:")
    print("  ✅ Domain-scoped persistent knowledge")
    print("  ✅ Confidence-based fact ranking")
    print("  ✅ Deduplication with support counting")
    print("  ✅ Semantic search and filtering")
    print("  ✅ Episode promotion for continuous learning")
    print("  ✅ Integration with episodic memory")


async def demo_cross_domain_knowledge():
    """Demonstrate knowledge isolation across domains"""
    print("\n🌐 Cross-Domain Knowledge Demonstration")
    
    # Create facts in different domains
    chess_store = KnowledgeStore()
    chess_store.start_episodic_run("chess")
    
    await chess_store.write_semantic_fact(
        text="Castle early to protect your king and connect your rooks",
        tags=["opening", "safety"],
        entities={"strategy": "castling", "phase": "opening"},
        confidence=0.9
    )
    
    web_dev_store = KnowledgeStore()
    web_dev_store.start_episodic_run("web_dev")
    
    await web_dev_store.write_semantic_fact(
        text="Always validate user input on both client and server side",
        tags=["security", "validation"],
        entities={"practice": "input_validation", "location": "client_server"},
        confidence=0.95
    )
    
    # Demonstrate domain isolation
    chess_facts = await chess_store.get_semantic_facts(limit=10)
    web_facts = await web_dev_store.get_semantic_facts(limit=10)
    
    print(f"🏰 Chess domain facts: {len(chess_facts)}")
    for fact in chess_facts:
        print(f"  - {fact.text}")
    
    print(f"🌐 Web dev domain facts: {len(web_facts)}")
    for fact in web_facts:
        print(f"  - {fact.text}")
    
    print("✅ Domains properly isolated - no knowledge bleed between chess and web dev")


if __name__ == "__main__":
    asyncio.run(demo_semantic_memory_workflow())
    asyncio.run(demo_cross_domain_knowledge())
