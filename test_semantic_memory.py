#!/usr/bin/env python3
"""
Comprehensive test suite for the semantic memory system
"""

import asyncio
import os
import sys
import tempfile
from unittest.mock import Mock

# Add the server directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from server.agent.semantic_memory import SemanticMemoryStore, SemanticFact
from server.agent.knowledge_store import KnowledgeStore


async def test_semantic_memory_basic_operations():
    """Test basic write and retrieve operations"""
    print("🧪 Testing basic semantic memory operations...")
    
    # Use temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        store = SemanticMemoryStore(db_path=db_path)
        
        # Test 1: Write a semantic fact
        print("\n--- Test 1: Write semantic fact ---")
        fact_id = await store.write_fact(
            domain="chess",
            text="Opening with e4 provides good central control and quick development",
            entities={"opening": "e4", "strategy": "central_control"},
            tags=["opening", "strategy"],
            confidence=0.9
        )
        
        assert fact_id is not None, "Failed to write semantic fact"
        print(f"✅ Written fact with ID: {fact_id}")
        
        # Test 2: Retrieve facts
        print("\n--- Test 2: Retrieve facts ---")
        facts = await store.retrieve_facts(domain="chess", limit=5)
        
        assert len(facts) == 1, f"Expected 1 fact, got {len(facts)}"
        assert facts[0].text.startswith("Opening with e4"), "Fact content mismatch"
        assert facts[0].confidence == 0.9, "Confidence mismatch"
        print(f"✅ Retrieved fact: {facts[0].text[:50]}...")
        
        # Test 3: Deduplication (write similar fact)
        print("\n--- Test 3: Test deduplication ---")
        duplicate_id = await store.write_fact(
            domain="chess",
            text="Opening with e4 provides good central control and quick development",
            confidence=0.85
        )
        
        assert duplicate_id == fact_id, "Deduplication failed - should return same ID"
        
        # Verify support count increased
        updated_facts = await store.retrieve_facts(domain="chess", limit=5)
        assert updated_facts[0].support_count == 2, f"Expected support_count=2, got {updated_facts[0].support_count}"
        assert updated_facts[0].confidence == 0.9, "Should keep higher confidence"
        print(f"✅ Deduplication working: support_count={updated_facts[0].support_count}")
        
        print("✅ Basic operations test passed!")
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


async def test_semantic_memory_filtering():
    """Test filtering by tags and entities"""
    print("\n🧪 Testing semantic memory filtering...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        store = SemanticMemoryStore(db_path=db_path)
        
        # Add multiple facts with different tags and entities
        await store.write_fact(
            domain="code_dev",
            text="Always use type hints in Python functions for better code clarity",
            tags=["python", "best_practice"],
            entities={"language": "python", "feature": "type_hints"},
            confidence=0.9
        )
        
        await store.write_fact(
            domain="code_dev", 
            text="Git commits should have descriptive messages under 50 characters",
            tags=["git", "best_practice"],
            entities={"tool": "git", "practice": "commit_messages"},
            confidence=0.85
        )
        
        await store.write_fact(
            domain="code_dev",
            text="Use virtual environments to isolate Python project dependencies",
            tags=["python", "environment"],
            entities={"language": "python", "tool": "venv"},
            confidence=0.8
        )
        
        # Test tag filtering
        python_facts = await store.retrieve_facts(
            domain="code_dev",
            tags=["python"],
            limit=10
        )
        
        assert len(python_facts) == 2, f"Expected 2 Python facts, got {len(python_facts)}"
        print(f"✅ Tag filtering: Found {len(python_facts)} Python facts")
        
        # Test entity filtering
        git_facts = await store.retrieve_facts(
            domain="code_dev",
            entities={"tool": "git"},
            limit=10
        )
        
        assert len(git_facts) == 1, f"Expected 1 Git fact, got {len(git_facts)}"
        print(f"✅ Entity filtering: Found {len(git_facts)} Git facts")
        
        print("✅ Filtering test passed!")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


async def test_semantic_memory_ranking():
    """Test relevance ranking and time decay"""
    print("\n🧪 Testing semantic memory ranking...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        store = SemanticMemoryStore(db_path=db_path)
        
        # Add facts with different confidence levels
        low_confidence_id = await store.write_fact(
            domain="testing",
            text="Low confidence fact for testing ranking",
            confidence=0.6
        )
        
        # Should be rejected due to low confidence
        assert low_confidence_id is None, "Low confidence fact should be rejected"
        
        high_confidence_id = await store.write_fact(
            domain="testing",
            text="High confidence fact for testing ranking",
            confidence=0.95
        )
        
        medium_confidence_id = await store.write_fact(
            domain="testing", 
            text="Medium confidence fact for testing ranking",
            confidence=0.75
        )
        
        # Retrieve and check ranking (high confidence should come first)
        facts = await store.retrieve_facts(domain="testing", limit=10)
        
        assert len(facts) == 2, f"Expected 2 facts (low confidence rejected), got {len(facts)}"
        assert facts[0].confidence > facts[1].confidence, "Facts not ranked by confidence"
        print(f"✅ Ranking test: High confidence fact ranked first ({facts[0].confidence} > {facts[1].confidence})")
        
        print("✅ Ranking test passed!")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


async def test_knowledge_store_integration():
    """Test semantic memory integration with KnowledgeStore"""
    print("\n🧪 Testing KnowledgeStore integration...")
    
    knowledge_store = KnowledgeStore()
    
    # Start a domain run
    run_id = knowledge_store.start_episodic_run("integration_test")
    assert knowledge_store.current_domain == "integration_test"
    print(f"✅ Started domain run: {run_id}")
    
    # Write semantic facts through knowledge store
    fact_id = await knowledge_store.write_semantic_fact(
        text="Integration testing is crucial for multi-component systems",
        tags=["testing", "integration"],
        entities={"practice": "integration_testing"},
        confidence=0.9
    )
    
    assert fact_id is not None, "Failed to write fact through knowledge store"
    print(f"✅ Written fact through KnowledgeStore: {fact_id}")
    
    # Retrieve facts
    facts = await knowledge_store.get_semantic_facts(query="testing systems", limit=5)
    assert len(facts) > 0, "Failed to retrieve facts through knowledge store"
    print(f"✅ Retrieved {len(facts)} facts through KnowledgeStore")
    
    # Build semantic context
    context = await knowledge_store.build_semantic_context("testing integration")
    assert "RELEVANT DOMAIN KNOWLEDGE" in context, "Context format incorrect"
    assert "Integration testing" in context, "Fact content not in context"
    print(f"✅ Built semantic context: {len(context)} characters")
    
    print("✅ KnowledgeStore integration test passed!")


async def test_episode_promotion():
    """Test promoting episodic experiences to semantic memory"""
    print("\n🧪 Testing episode promotion to semantic memory...")
    
    knowledge_store = KnowledgeStore()
    knowledge_store.start_episodic_run("promotion_test")
    
    # Simulate successful episode promotion
    episode_summary = "Successfully deployed application using Docker containerization"
    outcome = {
        "status": "pass",
        "deployment_time": 45,
        "notes": "Docker deployment worked smoothly"
    }
    
    fact_id = await knowledge_store.promote_episode_to_semantic(
        episode_summary=episode_summary,
        outcome=outcome,
        confidence=0.8
    )
    
    assert fact_id is not None, "Failed to promote episode to semantic memory"
    print(f"✅ Promoted episode to semantic fact: {fact_id}")
    
    # Verify the promoted fact exists
    facts = await knowledge_store.get_semantic_facts(query="Docker deployment")
    assert len(facts) > 0, "Promoted fact not found"
    
    promoted_fact = facts[0]
    assert "Docker" in promoted_fact.text, "Promoted fact doesn't contain key information"
    assert "promoted" in promoted_fact.tags, "Promoted fact missing 'promoted' tag"
    print(f"✅ Verified promoted fact: {promoted_fact.text[:50]}...")
    
    print("✅ Episode promotion test passed!")


async def test_domain_stats():
    """Test domain statistics functionality"""
    print("\n🧪 Testing domain statistics...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        store = SemanticMemoryStore(db_path=db_path)
        
        # Add several facts to a domain
        for i in range(3):
            await store.write_fact(
                domain="stats_test",
                text=f"Test fact number {i+1} for statistics testing",
                confidence=0.8 + (i * 0.05),  # Varying confidence
                tags=["test", f"fact_{i+1}"]
            )
        
        # Get domain statistics
        stats = await store.get_domain_stats("stats_test")
        
        assert stats["total_facts"] == 3, f"Expected 3 facts, got {stats['total_facts']}"
        assert stats["avg_confidence"] > 0.8, f"Average confidence too low: {stats['avg_confidence']}"
        assert stats["total_support"] == 3, f"Expected total support 3, got {stats['total_support']}"
        
        print(f"✅ Domain stats: {stats['total_facts']} facts, avg confidence: {stats['avg_confidence']:.3f}")
        
        print("✅ Domain statistics test passed!")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


async def run_all_tests():
    """Run all semantic memory tests"""
    print("🚀 Starting comprehensive semantic memory testing...\n")
    
    tests = [
        test_semantic_memory_basic_operations,
        test_semantic_memory_filtering,
        test_semantic_memory_ranking,
        test_knowledge_store_integration,
        test_episode_promotion,
        test_domain_stats
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All semantic memory tests passed!")
    else:
        print(f"⚠️ {failed} tests failed. See errors above.")
    
    return failed == 0


if __name__ == "__main__":
    asyncio.run(run_all_tests())
