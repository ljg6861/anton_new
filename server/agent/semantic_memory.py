"""
Semantic Memory Store: Persistent domain-scoped facts and knowledge

Unlike episodic memory (per-run logs), semantic memory holds stable facts,
constraints, and preferences that guide planning and execution across sessions.
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class SemanticFact:
    """Represents a single semantic fact in memory"""
    id: str
    domain: str
    text: str
    entities: Dict[str, Any]
    tags: List[str]
    confidence: float
    support_count: int
    created_at: datetime
    updated_at: datetime
    hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'domain': self.domain,
            'text': self.text,
            'entities': self.entities,
            'tags': self.tags,
            'confidence': self.confidence,
            'support_count': self.support_count,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'hash': self.hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticFact':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            domain=data['domain'],
            text=data['text'],
            entities=data['entities'],
            tags=data['tags'],
            confidence=data['confidence'],
            support_count=data['support_count'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            hash=data['hash']
        )


class SemanticMemoryStore:
    """
    Stores and retrieves persistent semantic facts across domains.
    
    Features:
    - Domain-scoped fact storage
    - Confidence-based filtering
    - Deduplication with support counting
    - Time decay ranking (slower than episodic)
    - Semantic similarity search
    """
    
    def __init__(self, db_path: str = "semantic_memory.db", rag_manager=None):
        self.db_path = db_path
        self.rag_manager = rag_manager
        self.time_decay_days = 180  # Much slower decay than episodic (6 months)
        self.similarity_threshold = 0.85  # For deduplication
        self.min_confidence = 0.7  # Minimum confidence to store
        
        self._initialize_database()
        logger.info("Semantic memory store initialized successfully")
    
    def _initialize_database(self):
        """Initialize SQLite database with facts table and indexes"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Create facts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    text TEXT NOT NULL,
                    entities TEXT,  -- JSON string
                    tags TEXT,      -- JSON string  
                    confidence REAL NOT NULL,
                    support_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    hash TEXT UNIQUE NOT NULL
                )
            """)
            
            # Create indexes for efficient querying
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_domain_created 
                ON facts(domain, created_at DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_hash 
                ON facts(hash)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_confidence 
                ON facts(confidence DESC)
            """)
            
            conn.commit()
    
    def _generate_fact_hash(self, domain: str, text: str) -> str:
        """Generate hash for fact deduplication"""
        content = f"{domain.lower()}:{text.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two text strings"""
        if not self.rag_manager:
            # Fallback to simple text comparison
            return 1.0 if text1.lower().strip() == text2.lower().strip() else 0.0
        
        try:
            embeddings = self.rag_manager.model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def _calculate_relevance_score(self, fact: SemanticFact, query: str = None) -> float:
        """
        Calculate relevance score for ranking facts.
        Factors: recency_decay × support_count × similarity(query, text) × confidence
        """
        # Time decay (much slower than episodic - 180 days half-life)
        days_old = (datetime.now() - fact.created_at).days
        time_factor = max(0.1, 1.0 - (days_old / self.time_decay_days))
        
        # Support count factor (recurring facts are more important)
        support_factor = min(3.0, 1.0 + (fact.support_count - 1) * 0.2)
        
        # Confidence factor (always prefer higher confidence)
        confidence_factor = fact.confidence
        
        # Query similarity factor
        similarity_factor = 1.0
        if query and query.strip():
            similarity_factor = self._calculate_similarity(query, fact.text)
        
        return time_factor * support_factor * confidence_factor * similarity_factor
    
    async def write_fact(
        self, 
        domain: str, 
        text: str, 
        entities: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        confidence: float = 0.7
    ) -> Optional[str]:
        """
        Write a semantic fact to memory with deduplication.
        
        Args:
            domain: Domain scope (e.g., 'chess', 'code_dev', 'ci_cd')
            text: The fact text (≤ 2-3 sentences)
            entities: Named entities mentioned in the fact
            tags: Categorization tags
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Fact ID if stored, None if rejected/deduplicated
        """
        # Validate inputs
        if confidence < self.min_confidence:
            logger.info(f"Fact rejected: confidence {confidence} below threshold {self.min_confidence}")
            return None
        
        if len(text.strip()) == 0:
            logger.warning("Empty fact text provided")
            return None
        
        # Clean and prepare data
        entities = entities or {}
        tags = tags or []
        fact_hash = self._generate_fact_hash(domain, text)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Check for exact hash match first
            existing = conn.execute(
                "SELECT id, support_count, confidence FROM facts WHERE hash = ?",
                (fact_hash,)
            ).fetchone()
            
            if existing:
                # Update existing fact - increment support and potentially update confidence
                new_support = existing['support_count'] + 1
                new_confidence = max(existing['confidence'], confidence)
                
                conn.execute("""
                    UPDATE facts 
                    SET support_count = ?, confidence = ?, updated_at = ?
                    WHERE id = ?
                """, (new_support, new_confidence, datetime.now(), existing['id']))
                
                conn.commit()
                logger.info(f"Updated existing fact support_count to {new_support}")
                return existing['id']
            
            # Check for similar facts using semantic similarity
            if self.rag_manager:
                similar_facts = await self._find_similar_facts(domain, text)
                for similar_fact, similarity in similar_facts:
                    if similarity >= self.similarity_threshold:
                        # Merge with similar fact
                        new_support = similar_fact.support_count + 1
                        new_confidence = max(similar_fact.confidence, confidence)
                        
                        conn.execute("""
                            UPDATE facts 
                            SET support_count = ?, confidence = ?, updated_at = ?
                            WHERE id = ?
                        """, (new_support, new_confidence, datetime.now(), similar_fact.id))
                        
                        conn.commit()
                        logger.info(f"Merged with similar fact (similarity: {similarity:.3f})")
                        return similar_fact.id
            
            # Create new fact
            fact_id = str(uuid4())
            now = datetime.now()
            
            conn.execute("""
                INSERT INTO facts (
                    id, domain, text, entities, tags, confidence, 
                    support_count, created_at, updated_at, hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact_id, domain, text, json.dumps(entities), json.dumps(tags),
                confidence, 1, now, now, fact_hash
            ))
            
            conn.commit()
            logger.info(f"Created new semantic fact in domain '{domain}' with confidence {confidence}")
            return fact_id
    
    async def _find_similar_facts(self, domain: str, text: str) -> List[Tuple[SemanticFact, float]]:
        """Find semantically similar facts in the same domain"""
        facts = await self.retrieve_facts(domain, limit=50)  # Get recent facts for comparison
        
        similar_facts = []
        for fact in facts:
            similarity = self._calculate_similarity(text, fact.text)
            if similarity > 0.7:  # Only consider reasonably similar facts
                similar_facts.append((fact, similarity))
        
        return sorted(similar_facts, key=lambda x: x[1], reverse=True)
    
    async def retrieve_facts(
        self,
        domain: str,
        query: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 6
    ) -> List[SemanticFact]:
        """
        Retrieve semantic facts with ranking by relevance.
        
        Args:
            domain: Domain to search in
            query: Optional query text for semantic matching
            entities: Optional entity filters
            tags: Optional tag filters
            limit: Maximum number of facts to return
            
        Returns:
            List of ranked SemanticFact objects
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Build query with filters
            sql = "SELECT * FROM facts WHERE domain = ?"
            params = [domain]
            
            # Add tag filtering
            if tags:
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append("tags LIKE ?")
                    params.append(f'%"{tag}"%')
                if tag_conditions:
                    sql += " AND (" + " OR ".join(tag_conditions) + ")"
            
            # Add entity filtering
            if entities:
                entity_conditions = []
                for key, value in entities.items():
                    entity_conditions.append("entities LIKE ?")
                    params.append(f'%"{key}": "{value}"%')  # Look for key-value pair
                if entity_conditions:
                    sql += " AND (" + " OR ".join(entity_conditions) + ")"
            
            # Order by confidence and recency for initial filtering
            sql += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
            params.append(limit * 3)  # Get more for ranking
            
            rows = conn.execute(sql, params).fetchall()
        
        # Convert to SemanticFact objects
        facts = []
        for row in rows:
            fact = SemanticFact(
                id=row['id'],
                domain=row['domain'],
                text=row['text'],
                entities=json.loads(row['entities'] or '{}'),
                tags=json.loads(row['tags'] or '[]'),
                confidence=row['confidence'],
                support_count=row['support_count'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                hash=row['hash']
            )
            facts.append(fact)
        
        # Calculate relevance scores and re-rank
        scored_facts = []
        for fact in facts:
            score = self._calculate_relevance_score(fact, query)
            scored_facts.append((score, fact))
        
        # Sort by relevance score and return top results
        scored_facts.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored_facts[:limit]]
    
    async def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics about facts in a domain"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_facts,
                    AVG(confidence) as avg_confidence,
                    SUM(support_count) as total_support,
                    MAX(created_at) as latest_fact,
                    MIN(created_at) as oldest_fact
                FROM facts WHERE domain = ?
            """, (domain,)).fetchone()
            
            return {
                'domain': domain,
                'total_facts': stats['total_facts'],
                'avg_confidence': stats['avg_confidence'] or 0.0,
                'total_support': stats['total_support'] or 0,
                'latest_fact': stats['latest_fact'],
                'oldest_fact': stats['oldest_fact']
            }
    
    async def promote_from_episode(
        self,
        domain: str,
        episode_summary: str,
        outcome: Dict[str, Any],
        confidence: float = 0.8
    ) -> Optional[str]:
        """
        Promote an episodic experience to a semantic fact.
        
        This is typically called by the evaluator after successful outcomes
        to distill lessons learned into persistent knowledge.
        """
        # Extract key insight from episode
        if outcome.get('status') != 'pass':
            logger.info("Not promoting failed episode to semantic memory")
            return None
        
        # Simple heuristic to create semantic fact from episode
        # In practice, this could use LLM to distill the insight
        fact_text = f"Learned: {episode_summary}"
        
        # Extract entities and tags from episode context
        entities = {
            'source': 'episodic_promotion',
            'original_outcome': outcome.get('status', 'unknown')
        }
        
        tags = ['promoted', 'experience_derived']
        
        # Add domain-specific tags
        if 'error' in episode_summary.lower() or 'fail' in episode_summary.lower():
            tags.append('error_pattern')
        if 'success' in episode_summary.lower() or 'work' in episode_summary.lower():
            tags.append('success_pattern')
        
        return await self.write_fact(
            domain=domain,
            text=fact_text,
            entities=entities,
            tags=tags,
            confidence=confidence
        )
    
    async def cleanup_old_facts(self, max_age_days: int = 365) -> int:
        """Remove very old facts with low support counts"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                DELETE FROM facts 
                WHERE created_at < ? AND support_count <= 1 AND confidence < 0.8
            """, (cutoff_date,))
            
            deleted_count = result.rowcount
            conn.commit()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old semantic facts")
            
            return deleted_count


# Global semantic memory instance
from server.agent.rag_manager import rag_manager
semantic_memory_store = SemanticMemoryStore(rag_manager=rag_manager)
