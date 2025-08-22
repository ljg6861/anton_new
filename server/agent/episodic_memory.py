"""
Episodic Memory Module for Anton AI Assistant

Records what happened in each run, namespaced by domain and role.
Provides retrieval of past context, decisions, and outcomes for improved performance.
Supports deduplication and decay mechanisms.
"""

import sqlite3
import uuid
import time
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path

from server.agent.rag_manager import rag_manager

logger = logging.getLogger(__name__)


@dataclass
class EpisodicEntry:
    """Represents a single episodic memory entry"""
    id: str
    run_id: str
    parent_run_id: Optional[str]
    domain: str
    role: str
    summary: str
    tags: List[str]
    entities: Dict[str, Any]
    outcome: Dict[str, Any]  # {status: 'pass'|'fail'|'partial', metrics, notes}
    confidence: float
    support_count: int
    created_at: float
    updated_at: float
    hash: str


class DomainResolver:
    """Resolves domain names with similarity matching to prefer existing domains"""
    
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
        self._domain_embeddings_cache = {}
    
    def resolve_domain(self, proposed_domain: str, existing_domains: List[str]) -> str:
        """
        Resolve domain name, preferring existing domains if similarity >= threshold.
        
        Args:
            proposed_domain: The proposed domain name
            existing_domains: List of existing domain names
            
        Returns:
            The resolved domain name (existing or new)
        """
        if not existing_domains:
            return proposed_domain.lower().strip()
        
        # Normalize proposed domain
        proposed_domain = proposed_domain.lower().strip()
        
        # Check for exact match first
        if proposed_domain in existing_domains:
            return proposed_domain
        
        # Calculate similarity with existing domains using embeddings
        try:
            # Get embedding for proposed domain
            proposed_embedding = rag_manager.model.encode([proposed_domain])[0]
            
            best_match = None
            best_similarity = 0.0
            
            for existing_domain in existing_domains:
                # Use cached embedding if available
                if existing_domain not in self._domain_embeddings_cache:
                    self._domain_embeddings_cache[existing_domain] = rag_manager.model.encode([existing_domain])[0]
                
                existing_embedding = self._domain_embeddings_cache[existing_domain]
                
                # Calculate cosine similarity
                similarity = float(
                    proposed_embedding.dot(existing_embedding) / 
                    (abs(proposed_embedding).sum() * abs(existing_embedding).sum() + 1e-9)
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = existing_domain
            
            # Return existing domain if similarity meets threshold
            if best_similarity >= self.similarity_threshold:
                logger.info(f"Resolved domain '{proposed_domain}' to existing '{best_match}' (similarity: {best_similarity:.3f})")
                return best_match
            
        except Exception as e:
            logger.warning(f"Domain similarity calculation failed: {e}")
        
        # Return new domain if no good match found
        logger.info(f"Creating new domain '{proposed_domain}' (best match similarity: {best_similarity:.3f})")
        return proposed_domain


class EpisodicMemoryStore:
    """
    Persistent storage for episodic memories with domain-namespaced retrieval.
    Handles deduplication, decay, and semantic retrieval of past experiences.
    """
    
    def __init__(self, db_path: str = "episodic_memory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.domain_resolver = DomainResolver()
        
        # Current session tracking
        self.current_run_id = None
        self.current_domain = None
        self.session_entries = []
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for episodic memory storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Episodes table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS episodes (
                        id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        parent_run_id TEXT,
                        domain TEXT NOT NULL,
                        role TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        tags TEXT NOT NULL,  -- JSON array
                        entities TEXT NOT NULL,  -- JSON object
                        outcome TEXT NOT NULL,  -- JSON object
                        confidence REAL NOT NULL,
                        support_count INTEGER DEFAULT 1,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        hash TEXT UNIQUE NOT NULL
                    )
                """)
                
                # Create indexes for efficient queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain_role_created ON episodes(domain, role, created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON episodes(run_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags ON episodes(tags)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities ON episodes(entities)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash ON episodes(hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain ON episodes(domain)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_role ON episodes(role)")
                
                conn.commit()
                logger.info("Episodic memory database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize episodic memory database: {e}")
            raise
    
    def _generate_hash(self, domain: str, role: str, summary: str, tags: List[str], entities: Dict) -> str:
        """Generate content hash for deduplication"""
        content = f"{domain}:{role}:{summary}:{sorted(tags)}:{json.dumps(entities, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def start_run(self, domain: str, parent_run_id: Optional[str] = None) -> str:
        """
        Start a new episodic memory run.
        
        Args:
            domain: The domain context (e.g., 'chess', 'tool_dev')
            parent_run_id: Optional parent run ID for hierarchical tracking
            
        Returns:
            Generated run ID
        """
        # Resolve domain against existing domains
        existing_domains = self.get_existing_domains()
        resolved_domain = self.domain_resolver.resolve_domain(domain, existing_domains)
        
        run_id = str(uuid.uuid4())
        self.current_run_id = run_id
        self.current_domain = resolved_domain
        self.session_entries = []
        
        logger.info(f"Started episodic memory run {run_id} for domain '{resolved_domain}'")
        return run_id
    
    def record_episode(
        self,
        role: str,
        summary: str,
        tags: List[str] = None,
        entities: Dict[str, Any] = None,
        outcome: Dict[str, Any] = None,
        confidence: float = 1.0
    ) -> str:
        """
        Record an episodic memory entry.
        
        Args:
            role: The role that performed this episode (assessor, researcher, planner, executor, evaluator)
            summary: Human-readable summary of what happened
            tags: List of searchable tags
            entities: Structured entities involved (files, tools, concepts, etc.)
            outcome: Outcome information {status: 'pass'|'fail'|'partial', metrics, notes}
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Episode ID
        """
        if not self.current_run_id or not self.current_domain:
            raise ValueError("Must start a run before recording episodes")
        
        # Set defaults
        tags = tags or []
        entities = entities or {}
        outcome = outcome or {"status": "unknown"}
        
        # Generate content hash for deduplication
        content_hash = self._generate_hash(self.current_domain, role, summary, tags, entities)
        
        episode_id = str(uuid.uuid4())
        current_time = time.time()
        
        episode = EpisodicEntry(
            id=episode_id,
            run_id=self.current_run_id,
            parent_run_id=None,  # Could be enhanced later for sub-runs
            domain=self.current_domain,
            role=role,
            summary=summary,
            tags=tags,
            entities=entities,
            outcome=outcome,
            confidence=confidence,
            support_count=1,
            created_at=current_time,
            updated_at=current_time,
            hash=content_hash
        )
        
        # Store in database with deduplication
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Check for existing entry with same hash
                    cursor.execute("SELECT id, support_count, confidence FROM episodes WHERE hash = ?", (content_hash,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing entry - increment support and adjust confidence
                        existing_id, support_count, existing_confidence = existing
                        new_support_count = support_count + 1
                        # Weighted average of confidences
                        new_confidence = (existing_confidence * support_count + confidence) / new_support_count
                        
                        cursor.execute("""
                            UPDATE episodes 
                            SET support_count = ?, confidence = ?, updated_at = ?
                            WHERE id = ?
                        """, (new_support_count, new_confidence, current_time, existing_id))
                        
                        logger.info(f"Updated existing episode {existing_id} (support: {new_support_count}, confidence: {new_confidence:.3f})")
                        episode_id = existing_id
                    else:
                        # Insert new entry
                        cursor.execute("""
                            INSERT INTO episodes (
                                id, run_id, parent_run_id, domain, role, summary, 
                                tags, entities, outcome, confidence, support_count,
                                created_at, updated_at, hash
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            episode.id, episode.run_id, episode.parent_run_id,
                            episode.domain, episode.role, episode.summary,
                            json.dumps(episode.tags), json.dumps(episode.entities),
                            json.dumps(episode.outcome), episode.confidence, episode.support_count,
                            episode.created_at, episode.updated_at, episode.hash
                        ))
                        
                        logger.info(f"Recorded new episode {episode_id} for role '{role}' in domain '{self.current_domain}'")
                    
                    conn.commit()
                    
                    # Track in current session
                    self.session_entries.append(episode)
                    
            except sqlite3.IntegrityError as e:
                # Handle race condition where hash was inserted between check and insert
                logger.warning(f"Hash collision detected for episode: {e}")
                return self.record_episode(role, summary, tags, entities, outcome, confidence)
            except Exception as e:
                logger.error(f"Failed to record episode: {e}")
                raise
        
        return episode_id
    
    def retrieve_episodes(
        self,
        domain: str = None,
        role: str = None,
        tags: List[str] = None,
        limit: int = 10,
        min_confidence: float = 0.0,
        time_decay_hours: float = 168.0  # 1 week default
    ) -> List[EpisodicEntry]:
        """
        Retrieve episodic memories with filtering and time decay.
        
        Args:
            domain: Filter by domain (None for current domain)
            role: Filter by role (None for any role)
            tags: Filter by tags (must contain all specified tags)
            limit: Maximum number of results
            min_confidence: Minimum confidence threshold
            time_decay_hours: Hours for time decay calculation
            
        Returns:
            List of episodic entries sorted by relevance score
        """
        # Use current domain if none specified
        if domain is None:
            domain = self.current_domain
        
        if not domain:
            logger.warning("No domain specified and no current domain set")
            return []
        
        # Resolve domain
        existing_domains = self.get_existing_domains()
        resolved_domain = self.domain_resolver.resolve_domain(domain, existing_domains)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                conditions = ["domain = ?"]
                params = [resolved_domain]
                
                if role:
                    conditions.append("role = ?")
                    params.append(role)
                
                if min_confidence > 0.0:
                    conditions.append("confidence >= ?")
                    params.append(min_confidence)
                
                # Tag filtering using JSON operations
                if tags:
                    for tag in tags:
                        conditions.append("json_extract(tags, '$') LIKE ?")
                        params.append(f'%"{tag}"%')
                
                query = f"""
                    SELECT id, run_id, parent_run_id, domain, role, summary, tags, entities, 
                           outcome, confidence, support_count, created_at, updated_at, hash
                    FROM episodes 
                    WHERE {' AND '.join(conditions)}
                    ORDER BY confidence DESC, support_count DESC, created_at DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to EpisodicEntry objects with time decay
                current_time = time.time()
                episodes = []
                
                for row in rows:
                    episode = EpisodicEntry(
                        id=row[0],
                        run_id=row[1],
                        parent_run_id=row[2],
                        domain=row[3],
                        role=row[4],
                        summary=row[5],
                        tags=json.loads(row[6]),
                        entities=json.loads(row[7]),
                        outcome=json.loads(row[8]),
                        confidence=row[9],
                        support_count=row[10],
                        created_at=row[11],
                        updated_at=row[12],
                        hash=row[13]
                    )
                    
                    # Apply time decay to confidence
                    age_hours = (current_time - episode.created_at) / 3600
                    if age_hours > 0 and time_decay_hours > 0:
                        decay_factor = max(0.1, 1.0 - (age_hours / time_decay_hours))
                        episode.confidence *= decay_factor
                    
                    episodes.append(episode)
                
                # Re-sort by decayed confidence
                episodes.sort(key=lambda e: (e.confidence, e.support_count), reverse=True)
                
                logger.info(f"Retrieved {len(episodes)} episodes for domain '{resolved_domain}', role '{role}'")
                return episodes
                
        except Exception as e:
            logger.error(f"Failed to retrieve episodes: {e}")
            return []
    
    def search_episodes_semantic(
        self,
        query: str,
        domain: str = None,
        role: str = None,
        limit: int = 5
    ) -> List[EpisodicEntry]:
        """
        Semantic search of episodic memories using embeddings.
        
        Args:
            query: Natural language query
            domain: Filter by domain (None for current domain)
            role: Filter by role (None for any role)
            limit: Maximum number of results
            
        Returns:
            List of episodic entries sorted by semantic similarity
        """
        # Get candidates using regular retrieval
        candidates = self.retrieve_episodes(
            domain=domain,
            role=role,
            limit=limit * 3  # Get more candidates for better semantic filtering
        )
        
        if not candidates:
            return []
        
        try:
            # Generate query embedding
            query_embedding = rag_manager.model.encode([query])[0]
            
            # Calculate semantic similarity for each candidate
            scored_candidates = []
            for episode in candidates:
                # Combine summary and tags for similarity calculation
                episode_text = f"{episode.summary} {' '.join(episode.tags)}"
                episode_embedding = rag_manager.model.encode([episode_text])[0]
                
                # Calculate cosine similarity
                similarity = float(
                    query_embedding.dot(episode_embedding) / 
                    (abs(query_embedding).sum() * abs(episode_embedding).sum() + 1e-9)
                )
                
                # Combine similarity with confidence and support
                score = similarity * episode.confidence * (1 + episode.support_count * 0.1)
                scored_candidates.append((score, episode))
            
            # Sort by combined score and return top results
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            results = [episode for _, episode in scored_candidates[:limit]]
            
            logger.info(f"Semantic search returned {len(results)} episodes for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            # Fallback to regular retrieval
            return candidates[:limit]
    
    def get_existing_domains(self) -> List[str]:
        """Get list of all existing domains"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT domain FROM episodes ORDER BY domain")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get existing domains: {e}")
            return []
    
    def get_run_summary(self, run_id: str = None) -> Dict[str, Any]:
        """
        Get summary of a specific run or current run.
        
        Args:
            run_id: Run ID to summarize (None for current run)
            
        Returns:
            Dictionary with run statistics and episode summaries
        """
        target_run_id = run_id or self.current_run_id
        if not target_run_id:
            return {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get run episodes
                cursor.execute("""
                    SELECT role, summary, outcome, confidence, created_at
                    FROM episodes 
                    WHERE run_id = ?
                    ORDER BY created_at
                """, (target_run_id,))
                
                episodes = cursor.fetchall()
                
                # Calculate statistics
                role_counts = {}
                outcomes = {"pass": 0, "fail": 0, "partial": 0, "unknown": 0}
                total_confidence = 0.0
                
                episode_summaries = []
                for role, summary, outcome_json, confidence, created_at in episodes:
                    role_counts[role] = role_counts.get(role, 0) + 1
                    total_confidence += confidence
                    
                    outcome = json.loads(outcome_json)
                    status = outcome.get("status", "unknown")
                    outcomes[status] = outcomes.get(status, 0) + 1
                    
                    episode_summaries.append({
                        "role": role,
                        "summary": summary,
                        "outcome": outcome,
                        "confidence": confidence,
                        "timestamp": created_at
                    })
                
                return {
                    "run_id": target_run_id,
                    "domain": self.current_domain,
                    "episode_count": len(episodes),
                    "role_distribution": role_counts,
                    "outcome_distribution": outcomes,
                    "average_confidence": total_confidence / len(episodes) if episodes else 0.0,
                    "episodes": episode_summaries
                }
                
        except Exception as e:
            logger.error(f"Failed to get run summary: {e}")
            return {}
    
    def cleanup_old_episodes(self, max_age_days: int = 30, min_support_count: int = 1):
        """
        Clean up old episodes with low support to prevent database bloat.
        
        Args:
            max_age_days: Maximum age in days for episodes to keep
            min_support_count: Minimum support count to keep old episodes
        """
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Delete old episodes with low support
                    cursor.execute("""
                        DELETE FROM episodes 
                        WHERE created_at < ? AND support_count < ?
                    """, (cutoff_time, min_support_count))
                    
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} old episodes")
                        
        except Exception as e:
            logger.error(f"Failed to cleanup old episodes: {e}")


# Global instance
episodic_memory = EpisodicMemoryStore()
