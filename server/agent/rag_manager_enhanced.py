"""
Improved RAG Manager with fallback implementation for environments without ML dependencies.

This implementation provides both a full-featured FAISS-based RAG system and a
simple fallback using text similarity for environments without ML libraries.
"""
import os
import pickle
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """Structured representation of a knowledge entry."""
    text: str
    source: str
    keywords: List[str]
    timestamp: float
    entry_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntry':
        return cls(**data)


class SimpleRAGManager:
    """
    A robust RAG manager with fallback implementation that doesn't require ML libraries.
    
    Features:
    - Text-based similarity using keyword matching and text overlap
    - JSON-based storage for portability and debugging
    - Efficient keyword indexing for fast retrieval
    - Memory management to prevent excessive storage
    - Error handling and graceful degradation
    """
    
    def __init__(self, storage_path: str = "knowledge_store.json", max_entries: int = 1000):
        self.storage_path = storage_path
        self.max_entries = max_entries
        self.knowledge_store: Dict[str, KnowledgeEntry] = {}
        self.keyword_index: Dict[str, List[str]] = defaultdict(list)  # keyword -> list of entry_ids
        self._load_from_disk()
        logger.info(f"SimpleRAGManager initialized with {len(self.knowledge_store)} entries")
    
    def add_knowledge(self, text: str, source: str, keywords: Optional[List[str]] = None) -> bool:
        """
        Add knowledge to the store.
        
        Args:
            text: The knowledge text to store
            source: Source identifier for the knowledge
            keywords: Optional keywords for better retrieval
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            import time
            
            # Generate unique ID for this entry
            entry_id = hashlib.md5(f"{text}_{source}_{time.time()}".encode()).hexdigest()
            
            # Extract keywords if not provided
            if keywords is None:
                keywords = self._extract_keywords(text)
            
            # Create knowledge entry
            entry = KnowledgeEntry(
                text=text,
                source=source,
                keywords=keywords,
                timestamp=time.time(),
                entry_id=entry_id
            )
            
            # Add to store
            self.knowledge_store[entry_id] = entry
            
            # Update keyword index
            for keyword in keywords:
                self.keyword_index[keyword.lower()].append(entry_id)
            
            # Manage memory by removing old entries if we exceed max
            self._manage_memory()
            
            logger.debug(f"Added knowledge entry: {entry_id} from {source}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}", exc_info=True)
            return False
    
    def retrieve_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge based on query.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of knowledge entries as dictionaries
        """
        try:
            if not self.knowledge_store:
                return []
            
            # Score all entries based on relevance to query
            scored_entries = []
            query_lower = query.lower()
            query_keywords = self._extract_keywords(query)
            
            for entry_id, entry in self.knowledge_store.items():
                score = self._calculate_relevance_score(entry, query_lower, query_keywords)
                if score > 0:
                    scored_entries.append((score, entry))
            
            # Sort by score and return top results
            scored_entries.sort(key=lambda x: x[0], reverse=True)
            
            results = []
            for score, entry in scored_entries[:top_k]:
                results.append({
                    'text': entry.text,
                    'source': entry.source,
                    'keywords': entry.keywords,
                    'relevance_score': score
                })
            
            logger.debug(f"Retrieved {len(results)} relevant entries for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge: {e}", exc_info=True)
            return []
    
    def save(self) -> bool:
        """Save the knowledge store to disk."""
        try:
            # Convert knowledge store to serializable format
            data = {
                'entries': {k: v.to_dict() for k, v in self.knowledge_store.items()},
                'keyword_index': dict(self.keyword_index),
                'metadata': {
                    'version': '1.0',
                    'entry_count': len(self.knowledge_store)
                }
            }
            
            # Write to temporary file first, then rename for atomic operation
            temp_path = self.storage_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            os.rename(temp_path, self.storage_path)
            logger.info(f"Saved {len(self.knowledge_store)} knowledge entries to {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save knowledge store: {e}", exc_info=True)
            return False
    
    def _load_from_disk(self) -> bool:
        """Load knowledge store from disk."""
        try:
            if not os.path.exists(self.storage_path):
                logger.info("No existing knowledge store found, starting fresh")
                return True
            
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load entries
            entries_data = data.get('entries', {})
            for entry_id, entry_dict in entries_data.items():
                entry = KnowledgeEntry.from_dict(entry_dict)
                self.knowledge_store[entry_id] = entry
            
            # Rebuild keyword index
            self.keyword_index = defaultdict(list)
            for entry_id, entry in self.knowledge_store.items():
                for keyword in entry.keywords:
                    self.keyword_index[keyword.lower()].append(entry_id)
            
            logger.info(f"Loaded {len(self.knowledge_store)} knowledge entries from disk")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load knowledge store: {e}", exc_info=True)
            # Continue with empty store rather than failing
            self.knowledge_store = {}
            self.keyword_index = defaultdict(list)
            return False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using simple heuristics."""
        import re
        
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
        
        # Filter out common stop words and keep meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Keep words that are 3+ characters and not stop words
        keywords = [w for w in words if len(w) >= 3 and w not in stop_words]
        
        # Keep only unique keywords, maintain order
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        return unique_keywords[:20]  # Limit to 20 keywords to avoid noise
    
    def _calculate_relevance_score(
        self, 
        entry: KnowledgeEntry, 
        query_lower: str, 
        query_keywords: List[str]
    ) -> float:
        """Calculate relevance score between query and knowledge entry."""
        score = 0.0
        
        # Exact text match (highest score)
        if query_lower in entry.text.lower():
            score += 1.0
        
        # Keyword matching
        entry_keywords_lower = [k.lower() for k in entry.keywords]
        keyword_matches = 0
        for query_keyword in query_keywords:
            if query_keyword.lower() in entry_keywords_lower:
                keyword_matches += 1
        
        if query_keywords:
            keyword_score = keyword_matches / len(query_keywords)
            score += keyword_score * 0.8
        
        # Word overlap scoring
        query_words = set(query_lower.split())
        entry_words = set(entry.text.lower().split())
        word_overlap = len(query_words.intersection(entry_words))
        
        if query_words:
            word_score = word_overlap / len(query_words)
            score += word_score * 0.6
        
        # Boost score for recent entries (slight recency bias)
        import time
        age_days = (time.time() - entry.timestamp) / (24 * 3600)
        if age_days < 7:  # Boost entries from last week
            score += 0.1
        
        return score
    
    def _manage_memory(self):
        """Manage memory by removing old entries if we exceed the limit."""
        if len(self.knowledge_store) <= self.max_entries:
            return
        
        # Sort entries by timestamp (oldest first)
        entries_by_time = sorted(
            self.knowledge_store.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Remove oldest entries
        entries_to_remove = len(self.knowledge_store) - self.max_entries
        for i in range(entries_to_remove):
            entry_id, entry = entries_by_time[i]
            
            # Remove from knowledge store
            del self.knowledge_store[entry_id]
            
            # Remove from keyword index
            for keyword in entry.keywords:
                keyword_lower = keyword.lower()
                if entry_id in self.keyword_index[keyword_lower]:
                    self.keyword_index[keyword_lower].remove(entry_id)
                    # Clean up empty keyword entries
                    if not self.keyword_index[keyword_lower]:
                        del self.keyword_index[keyword_lower]
        
        logger.info(f"Removed {entries_to_remove} old entries to manage memory")
    
    @property
    def ntotal(self) -> int:
        """Return total number of knowledge entries (for compatibility)."""
        return len(self.knowledge_store)


class EnhancedRAGManager:
    """
    Enhanced RAG Manager that tries to use FAISS if available, falls back to SimpleRAGManager.
    
    This provides the best of both worlds - high performance with ML libraries when available,
    and reliable fallback functionality in any environment.
    """
    
    def __init__(self, *args, **kwargs):
        self._use_faiss = False
        self._manager = None
        
        try:
            # Try to use the original FAISS-based implementation
            import numpy as np
            import faiss
            from sentence_transformers import SentenceTransformer
            
            # If we get here, dependencies are available
            from server.agent.rag_manager_original import OriginalRAGManager
            self._manager = OriginalRAGManager(*args, **kwargs)
            self._use_faiss = True
            logger.info("Using FAISS-based RAG manager")
            
        except ImportError as e:
            # Fallback to simple implementation
            logger.info(f"FAISS dependencies not available ({e}), using fallback RAG manager")
            self._manager = SimpleRAGManager()
            self._use_faiss = False
    
    def add_knowledge(self, text: str, source: str) -> None:
        """Add knowledge to the store."""
        if self._use_faiss:
            self._manager.add_knowledge(text, source)
        else:
            self._manager.add_knowledge(text, source)
    
    def retrieve_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge."""
        return self._manager.retrieve_knowledge(query, top_k)
    
    def save(self) -> None:
        """Save the knowledge store."""
        if self._use_faiss:
            self._manager.save()
        else:
            self._manager.save()
    
    @property
    def ntotal(self) -> int:
        """Return total number of entries."""
        if hasattr(self._manager, 'ntotal'):
            return self._manager.ntotal
        return len(getattr(self._manager, 'knowledge_store', {}))


# Create the enhanced manager instance
rag_manager = EnhancedRAGManager()