"""
Intelligent Context Manager for the Anton agent system.

This component addresses critical context management weaknesses by:
1. Intelligent context pruning to prevent overflow
2. Prioritization and summarization of information
3. Retrieval-augmented context management
4. Semantic importance scoring
5. Adaptive context window management
"""
import json
import logging
import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import re
import math

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context information."""
    SYSTEM_PROMPT = "system_prompt"
    TASK_DESCRIPTION = "task_description"
    CONVERSATION = "conversation"
    FILE_CONTENT = "file_content"
    TOOL_OUTPUT = "tool_output"
    MEMORY = "memory"
    ERROR_INFO = "error_info"
    PROGRESS_UPDATE = "progress_update"


class ContextPriority(Enum):
    """Priority levels for context information."""
    CRITICAL = 5  # Must be preserved (system prompts, current task)
    HIGH = 4      # Important recent information
    MEDIUM = 3    # Relevant but can be summarized
    LOW = 2       # Background information
    MINIMAL = 1   # Can be removed if needed


@dataclass
class ContextItem:
    """A single piece of context information with metadata."""
    content: str
    context_type: ContextType
    priority: ContextPriority
    timestamp: float
    importance_score: float = 0.0
    token_count: int = 0
    source: str = ""
    dependencies: Set[str] = field(default_factory=set)
    summary: Optional[str] = None
    keywords: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = self._estimate_token_count()
        if not self.keywords:
            self.keywords = self._extract_keywords()
    
    def _estimate_token_count(self) -> int:
        """Estimate token count for the content."""
        # Rough approximation: 1 token ≈ 4 characters
        return len(self.content) // 4
    
    def _extract_keywords(self) -> Set[str]:
        """Extract keywords from content for relevance scoring."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', self.content.lower())
        # Filter out common words and keep meaningful terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = {word for word in words if len(word) > 3 and word not in stop_words}
        return keywords


@dataclass
class ContextWindow:
    """Represents the current context window with size limits."""
    max_tokens: int = 4000  # Conservative limit for context window
    current_tokens: int = 0
    items: List[ContextItem] = field(default_factory=list)
    
    def has_capacity(self, additional_tokens: int) -> bool:
        """Check if window has capacity for additional tokens."""
        return (self.current_tokens + additional_tokens) <= self.max_tokens
    
    def get_utilization(self) -> float:
        """Get current utilization as percentage."""
        return (self.current_tokens / self.max_tokens) * 100 if self.max_tokens > 0 else 0


@dataclass
class ContextSummary:
    """Summary of context information for efficient storage."""
    original_content: str
    summary_text: str
    key_points: List[str]
    preserved_keywords: Set[str]
    compression_ratio: float
    created_at: float


class IntelligentContextManager:
    """
    Intelligent context manager with advanced pruning and optimization.
    
    Features:
    - Semantic importance scoring for context prioritization
    - Intelligent summarization of less critical information
    - Adaptive context window management
    - Retrieval-augmented context injection
    - Performance-optimized context operations
    """
    
    def __init__(self, max_context_tokens: int = 4000):
        self.context_window = ContextWindow(max_tokens=max_context_tokens)
        
        # Context storage and management
        self.context_items: List[ContextItem] = []
        self.summaries: Dict[str, ContextSummary] = {}
        
        # Importance scoring weights
        self.importance_weights = {
            ContextType.SYSTEM_PROMPT: 1.0,
            ContextType.TASK_DESCRIPTION: 0.9,
            ContextType.CONVERSATION: 0.7,
            ContextType.FILE_CONTENT: 0.6,
            ContextType.TOOL_OUTPUT: 0.5,
            ContextType.MEMORY: 0.8,
            ContextType.ERROR_INFO: 0.8,
            ContextType.PROGRESS_UPDATE: 0.4
        }
        
        # Keyword tracking for relevance
        self.global_keywords = defaultdict(int)
        self.keyword_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Performance metrics
        self.pruning_history = deque(maxlen=50)
        self.context_access_patterns = defaultdict(int)
        
        logger.info(f"Intelligent Context Manager initialized with {max_context_tokens} token limit")
    
    def add_context(
        self,
        content: str,
        context_type: ContextType,
        priority: Optional[ContextPriority] = None,
        source: str = "",
        dependencies: Optional[Set[str]] = None
    ) -> str:
        """
        Add new context with intelligent management.
        
        Returns:
            Context item ID for future reference
        """
        # Auto-assign priority if not provided
        if priority is None:
            priority = self._auto_assign_priority(context_type, content)
        
        # Create context item
        item = ContextItem(
            content=content,
            context_type=context_type,
            priority=priority,
            timestamp=time.time(),
            source=source,
            dependencies=dependencies or set()
        )
        
        # Calculate importance score
        item.importance_score = self._calculate_importance_score(item)
        
        # Generate unique ID
        item_id = self._generate_item_id(item)
        
        # Update global keyword tracking
        self._update_keyword_tracking(item.keywords)
        
        # Check if we need to prune before adding
        if not self.context_window.has_capacity(item.token_count):
            self._intelligent_prune(target_tokens=item.token_count)
        
        # Add to context
        self.context_items.append(item)
        self.context_window.items.append(item)
        self.context_window.current_tokens += item.token_count
        
        logger.debug(f"Added context item {item_id[:8]}... ({item.token_count} tokens, priority: {priority.name})")
        return item_id
    
    def _auto_assign_priority(self, context_type: ContextType, content: str) -> ContextPriority:
        """Automatically assign priority based on context type and content."""
        if context_type in [ContextType.SYSTEM_PROMPT, ContextType.TASK_DESCRIPTION]:
            return ContextPriority.CRITICAL
        elif context_type in [ContextType.ERROR_INFO, ContextType.MEMORY]:
            return ContextPriority.HIGH
        elif "error" in content.lower() or "failed" in content.lower():
            return ContextPriority.HIGH
        elif len(content) > 1000:  # Long content might be important
            return ContextPriority.MEDIUM
        else:
            return ContextPriority.LOW
    
    def _calculate_importance_score(self, item: ContextItem) -> float:
        """Calculate semantic importance score for context item."""
        score = 0.0
        
        # Base score from type and priority
        type_weight = self.importance_weights.get(item.context_type, 0.5)
        priority_weight = item.priority.value / 5.0
        score += (type_weight + priority_weight) * 0.4
        
        # Recency bonus (more recent = more important)
        age_hours = (time.time() - item.timestamp) / 3600
        recency_score = max(0, 1 - (age_hours / 24))  # Decay over 24 hours
        score += recency_score * 0.2
        
        # Keyword relevance (match with global keywords)
        keyword_relevance = self._calculate_keyword_relevance(item.keywords)
        score += keyword_relevance * 0.2
        
        # Content uniqueness (penalize repetitive content)
        uniqueness_score = self._calculate_content_uniqueness(item.content)
        score += uniqueness_score * 0.1
        
        # Length bonus for substantial content
        length_score = min(1.0, len(item.content) / 500)
        score += length_score * 0.1
        
        return min(1.0, score)
    
    def _calculate_keyword_relevance(self, keywords: Set[str]) -> float:
        """Calculate relevance based on keyword frequency."""
        if not keywords or not self.global_keywords:
            return 0.0
        
        relevance = 0.0
        for keyword in keywords:
            frequency = self.global_keywords.get(keyword, 0)
            relevance += min(1.0, frequency / 10)  # Normalize frequency
        
        return min(1.0, relevance / len(keywords))
    
    def _calculate_content_uniqueness(self, content: str) -> float:
        """Calculate uniqueness score to avoid redundant information."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Check against existing content
        for item in self.context_items[-10:]:  # Check last 10 items
            existing_hash = hashlib.md5(item.content.encode()).hexdigest()
            if content_hash == existing_hash:
                return 0.0  # Duplicate content
            
            # Check for high similarity
            similarity = self._calculate_text_similarity(content, item.content)
            if similarity > 0.8:
                return 0.2  # Highly similar content
        
        return 1.0  # Unique content
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _update_keyword_tracking(self, keywords: Set[str]) -> None:
        """Update global keyword tracking for relevance scoring."""
        for keyword in keywords:
            self.global_keywords[keyword] += 1
            
            # Update co-occurrence matrix
            for other_keyword in keywords:
                if keyword != other_keyword:
                    self.keyword_cooccurrence[keyword][other_keyword] += 1
    
    def _generate_item_id(self, item: ContextItem) -> str:
        """Generate unique ID for context item."""
        content_snippet = item.content[:50]
        timestamp_str = str(item.timestamp)
        combined = f"{content_snippet}_{timestamp_str}_{item.context_type.value}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _intelligent_prune(self, target_tokens: int) -> None:
        """Intelligently prune context to make space for new content."""
        logger.info(f"Starting intelligent pruning to free {target_tokens} tokens")
        
        # Calculate tokens to free (add buffer)
        tokens_to_free = target_tokens + (self.context_window.max_tokens * 0.1)  # 10% buffer
        tokens_freed = 0
        pruned_items = []
        
        # Sort items by removal priority (lower scores first)
        removal_candidates = []
        for item in self.context_window.items:
            removal_score = self._calculate_removal_score(item)
            removal_candidates.append((removal_score, item))
        
        removal_candidates.sort(key=lambda x: x[0])  # Sort by removal score
        
        # Remove items until we have enough space
        for removal_score, item in removal_candidates:
            if tokens_freed >= tokens_to_free:
                break
            
            # Skip critical items
            if item.priority == ContextPriority.CRITICAL:
                continue
            
            # Try summarization first for important content
            if item.priority in [ContextPriority.HIGH, ContextPriority.MEDIUM] and len(item.content) > 200:
                summary = self._create_summary(item)
                if summary:
                    # Replace with summary
                    summarized_item = ContextItem(
                        content=summary.summary_text,
                        context_type=item.context_type,
                        priority=ContextPriority.LOW,  # Reduced priority for summaries
                        timestamp=item.timestamp,
                        source=f"summary_of_{item.source}",
                        dependencies=item.dependencies
                    )
                    
                    # Replace in context window
                    index = self.context_window.items.index(item)
                    self.context_window.items[index] = summarized_item
                    self.context_window.current_tokens -= item.token_count
                    self.context_window.current_tokens += summarized_item.token_count
                    
                    tokens_freed += (item.token_count - summarized_item.token_count)
                    pruned_items.append(("summarized", item, summarized_item))
                    continue
            
            # Remove the item
            self.context_window.items.remove(item)
            self.context_window.current_tokens -= item.token_count
            tokens_freed += item.token_count
            pruned_items.append(("removed", item, None))
        
        # Record pruning operation
        self.pruning_history.append({
            "timestamp": time.time(),
            "tokens_freed": tokens_freed,
            "items_pruned": len(pruned_items),
            "target_tokens": target_tokens
        })
        
        logger.info(f"Pruning completed: freed {tokens_freed} tokens, processed {len(pruned_items)} items")
    
    def _calculate_removal_score(self, item: ContextItem) -> float:
        """Calculate score for item removal (lower = remove first)."""
        # Invert importance score (high importance = low removal score)
        base_score = 1.0 - item.importance_score
        
        # Age factor (older items easier to remove)
        age_hours = (time.time() - item.timestamp) / 3600
        age_factor = min(1.0, age_hours / 12)  # Full age factor after 12 hours
        
        # Priority factor
        priority_factor = (5 - item.priority.value) / 4  # Invert priority
        
        # Content type factor
        type_penalty = {
            ContextType.SYSTEM_PROMPT: 0.0,  # Never remove
            ContextType.TASK_DESCRIPTION: 0.1,  # Rarely remove
            ContextType.ERROR_INFO: 0.2,
            ContextType.MEMORY: 0.3,
            ContextType.CONVERSATION: 0.4,
            ContextType.FILE_CONTENT: 0.5,
            ContextType.TOOL_OUTPUT: 0.6,
            ContextType.PROGRESS_UPDATE: 0.8  # Remove first
        }.get(item.context_type, 0.5)
        
        return base_score + age_factor * 0.3 + priority_factor * 0.4 + type_penalty * 0.3
    
    def _create_summary(self, item: ContextItem) -> Optional[ContextSummary]:
        """Create a summary of context item content."""
        try:
            content = item.content
            
            # Simple extractive summarization
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) <= 2:
                return None  # Too short to summarize
            
            # Score sentences by keyword frequency and position
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = 0.0
                
                # Position score (earlier sentences more important)
                position_score = 1.0 - (i / len(sentences))
                score += position_score * 0.3
                
                # Keyword score
                sentence_keywords = set(re.findall(r'\b\w+\b', sentence.lower()))
                keyword_score = len(sentence_keywords.intersection(item.keywords))
                score += keyword_score * 0.4
                
                # Length score (prefer substantial sentences)
                length_score = min(1.0, len(sentence) / 100)
                score += length_score * 0.3
                
                sentence_scores.append((score, sentence))
            
            # Select top sentences for summary
            sentence_scores.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [sent for score, sent in sentence_scores[:2]]  # Top 2 sentences
            
            summary_text = ". ".join(top_sentences) + "."
            
            # Extract key points
            key_points = []
            for sentence in top_sentences:
                if any(keyword in sentence.lower() for keyword in ["discovered", "found", "error", "solution"]):
                    key_points.append(sentence.strip())
            
            compression_ratio = len(summary_text) / len(content)
            
            summary = ContextSummary(
                original_content=content,
                summary_text=summary_text,
                key_points=key_points,
                preserved_keywords=item.keywords,
                compression_ratio=compression_ratio,
                created_at=time.time()
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create summary: {e}")
            return None
    
    def get_context_for_prompt(
        self,
        context_types: Optional[List[ContextType]] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Retrieve optimized context for prompt generation.
        
        Args:
            context_types: Specific types of context to include
            max_tokens: Override default token limit
            
        Returns:
            Formatted context string
        """
        if max_tokens is None:
            max_tokens = self.context_window.max_tokens
        
        # Filter items by type if specified
        relevant_items = self.context_window.items
        if context_types:
            relevant_items = [item for item in relevant_items if item.context_type in context_types]
        
        # Sort by importance and recency
        relevant_items.sort(key=lambda x: (x.importance_score, x.timestamp), reverse=True)
        
        # Build context within token limit
        context_parts = []
        used_tokens = 0
        
        for item in relevant_items:
            if used_tokens + item.token_count > max_tokens:
                # Try to include partial content for important items
                if item.priority in [ContextPriority.CRITICAL, ContextPriority.HIGH]:
                    remaining_tokens = max_tokens - used_tokens
                    if remaining_tokens > 50:  # Minimum useful content
                        partial_content = item.content[:remaining_tokens * 4]  # Rough token->char conversion
                        context_parts.append(f"[{item.context_type.value}]: {partial_content}...")
                        break
                else:
                    break
            
            context_parts.append(f"[{item.context_type.value}]: {item.content}")
            used_tokens += item.token_count
        
        context_text = "\n\n".join(context_parts)
        
        # Update access patterns
        for item in relevant_items[:len(context_parts)]:
            self.context_access_patterns[item.context_type.value] += 1
        
        logger.debug(f"Generated context with {used_tokens} tokens from {len(context_parts)} items")
        return context_text
    
    def get_relevant_memories(self, query: str, max_items: int = 3) -> List[ContextItem]:
        """
        Retrieve relevant memories based on query using keyword matching.
        """
        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
        memory_items = [item for item in self.context_items if item.context_type == ContextType.MEMORY]
        
        # Score memories by relevance
        scored_memories = []
        for item in memory_items:
            relevance_score = self._calculate_query_relevance(item, query_keywords)
            if relevance_score > 0.1:  # Minimum relevance threshold
                scored_memories.append((relevance_score, item))
        
        # Sort by relevance and return top items
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_memories[:max_items]]
    
    def _calculate_query_relevance(self, item: ContextItem, query_keywords: Set[str]) -> float:
        """Calculate relevance of context item to query keywords."""
        if not query_keywords or not item.keywords:
            return 0.0
        
        # Keyword overlap
        keyword_overlap = len(query_keywords.intersection(item.keywords))
        overlap_score = keyword_overlap / len(query_keywords)
        
        # Importance boost
        importance_boost = item.importance_score * 0.3
        
        # Recency boost
        age_hours = (time.time() - item.timestamp) / 3600
        recency_boost = max(0, 1 - (age_hours / 48)) * 0.2  # Decay over 48 hours
        
        return overlap_score + importance_boost + recency_boost
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get comprehensive context management statistics."""
        stats = {
            "context_window": {
                "max_tokens": self.context_window.max_tokens,
                "current_tokens": self.context_window.current_tokens,
                "utilization_percent": self.context_window.get_utilization(),
                "items_count": len(self.context_window.items)
            },
            "total_items": len(self.context_items),
            "items_by_type": defaultdict(int),
            "items_by_priority": defaultdict(int),
            "average_importance": 0.0,
            "pruning_operations": len(self.pruning_history),
            "top_keywords": [],
            "summaries_created": len(self.summaries)
        }
        
        # Aggregate statistics
        total_importance = 0.0
        for item in self.context_items:
            stats["items_by_type"][item.context_type.value] += 1
            stats["items_by_priority"][item.priority.name] += 1
            total_importance += item.importance_score
        
        if self.context_items:
            stats["average_importance"] = total_importance / len(self.context_items)
        
        # Top keywords
        sorted_keywords = sorted(self.global_keywords.items(), key=lambda x: x[1], reverse=True)
        stats["top_keywords"] = sorted_keywords[:10]
        
        return stats
    
    def optimize_context_management(self) -> None:
        """Optimize context management based on usage patterns."""
        logger.info("Optimizing context management...")
        
        # Adjust importance weights based on access patterns
        total_accesses = sum(self.context_access_patterns.values())
        if total_accesses > 0:
            for context_type, access_count in self.context_access_patterns.items():
                access_ratio = access_count / total_accesses
                # Boost importance weights for frequently accessed types
                if access_ratio > 0.3:  # High access
                    context_enum = ContextType(context_type)
                    self.importance_weights[context_enum] = min(1.0, self.importance_weights[context_enum] * 1.1)
        
        # Remove very old, low-importance items
        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days ago
        items_to_remove = []
        
        for item in self.context_items:
            if (item.timestamp < cutoff_time and 
                item.importance_score < 0.3 and 
                item.priority in [ContextPriority.LOW, ContextPriority.MINIMAL]):
                items_to_remove.append(item)
        
        for item in items_to_remove:
            self.context_items.remove(item)
            if item in self.context_window.items:
                self.context_window.items.remove(item)
                self.context_window.current_tokens -= item.token_count
        
        logger.info(f"Context optimization completed: removed {len(items_to_remove)} old items")
    
    def export_context_state(self) -> Dict[str, Any]:
        """Export current context state for debugging or persistence."""
        return {
            "context_window": {
                "max_tokens": self.context_window.max_tokens,
                "current_tokens": self.context_window.current_tokens,
                "items_count": len(self.context_window.items)
            },
            "total_items": len(self.context_items),
            "importance_weights": {k.value: v for k, v in self.importance_weights.items()},
            "recent_pruning": list(self.pruning_history)[-5:],  # Last 5 operations
            "keyword_stats": dict(list(self.global_keywords.items())[:20])  # Top 20 keywords
        }


# Create singleton instance
intelligent_context_manager = IntelligentContextManager()