"""
Centralized knowledge management system that tracks context across planner, doer, and evaluator components.
Provides persistent storage, context prioritization, and knowledge transfer capabilities.
Enhanced with episodic memory for recording and retrieving past run experiences.
Enhanced with semantic memory for persistent domain-scoped facts and knowledge.
"""
from pathlib import Path
import time
import logging
from typing import Dict, Set, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from server.agent.concept_graph import load_pack, rag_topk_nodes, expand_nodes, format_context
from server.agent.learning_loop import learning_loop
from server.agent.pack_builder import load_centroids
from server.agent.rag_manager import rag_manager
from server.agent.episodic_memory import episodic_memory, EpisodicEntry
from server.agent.semantic_memory import semantic_memory_store, SemanticFact

# Configure logger for knowledge store
logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of context information"""
    FILE_CONTENT = "file_content"
    DIRECTORY_LISTING = "directory_listing"
    TOOL_EXECUTION = "tool_execution"
    PLANNER_INSIGHT = "planner_insight"
    EVALUATOR_FEEDBACK = "evaluator_feedback"
    TASK_PROGRESS = "task_progress"
    MESSAGE = "message"
    THOUGHT = "thought"
    ACTION = "action"


class ImportanceLevel(Enum):
    """Importance levels for context prioritization"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ContextItem:
    """Represents a single piece of context with metadata"""
    content: str
    context_type: ContextType
    importance: ImportanceLevel
    timestamp: float
    source: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class KnowledgeStore:
    """
    Centralized knowledge management that tracks and persists context across all agent components.
    Integrates with existing RAG manager for persistent storage and episodic memory for run-based experiences.
    """
    
    def __init__(self):
        # In-memory context tracking (similar to existing context_store)
        self.explored_files: Set[str] = set()
        self.code_content: Dict[str, str] = {}
        self.task_progress: List[str] = []
        # For deduplicating messages (avoid re-adding identical history across requests)
        self._message_hashes: Set[str] = set()

        # Enhanced context management
        self.context_items: List[ContextItem] = []
        self.importance_weights = {
            ImportanceLevel.LOW: 1.0,
            ImportanceLevel.MEDIUM: 2.0,
            ImportanceLevel.HIGH: 4.0,
            ImportanceLevel.CRITICAL: 8.0
        }

        # Conversation state management (replaces ConversationState)
        self.messages: List[Dict[str, str]] = []
        self.tool_outputs: Dict[str, Any] = {}
        self.start_time = time.time()
        self.is_complete = False
        self.final_response = ""
        
        # Episodic memory integration
        self.current_domain = None
        self.current_run_id = None
    
    def add_context(
        self, 
        content: str, 
        context_type: ContextType,
        importance: ImportanceLevel = ImportanceLevel.MEDIUM,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new context item with importance weighting"""
        context_item = ContextItem(
            content=content,
            context_type=context_type,
            importance=importance,
            timestamp=time.time(),
            source=source,
            metadata=metadata or {}
        )
        
        self.context_items.append(context_item)
        
        # Update legacy fields for backward compatibility
        if context_type == ContextType.FILE_CONTENT:
            file_path = metadata.get("file_path") if metadata else None
            if file_path:
                self.explored_files.add(file_path)
                self.code_content[file_path] = content
        elif context_type == ContextType.TASK_PROGRESS:
            self.task_progress.append(content)
        
        # Persist high importance items immediately to RAG
        if importance in [ImportanceLevel.HIGH, ImportanceLevel.CRITICAL]:
            self._persist_to_rag(context_item)
    
    def update_from_tool_execution(self, tool_name: str, tool_args: dict, result: str) -> None:
        """Update knowledge store from tool execution results"""
        if tool_name == "read_file":
            file_path = tool_args.get("file_path")
            if file_path:
                # Determine importance based on file type and size
                importance = self._determine_file_importance(file_path, result)
                self.add_context(
                    content=result[:10000] if len(result) > 10000 else result,
                    context_type=ContextType.FILE_CONTENT,
                    importance=importance,
                    source=f"tool_execution_{tool_name}",
                    metadata={"file_path": file_path, "full_size": len(result)}
                )
        
        elif tool_name == "list_directory":
            path = tool_args.get("path", ".")
            self.explored_files.add(path)  # Add directory to explored files
            self.add_context(
                content=result,
                context_type=ContextType.DIRECTORY_LISTING,
                importance=ImportanceLevel.LOW,
                source=f"tool_execution_{tool_name}",
                metadata={"directory_path": path}
            )
            
        else:
            # Generic tool execution tracking
            self.add_context(
                content=f"Tool {tool_name} executed with result: {result[:500]}",
                context_type=ContextType.TOOL_EXECUTION,
                importance=ImportanceLevel.LOW,
                source=f"tool_execution_{tool_name}",
                metadata={"tool_args": tool_args}
            )
    
    def add_planner_insight(self, insight: str, importance: ImportanceLevel = ImportanceLevel.MEDIUM) -> None:
        """Add insights from planner analysis"""
        self.add_context(
            content=insight,
            context_type=ContextType.PLANNER_INSIGHT,
            importance=importance,
            source="planner"
        )
    
    def add_evaluator_feedback(self, feedback: str, importance: ImportanceLevel = ImportanceLevel.HIGH) -> None:
        """Add evaluator feedback to knowledge store"""
        self.add_context(
            content=feedback,
            context_type=ContextType.EVALUATOR_FEEDBACK,
            importance=importance,
            source="evaluator"
        )
        
        # Evaluator feedback is always important, so persist it
        self._persist_to_rag(self.context_items[-1])
    
    def get_prioritized_context(self, max_items: int = 10, context_types: Optional[List[ContextType]] = None) -> List[ContextItem]:
        """Get context items prioritized by importance and recency"""
        filtered_items = self.context_items
        
        if context_types:
            filtered_items = [item for item in filtered_items if item.context_type in context_types]
        
        # Sort by importance weight and recency (more recent = higher score)
        current_time = time.time()
        scored_items = []
        
        for item in filtered_items:
            # Calculate recency score (more recent = higher score, decay over time)
            age_hours = (current_time - item.timestamp) / 3600
            recency_score = max(0.1, 1.0 / (1.0 + age_hours * 0.1))  # Decay over time
            
            # Combine importance weight with recency
            total_score = self.importance_weights[item.importance] * recency_score
            scored_items.append((total_score, item))
        
        # Sort by score (highest first) and return top items
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items[:max_items]]
    
    def build_context_summary(self) -> str:
        """Build a comprehensive context summary for planner"""
        summary_parts = []
        
        # Get prioritized context
        priority_items = self.get_prioritized_context(max_items=15)
        
        if self.explored_files:
            summary_parts.append("Explored files: " + ", ".join(list(self.explored_files)[:10]))
        
        if self.code_content:
            summary_parts.append("Retrieved file contents:")
            for filename in list(self.code_content.keys())[:5]:
                summary_parts.append(f"- {filename}")
        
        if self.task_progress:
            summary_parts.append("Progress so far:")
            for step in self.task_progress[-5:]:  # Last 5 progress items
                summary_parts.append(f"- {step}")
        
        # Add high-priority insights
        high_priority = [item for item in priority_items if item.importance in [ImportanceLevel.HIGH, ImportanceLevel.CRITICAL]]
        if high_priority:
            summary_parts.append("\nKey insights:")
            for item in high_priority[:3]:
                summary_parts.append(f"- [{item.context_type.value}] {item.content[:200]}...")
        
        return "\n".join(summary_parts)
    
    def get_legacy_context_store(self) -> dict:
        """Return legacy context store format for backward compatibility"""
        return {
            "explored_files": self.explored_files,
            "code_content": self.code_content,
            "task_progress": self.task_progress
        }
    
    def _determine_file_importance(self, file_path: str, content: str) -> ImportanceLevel:
        """Determine importance of a file based on path and content characteristics"""
        # Configuration files and main modules are high importance
        important_patterns = [
            "config", "main", "app", "server", "client", 
            "requirements", "setup", "package", "Dockerfile"
        ]
        
        if any(pattern in file_path.lower() for pattern in important_patterns):
            return ImportanceLevel.HIGH
            
        # Large files or files with many functions/classes are medium importance
        if len(content) > 5000 or content.count("def ") > 5 or content.count("class ") > 2:
            return ImportanceLevel.MEDIUM
            
        return ImportanceLevel.LOW
    
    def _persist_to_rag(self, context_item: ContextItem) -> None:
        """Persist important context to RAG for long-term memory"""
        try:
            # Format context for RAG storage
            rag_content = f"[{context_item.context_type.value}] {context_item.content}"
            
            # Add to RAG knowledge base
            rag_manager.add_knowledge(
                text=rag_content,
                source=f"{context_item.source}_{context_item.timestamp}"
            )
            logger.debug(f"Successfully persisted context to RAG: {context_item.context_type.value}")
        except Exception as e:
            # Log the error but don't fail - RAG persistence is not critical
            logger.error(f"Failed to persist context to RAG: {e}", exc_info=True)

    def build_domain_knowledge_context(
        self,
        query: str,
        pack_dir: str,
        topk: int = 5,
        expand_radius: int = 1,
        max_nodes: int = 8,
        max_examples_per_node: int = 1
    ) -> str:
        """
        Retrieve top concepts for the query from the given pack, expand by prerequisites,
        and return a compact context string to feed the LLM.
        """
        try:
            adj, nodes_by_id = load_pack(pack_dir)
            pack_name = Path(pack_dir).name  # e.g., 'calc.v1'
            # 1) RAG top-k
            node_ids = rag_topk_nodes(rag_manager, query, pack_name, topk=topk)
            if not node_ids:
                return ""
            # 2) Graph expand
            expanded = expand_nodes(node_ids, adj, edge_types=("depends_on",), radius=expand_radius)
            # Keep retrieved nodes first, then add prereqs (dedup while preserving order)
            keep_order = dict.fromkeys(node_ids + [x for x in expanded if x not in node_ids])
            ordered_ids = list(keep_order.keys())
            # 3) Format
            context = format_context(nodes_by_id, ordered_ids, max_nodes=max_nodes, max_examples_per_node=max_examples_per_node)
            return context
        except Exception as e:
            logger.error(f"Failed to build domain knowledge context: {e}", exc_info=True)
            return ""
        
    def select_pack_by_embedding(self, prompt: str, fallback="packs/anton_repo.v1") -> str:
        # Quick domain check for critical cases
        prompt_lower = prompt.lower()
        
        # Music theory queries should use music pack
        if any(word in prompt_lower for word in ["music", "song", "chord", "harmony", "melody", "musical", "crazy train", "theory"]):
            music_pack = "learning/packs/music_theory.v1"
            if Path(music_pack).exists():
                logger.info(f'Domain override: using music_theory.v1 for music query')
                return music_pack
                
        # Math queries should use calc pack  
        if any(word in prompt_lower for word in ["calculus", "derivative", "integral", "math", "equation"]):
            calc_pack = "learning/packs/calc.v1"
            if Path(calc_pack).exists():
                logger.info(f'Domain override: using calc.v1 for math query')
                return calc_pack
        
        # Fall back to embedding selection
        centroids = load_centroids()
        logger.info('using embedding-based selection')
        if not centroids:
            logger.info('returning fallback pack, no centroids found')
            return fallback
        q = rag_manager.model.encode([prompt])[0]  # (dim,)
        best_pack, best_sim = None, -1.0
        for pack_dir, data in centroids.items():
            c = np.array(data["centroid"], dtype=np.float32)
            sim = float(np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c) + 1e-9))
            if sim > best_sim:
                best_pack, best_sim = pack_dir, sim
        # add a floor so random chit-chat doesn’t select a pack
        logger.info('selected pack: ' + str(best_pack))
        return best_pack if best_sim >= 0.35 else fallback
    
    def query_relevant_knowledge(self, query: str, max_results: int = 5) -> List[str]:
        """Query RAG for relevant past knowledge"""
        try:
            # Query both RAG and current context
            rag_results = rag_manager.retrieve_knowledge(query, top_k=max_results)
            
            # Handle different return types from RAG manager (dict or string)
            rag_texts = []
            for result in rag_results:
                if isinstance(result, dict):
                    rag_texts.append(result.get('text', ''))
                elif isinstance(result, str):
                    rag_texts.append(result)
                else:
                    rag_texts.append(str(result))
            
            # Also search current context items
            query_lower = query.lower()
            relevant_current = [
                item.content for item in self.context_items 
                if query_lower in item.content.lower()
            ][:max_results//2]
            
            return rag_texts + relevant_current
        except Exception as e:
            logger.error(f"Failed to query knowledge: {e}", exc_info=True)
            return []

    def start_learning_task(self, user_prompt: str):
        """Start tracking a task in the learning loop."""
        learning_loop.start_task(user_prompt)

    def add_learning_action(self, action_type: str, details: dict):
        """Record an action in the current learning task."""
        learning_loop.record_action(action_type, details)

    def complete_learning_task(self, success: bool, feedback: str):
        """Complete the current learning task."""
        return learning_loop.complete_task(success, feedback)

    def get_relevant_past_experiences(self, prompt: str):
        """Get relevant past learnings for the current task."""
        return learning_loop.get_relevant_learnings(prompt)
    
    # Conversation state management methods (replacing ConversationState)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation"""
        message = {"role": role, "content": content}
        if metadata:
            message.update(metadata)
        key = f"{role}:{content}"
        if key in self._message_hashes:
            return  # Already stored
        self._message_hashes.add(key)
        self.messages.append(message)
        
        # Also track as context item for advanced prioritization
        self.add_context(
            content=content,
            context_type=ContextType.MESSAGE,
            importance=ImportanceLevel.MEDIUM,
            source=f"message_{role}",
            metadata={"role": role, **(metadata or {})}
        )
    
    def add_tool_output(self, tool_name: str, output: Any, metadata: Optional[Dict] = None):
        """Store tool execution results"""
        self.tool_outputs[tool_name] = {
            "output": output,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Track as context item with high importance
        self.add_context(
            content=f"Tool {tool_name}: {str(output)[:500]}",
            context_type=ContextType.TOOL_EXECUTION,
            importance=ImportanceLevel.HIGH,
            source=f"tool_{tool_name}",
            metadata={"tool_name": tool_name, "tool_args": metadata.get("args", {}) if metadata else {}}
        )
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM consumption"""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]
    
    def mark_complete(self, final_response: str):
        """Mark conversation as complete"""
        self.is_complete = True
        self.final_response = final_response
        self.add_context(
            content=final_response,
            context_type=ContextType.MESSAGE,
            importance=ImportanceLevel.HIGH,
            source="final_response",
            metadata={"final": True}
        )
    
    def get_duration(self) -> float:
        """Get conversation duration in seconds"""
        return time.time() - self.start_time
    
    def reset_conversation(self, preserve_important_context: bool = True):
        """
        Reset conversation state for a new conversation.
        
        Args:
            preserve_important_context: If True, keeps CRITICAL and HIGH importance contexts
                                      from previous sessions. If False, starts completely fresh.
        """
        # Clear session-specific state
        self.messages = []
        self.tool_outputs = {}
        self.start_time = time.time()
        self.is_complete = False
        self.final_response = ""
        
        # Handle context isolation
        if preserve_important_context:
            # Keep only CRITICAL and HIGH importance contexts from previous sessions
            preserved_context = [
                item for item in self.context_items 
                if item.importance in [ImportanceLevel.CRITICAL, ImportanceLevel.HIGH]
            ]
            self.context_items = preserved_context
            logger.info(f"Reset conversation, preserved {len(preserved_context)} important context items")
        else:
            # Complete fresh start - clear all context
            self.context_items = []
            logger.info("Reset conversation with complete fresh start")
    
    def start_new_session(self):
        """Start a completely new session with fresh context"""
        self.reset_conversation(preserve_important_context=False)
    
    # ===============================
    # Episodic Memory Integration
    # ===============================
    
    def start_episodic_run(self, domain: str, parent_run_id: Optional[str] = None) -> str:
        """
        Start a new episodic memory run for the given domain.
        
        Args:
            domain: The domain context (e.g., 'chess', 'tool_dev', 'code_analysis')
            parent_run_id: Optional parent run ID for hierarchical tracking
            
        Returns:
            Generated run ID
        """
        run_id = episodic_memory.start_run(domain, parent_run_id)
        self.current_run_id = run_id
        self.current_domain = domain
        
        # Record the run start as an episode
        self.record_episode(
            role="system",
            summary=f"Started new run for domain '{domain}'",
            tags=["run_start", domain],
            entities={"domain": domain, "parent_run": parent_run_id},
            outcome={"status": "pass", "notes": "Run initialized successfully"}
        )
        
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
        Record an episodic memory entry for the current run.
        
        Args:
            role: The role performing this episode (assessor, researcher, planner, executor, evaluator)
            summary: Human-readable summary of what happened
            tags: List of searchable tags
            entities: Structured entities involved (files, tools, concepts, etc.)
            outcome: Outcome information {status: 'pass'|'fail'|'partial', metrics, notes}
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Episode ID
        """
        if not self.current_run_id:
            # Auto-start a run if none exists
            self.start_episodic_run("general")
        
        return episodic_memory.record_episode(
            role=role,
            summary=summary,
            tags=tags,
            entities=entities,
            outcome=outcome,
            confidence=confidence
        )
    
    def get_relevant_episodes(
        self,
        role: str = None,
        tags: List[str] = None,
        limit: int = 5,
        min_confidence: float = 0.3,
        time_decay_hours: float = 168.0  # 1 week
    ) -> List[EpisodicEntry]:
        """
        Retrieve relevant episodes for the current context.
        
        Args:
            role: Filter by role (None for any role)
            tags: Filter by tags (must contain all specified tags)
            limit: Maximum number of results
            min_confidence: Minimum confidence threshold
            time_decay_hours: Hours for time decay calculation
            
        Returns:
            List of relevant episodic entries
        """
        return episodic_memory.retrieve_episodes(
            domain=self.current_domain,
            role=role,
            tags=tags,
            limit=limit,
            min_confidence=min_confidence,
            time_decay_hours=time_decay_hours
        )
    
    def search_episodes_by_query(self, query: str, role: str = None, limit: int = 5) -> List[EpisodicEntry]:
        """
        Semantic search of past episodes using natural language query.
        
        Args:
            query: Natural language description of what to search for
            role: Filter by role (None for any role)
            limit: Maximum number of results
            
        Returns:
            List of semantically similar episodes
        """
        return episodic_memory.search_episodes_semantic(
            query=query,
            domain=self.current_domain,
            role=role,
            limit=limit
        )
    
    def get_domain_history(self, domain: str = None, limit: int = 10) -> List[EpisodicEntry]:
        """
        Get recent history for a specific domain.
        
        Args:
            domain: Domain to get history for (None for current domain)
            limit: Maximum number of results
            
        Returns:
            List of recent episodes for the domain
        """
        target_domain = domain or self.current_domain
        return episodic_memory.retrieve_episodes(
            domain=target_domain,
            limit=limit,
            min_confidence=0.0
        )
    
    def get_role_experiences(self, role: str, limit: int = 10) -> List[EpisodicEntry]:
        """
        Get past experiences for a specific role across all domains.
        
        Args:
            role: Role to get experiences for
            limit: Maximum number of results
            
        Returns:
            List of episodes for the specified role
        """
        return episodic_memory.retrieve_episodes(
            role=role,
            limit=limit,
            min_confidence=0.0
        )
    
    def get_current_run_summary(self) -> Dict[str, Any]:
        """
        Get summary of the current episodic run.
        
        Returns:
            Dictionary with run statistics and episode summaries
        """
        return episodic_memory.get_run_summary(self.current_run_id)
    
    def build_episodic_context(self, query: str, max_episodes: int = 3) -> str:
        """
        Build context string from relevant past episodes.
        
        Args:
            query: Query to find relevant episodes
            max_episodes: Maximum number of episodes to include
            
        Returns:
            Formatted context string with past episode insights
        """
        relevant_episodes = self.search_episodes_by_query(query, limit=max_episodes)
        
        if not relevant_episodes:
            return ""
        
        context_parts = ["=== RELEVANT PAST EXPERIENCES ==="]
        
        for i, episode in enumerate(relevant_episodes, 1):
            outcome_status = episode.outcome.get("status", "unknown")
            confidence_str = f"confidence: {episode.confidence:.2f}"
            support_str = f"support: {episode.support_count}"
            
            context_parts.append(
                f"\n{i}. [{episode.role}] {episode.summary}\n"
                f"   Domain: {episode.domain} | Outcome: {outcome_status} | {confidence_str} | {support_str}\n"
                f"   Tags: {', '.join(episode.tags[:5])}"  # Limit tags to avoid clutter
            )
            
            # Add any important notes from outcome
            if "notes" in episode.outcome:
                context_parts.append(f"   Notes: {episode.outcome['notes']}")
        
        context_parts.append("=== END PAST EXPERIENCES ===")
        return "\n".join(context_parts)
    
    def update_episode_outcome(self, episode_id: str, outcome: Dict[str, Any]):
        """
        Update the outcome of a previously recorded episode.
        This is useful for updating episodes after learning the final result.
        
        Args:
            episode_id: ID of the episode to update
            outcome: New outcome information
        """
        # This would require adding an update method to episodic_memory
        # For now, just log the update intent
        logger.info(f"Episode outcome update requested for {episode_id}: {outcome}")
    
    # ===============================
    # SEMANTIC MEMORY INTEGRATION
    # ===============================
    
    async def write_semantic_fact(
        self, 
        text: str,
        entities: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        confidence: float = 0.7
    ) -> Optional[str]:
        """
        Write a semantic fact to persistent memory.
        
        Args:
            text: The fact text (≤ 2-3 sentences)
            entities: Named entities mentioned in the fact
            tags: Categorization tags
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Fact ID if stored, None if rejected
        """
        if not self.current_domain:
            logger.warning("No current domain set - cannot write semantic fact")
            return None
        
        return await semantic_memory_store.write_fact(
            domain=self.current_domain,
            text=text,
            entities=entities,
            tags=tags,
            confidence=confidence
        )
    
    async def get_semantic_facts(
        self,
        query: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 6
    ) -> List[SemanticFact]:
        """
        Retrieve semantic facts for the current domain.
        
        Args:
            query: Optional query text for semantic matching
            entities: Optional entity filters
            tags: Optional tag filters
            limit: Maximum number of facts to return
            
        Returns:
            List of ranked SemanticFact objects
        """
        if not self.current_domain:
            logger.warning("No current domain set - cannot retrieve semantic facts")
            return []
        
        return await semantic_memory_store.retrieve_facts(
            domain=self.current_domain,
            query=query,
            entities=entities,
            tags=tags,
            limit=limit
        )
    
    async def build_semantic_context(self, query: str, max_facts: int = 4) -> str:
        """
        Build context string from relevant semantic facts.
        
        Args:
            query: Query to find relevant facts
            max_facts: Maximum number of facts to include
            
        Returns:
            Formatted context string with semantic knowledge
        """
        relevant_facts = await self.get_semantic_facts(query=query, limit=max_facts)
        
        if not relevant_facts:
            return ""
        
        context_parts = ["=== RELEVANT DOMAIN KNOWLEDGE ==="]
        
        for i, fact in enumerate(relevant_facts, 1):
            confidence_str = f"confidence: {fact.confidence:.2f}"
            support_str = f"support: {fact.support_count}"
            
            context_parts.append(
                f"\n{i}. {fact.text}\n"
                f"   {confidence_str} | {support_str}"
            )
            
            # Add tags if available
            if fact.tags:
                context_parts.append(f"   Tags: {', '.join(fact.tags[:4])}")
        
        context_parts.append("=== END DOMAIN KNOWLEDGE ===")
        return "\n".join(context_parts)
    
    async def promote_episode_to_semantic(
        self,
        episode_summary: str,
        outcome: Dict[str, Any],
        confidence: float = 0.8
    ) -> Optional[str]:
        """
        Promote a successful episodic experience to semantic memory.
        
        This is typically called by the evaluator after successful outcomes
        to distill lessons learned into persistent knowledge.
        
        Args:
            episode_summary: Summary of the episodic experience
            outcome: Outcome information from the episode
            confidence: Confidence in the semantic fact
            
        Returns:
            Semantic fact ID if promoted, None if not suitable
        """
        if not self.current_domain:
            logger.warning("No current domain set - cannot promote to semantic memory")
            return None
        
        return await semantic_memory_store.promote_from_episode(
            domain=self.current_domain,
            episode_summary=episode_summary,
            outcome=outcome,
            confidence=confidence
        )
    
    async def get_domain_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about knowledge in the current domain"""
        if not self.current_domain:
            return {"error": "No current domain set"}
        
        # Get semantic memory stats
        semantic_stats = await semantic_memory_store.get_domain_stats(self.current_domain)
        
        # Get episodic memory stats (would need to add this method to episodic memory)
        episodic_count = len(episodic_memory.retrieve_episodes(limit=1000))
        
        return {
            "domain": self.current_domain,
            "semantic_facts": semantic_stats.get("total_facts", 0),
            "avg_confidence": semantic_stats.get("avg_confidence", 0.0),
            "episodic_experiences": episodic_count,
            "combined_knowledge_sources": 2
        }