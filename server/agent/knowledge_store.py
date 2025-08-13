"""
Centralized knowledge management system that tracks context across planner, doer, and evaluator components.
Provides persistent storage, context prioritization, and knowledge transfer capabilities.
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
    Integrates with existing RAG manager for persistent storage.
    """
    
    def __init__(self):
        # In-memory context tracking (similar to existing context_store)
        self.explored_files: Set[str] = set()
        self.code_content: Dict[str, str] = {}
        self.task_progress: List[str] = []
        
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
        centroids = load_centroids()
        logger.info('loaded centroids')
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