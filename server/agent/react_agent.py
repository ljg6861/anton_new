"""
ReAct (Reason-Act) Agent: single-loop reasoning, tool use, and response streaming.
Integrates KnowledgeStore, LearningLoop, RAG, and per-user memory.
Uses three-memory architecture with token budgeting.

This is a backwards-compatible import layer for the refactored ReAct components.
"""

# Import all components from the new modular structure
from .react import (
    TokenBudget,
    TokenLoopDetector,
    MemoryManager,
    SystemPromptBuilder,
    ToolFormatter,
    ResponseProcessor,
    ReActAgent
)

# Make them available at the module level for backwards compatibility
__all__ = [
    'TokenBudget',
    'TokenLoopDetector',
    'MemoryManager', 
    'SystemPromptBuilder',
    'ToolFormatter',
    'ResponseProcessor',
    'ReActAgent'
]