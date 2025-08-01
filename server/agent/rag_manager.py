"""
Enhanced RAG Manager with fallback implementation.

This module provides a robust RAG system that falls back to a simple implementation
when ML dependencies are not available.
"""
from server.agent.rag_manager_enhanced import rag_manager

# Export the enhanced manager for backward compatibility
__all__ = ['rag_manager']