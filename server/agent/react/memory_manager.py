"""
Manages three types of memory with token budgeting
"""
from typing import List, Dict

from .token_budget import TokenBudget


class MemoryManager:
    """Manages three types of memory with token budgeting"""
    
    def __init__(self, budget: TokenBudget):
        self.budget = budget
        self.session_decisions = []
        self.session_todos = []
        self.session_context = ""
        
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars for English)"""
        return len(text) // 4
        
    def truncate_to_budget(self, text: str, budget: int) -> str:
        """Truncate text to fit within token budget"""
        estimated_tokens = self.estimate_tokens(text)
        if estimated_tokens <= budget:
            return text
        
        return self._truncate_at_word_boundary(text, budget)
    
    def _truncate_at_word_boundary(self, text: str, budget: int) -> str:
        """Truncate text at word boundary to fit budget"""
        target_chars = budget * 4
        if len(text) <= target_chars:
            return text
            
        truncated = text[:target_chars]
        last_space = truncated.rfind(' ')
        
        # Don't cut too much
        if last_space > target_chars * 0.8:
            truncated = truncated[:last_space]
        
        return truncated + "..."
        
    def build_working_memory(self, messages: List[Dict[str, str]]) -> str:
        """Build working memory from recent messages (WM)"""
        budget = self.budget.working_memory_budget
        selected_messages = []
        total_tokens = 0
        
        for msg in reversed(messages):
            formatted_msg = self._format_message(msg)
            msg_tokens = self.estimate_tokens(formatted_msg)
            
            if total_tokens + msg_tokens <= budget:
                selected_messages.insert(0, formatted_msg)
                total_tokens += msg_tokens
            else:
                break
                
        return "\n".join(selected_messages)
    
    def _format_message(self, msg: Dict[str, str]) -> str:
        """Format a message for working memory"""
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        return f"[{role}]: {content}"
        
    def build_session_memory(self) -> str:
        """Build session memory summary (SM)"""
        parts = []
        
        if self.session_context:
            parts.append(f"Context: {self.session_context}")
            
        if self.session_decisions:
            decisions_text = "; ".join(self.session_decisions[-5:])
            parts.append(f"Decisions: {decisions_text}")
            
        if self.session_todos:
            todos_text = "; ".join(self.session_todos[-3:])
            parts.append(f"TODOs: {todos_text}")
            
        session_text = " | ".join(parts)
        return self.truncate_to_budget(session_text, self.budget.session_summary_budget)
        
    def add_decision(self, decision: str):
        """Add a decision to session memory"""
        self.session_decisions.append(decision)
        self._maintain_decision_list_size()
            
    def _maintain_decision_list_size(self):
        """Keep decision list within reasonable size"""
        if len(self.session_decisions) > 10:
            self.session_decisions = self.session_decisions[-10:]
            
    def add_todo(self, todo: str):
        """Add a TODO to session memory"""
        if todo not in self.session_todos:
            self.session_todos.append(todo)
            
    def complete_todo(self, todo: str):
        """Mark a TODO as complete"""
        if todo in self.session_todos:
            self.session_todos.remove(todo)
            
    def set_session_context(self, context: str):
        """Set the current session context"""
        self.session_context = context
