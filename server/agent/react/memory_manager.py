"""
Manages three types of memory with token budgeting
"""
from typing import List, Dict, Any
import json
import httpx

from .token_budget import TokenBudget


class MemoryManager:
    """Manages three types of memory with token budgeting"""

    def __init__(self, budget: TokenBudget):
        # Core budgets & session state
        self.budget = budget
        self.session_decisions: List[str] = []
        self.session_todos: List[str] = []
        self.session_context: str = ""
        # Conversation summarization state
        self.conversation_summary: str = ""  # Rolling summary of overflow messages
        self._last_summarized_index: int = 0  # Index in message list up to which we've summarized
        self._use_heuristic_fallback: bool = True  # Switch off when LLM summarization engaged
        
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
        """Build working memory from recent messages with graceful summarization of overflow.

        Strategy:
        - Keep most recent messages until near budget.
        - Any earlier (overflow) messages not yet summarized are compressed into a rolling
          conversation_summary so nothing is lost logically.
        - The summary itself is budgeted; if it grows too large, we recursively compress it.
        """
        budget = self.budget.working_memory_budget

        if not messages:
            return self.conversation_summary or ""

        # We reserve up to 30% of working memory for the rolling summary (if it exists)
        summary_reserved_tokens = int(budget * 0.30) if self.conversation_summary else 0
        remaining_budget = budget - summary_reserved_tokens

        selected_messages: List[str] = []
        total_tokens = 0

        # Walk backwards to pick most recent messages within remaining budget
        cut_index = 0  # Oldest index included
        for i in range(len(messages) - 1, -1, -1):
            formatted_msg = self._format_message(messages[i])
            msg_tokens = self.estimate_tokens(formatted_msg)
            if total_tokens + msg_tokens <= remaining_budget:
                selected_messages.insert(0, formatted_msg)
                total_tokens += msg_tokens
                cut_index = i
            else:
                cut_index = i  # first excluded index (older than this index will be summarized)
                break

        # Determine overflow range: messages before cut_index
        overflow_end = max(0, cut_index)  # exclusive
        if self._use_heuristic_fallback and overflow_end > self._last_summarized_index:
            # New overflowed messages that haven't been summarized yet (heuristic path)
            new_overflow_segment = messages[self._last_summarized_index:overflow_end]
            if new_overflow_segment:
                segment_summary = self._summarize_messages(new_overflow_segment)
                if segment_summary:
                    if self.conversation_summary:
                        self.conversation_summary += "\n" + segment_summary
                    else:
                        self.conversation_summary = segment_summary
                self._last_summarized_index = overflow_end
                self._compress_conversation_summary(target_tokens=int(budget * 0.35))

        # If summary exists but we didn't reserve earlier (e.g., first time) adjust
        summary_text = ""
        if self.conversation_summary:
            summary_text = self._prefix_summary(self.conversation_summary, int(budget * 0.35))

        parts = []
        if summary_text:
            parts.append(summary_text)
        if selected_messages:
            parts.append("\n".join(selected_messages))
        return "\n".join([p for p in parts if p])

    def _summarize_messages(self, messages: List[Dict[str, str]]) -> str:
        """Heuristically summarize a list of messages (older overflow segment).

        Approach: group consecutive messages by role, truncate each content, and note counts.
        This is a lightweight, no-LLM approach to avoid network calls; can be upgraded later.
        """
        if not messages:
            return ""
        groups: List[Dict[str, Any]] = []  # type: ignore
        current_role = None
        current_msgs: List[str] = []
        for m in messages:
            role = m.get('role', 'unknown')
            content = (m.get('content') or '').strip()
            if current_role is None:
                current_role = role
            if role != current_role:
                groups.append({'role': current_role, 'messages': current_msgs})
                current_role = role
                current_msgs = []
            # Compress each message content
            snippet = content.replace('\n', ' ')[:200]
            current_msgs.append(snippet)
        if current_role is not None:
            groups.append({'role': current_role, 'messages': current_msgs})

        summary_lines = ["EARLIER CONVERSATION (compressed):"]
        for g in groups:
            msgs = g['messages']
            if not msgs:
                continue
            # If many messages, show first and last
            if len(msgs) > 3:
                line = f"[{g['role']}] {msgs[0]} ... ({len(msgs)-2} more) ... {msgs[-1]}"
            else:
                line = f"[{g['role']}] " + " | ".join(msgs)
            summary_lines.append(line)
        summary_text = "\n".join(summary_lines)
        return self._truncate_at_word_boundary(summary_text, int(self.budget.working_memory_budget * 0.35))

    def _compress_conversation_summary(self, target_tokens: int):
        """Compress the rolling summary if it exceeds target token budget.

        Simple strategy: if too large, keep first 60% + last 30% with ellipsis marker.
        """
        if not self.conversation_summary:
            return
        tokens = self.estimate_tokens(self.conversation_summary)
        if tokens <= target_tokens:
            return
        # Compress
        text = self.conversation_summary
        keep_front = int(len(text) * 0.6)
        keep_back = int(len(text) * 0.3)
        compressed = text[:keep_front] + "\n... <summary compressed> ...\n" + text[-keep_back:]
        # Truncate again just in case
        self.conversation_summary = self._truncate_at_word_boundary(compressed, target_tokens)

    def _prefix_summary(self, summary: str, max_tokens: int) -> str:
        """Prepare summary block for insertion into working memory."""
        trimmed = self._truncate_at_word_boundary(summary, max_tokens)
        return trimmed if trimmed.startswith('EARLIER CONVERSATION') else f"EARLIER CONVERSATION SUMMARY (rolling):\n{trimmed}"

    async def update_llm_conversation_summary(self, messages: List[Dict[str, str]], api_base_url: str):
        """Use LLM to summarize newly overflowed messages beyond working memory budget.

        Only summarizes messages that won't fit and have not been previously summarized.
        """
        budget = self.budget.working_memory_budget
        if not messages:
            return

        # Determine how many recent messages fit (ignoring existing summary space)
        remaining_budget = budget
        total_tokens = 0
        fit_start_index = len(messages)  # index of first message that fits in window
        for i in range(len(messages) - 1, -1, -1):
            formatted = self._format_message(messages[i])
            t = self.estimate_tokens(formatted)
            if total_tokens + t <= remaining_budget:
                total_tokens += t
                fit_start_index = i
            else:
                break

        overflow_end = max(0, fit_start_index)  # exclusive
        if overflow_end <= self._last_summarized_index:
            # Nothing new to summarize
            return

        overflow_segment = messages[self._last_summarized_index:overflow_end]
        if not overflow_segment:
            return

        # Build summarization prompt
        formatted_msgs = []
        for m in overflow_segment:
            role = m.get('role', 'unknown')
            content = (m.get('content') or '').strip()
            formatted_msgs.append(f"[{role}] {content}")
        joined = "\n".join(formatted_msgs)

        system_prompt = (
            "You are a conversation summarizer. Produce a concise bullet summary of the older "
            "messages preserving: decisions made, user intents/questions, TODOs, action items, "
            "file names, code symbols. 300 words max. Output only the summary bullets."
        )
        user_prompt = (
            "Summarize the following earlier messages (do NOT include recent ones).\n\n" + joined
        )

        req_payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }

        summary_text = ""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = f"{api_base_url}/v1/chat/stream"
                async with client.stream("POST", url, json=req_payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = data.get("message", {}).get("content", "")
                        if content:
                            summary_text += content
        except Exception as e:
            # Fallback to heuristic summarization if LLM fails
            segment_summary = self._summarize_messages(overflow_segment)
            if segment_summary:
                if self.conversation_summary:
                    self.conversation_summary += "\n" + segment_summary
                else:
                    self.conversation_summary = segment_summary
                self._last_summarized_index = overflow_end
                self._compress_conversation_summary(target_tokens=int(budget * 0.35))
            return

        if summary_text.strip():
            cleaned = summary_text.strip()
            prefixed = "EARLIER CONVERSATION (compressed):\n" + cleaned
            if self.conversation_summary:
                self.conversation_summary += "\n" + prefixed
            else:
                self.conversation_summary = prefixed
            self._last_summarized_index = overflow_end
            self._use_heuristic_fallback = False
            self._compress_conversation_summary(target_tokens=int(budget * 0.35))
    
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
