"""
ConversationState manages all request-specific data including messages, 
tool outputs, and conversation context in a single, centralized class.
"""
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class StateType(Enum):
    """Types of conversation state data"""
    MESSAGE = "message"
    TOOL_OUTPUT = "tool_output" 
    THOUGHT = "thought"
    ACTION = "action"
    CONTEXT = "context"


@dataclass
class StateEntry:
    """Individual entry in conversation state"""
    content: str
    state_type: StateType
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationState:
    """
    Centralized state management for a single conversation request.
    Replaces scattered state management across the agent components.
    """
    
    def __init__(self, initial_messages: Optional[List[Dict[str, str]]] = None):
        self.messages: List[Dict[str, str]] = initial_messages or []
        self.state_entries: List[StateEntry] = []
        self.tool_outputs: Dict[str, Any] = {}
        self.context_data: Dict[str, Any] = {}
        self.explored_files: Set[str] = set()
        self.start_time = time.time()
        self.is_complete = False
        self.final_response = ""
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation"""
        message = {"role": role, "content": content}
        if metadata:
            message.update(metadata)
        self.messages.append(message)
        
        # Also track as state entry
        self.add_state_entry(content, StateType.MESSAGE, metadata or {})
    
    def add_state_entry(self, content: str, state_type: StateType, metadata: Optional[Dict] = None):
        """Add an entry to the state log"""
        entry = StateEntry(
            content=content,
            state_type=state_type, 
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.state_entries.append(entry)
    
    def add_tool_output(self, tool_name: str, output: Any, metadata: Optional[Dict] = None):
        """Store tool execution results"""
        self.tool_outputs[tool_name] = {
            "output": output,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Track as state entry too
        self.add_state_entry(
            f"Tool {tool_name}: {str(output)[:200]}", 
            StateType.TOOL_OUTPUT,
            {"tool_name": tool_name}
        )
    
    def add_context(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Store contextual information"""
        self.context_data[key] = {
            "value": value,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.add_state_entry(
            f"Context {key}: {str(value)[:100]}",
            StateType.CONTEXT,
            {"context_key": key}
        )
    
    def add_file_exploration(self, filename: str, content: Optional[str] = None):
        """Track explored files"""
        self.explored_files.add(filename)
        if content:
            self.add_context(f"file_content_{filename}", content, {"filename": filename})
    
    def get_recent_context(self, limit: int = 10) -> List[StateEntry]:
        """Get recent state entries for context"""
        return self.state_entries[-limit:]
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM consumption"""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]
    
    def build_context_summary(self) -> str:
        """Build a summary of current conversation context"""
        summary_parts = []
        
        if self.explored_files:
            summary_parts.append(f"Explored files: {', '.join(list(self.explored_files)[:5])}")
        
        if self.tool_outputs:
            recent_tools = list(self.tool_outputs.keys())[-3:]
            summary_parts.append(f"Recent tools used: {', '.join(recent_tools)}")
        
        recent_entries = self.get_recent_context(5)
        if recent_entries:
            summary_parts.append("Recent activity:")
            for entry in recent_entries:
                summary_parts.append(f"- {entry.state_type.value}: {entry.content[:100]}")
        
        return "\n".join(summary_parts) if summary_parts else "No significant context yet."
    
    def mark_complete(self, final_response: str):
        """Mark conversation as complete"""
        self.is_complete = True
        self.final_response = final_response
        self.add_state_entry(final_response, StateType.MESSAGE, {"final": True})
    
    def get_duration(self) -> float:
        """Get conversation duration in seconds"""
        return time.time() - self.start_time