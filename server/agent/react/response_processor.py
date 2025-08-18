"""
Processes LLM responses and extracts thinking/content
"""
import re
from typing import AsyncGenerator

from server.agent.knowledge_store import KnowledgeStore, ContextType, ImportanceLevel
from server.agent.learning_loop import learning_loop
from .memory_manager import MemoryManager


class ResponseProcessor:
    """Processes LLM responses and extracts thinking/content"""
    
    def __init__(self, knowledge_store: KnowledgeStore, memory_manager: MemoryManager):
        self.knowledge_store = knowledge_store
        self.memory = memory_manager
    
    async def process_streaming_response(self, llm_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        """Process streaming response and yield appropriate events"""
        response_buffer = ""
        thinking_content = ""
        thinking_started = False
        thinking_ended = False
        answering = False
        content_after_thinking = ""
        pre_think_buffer = ""
        
        async for token in llm_stream:
            response_buffer += token
            
            if not thinking_ended and thinking_started:

                if thinking_started and "</think>" in response_buffer:
                    thinking_ended = True
                    thinking_content = self._extract_thinking_content(response_buffer, pre_think_buffer)
                    self._record_thinking(thinking_content)
                    
                    content_after_thinking = response_buffer.rsplit("</think>", 1)[-1]
                    yield f'<token>{content_after_thinking}</token>'
                
                elif thinking_started:
                    yield f'<thought>{token}</thought>'
            else:
                content_after_thinking = response_buffer.rsplit("</think>", 1)[-1]
                yield f'<token>{token}</token>'
        
        yield f"response_buffer:{response_buffer}"
            
    def _extract_thinking_content(self, response_buffer: str, pre_think_buffer: str) -> str:
        """Extract thinking content from response"""
        think_match = re.search(r'<think>(.*?)</think>', response_buffer, re.DOTALL)
        if think_match:
            return (pre_think_buffer + think_match.group(1)).strip()
        return ""
    
    def _extract_thinking_from_complete_response(self, response_buffer: str) -> str:
        """Extract thinking content from complete response"""
        thinking_match = re.search(r'<think>(.*?)</think>', response_buffer, re.DOTALL)
        if thinking_match:
            return thinking_match.group(1).strip()
        return ""
    
    def _record_thinking(self, thinking_content: str):
        """Record thinking content in knowledge store and learning loop"""
        if thinking_content:
            self.knowledge_store.add_context(
                thinking_content,
                ContextType.THOUGHT,
                ImportanceLevel.MEDIUM,
                "react_agent"
            )
            learning_loop.record_action("thinking", {"content": thinking_content[:200] + "..."})
