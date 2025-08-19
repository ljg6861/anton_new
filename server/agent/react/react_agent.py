"""
ReAct (Reason-Act) Agent: single-loop reasoning, tool use, and response streaming.
Main agent class that coordinates all ReAct components.
Enhanced with tool learning capabilities.
"""
import json
import logging
import re
from typing import AsyncGenerator, List, Dict, Any, Optional

import httpx

from server.agent.knowledge_store import KnowledgeStore, ContextType, ImportanceLevel
from server.agent.tool_executor import process_tool_calls
from server.agent.tool_learning_store import tool_learning_store
from server.agent.config import ASSISTANT_ROLE, SYSTEM_ROLE, USER_ROLE
from server.agent import config
from server.agent.learning_loop import learning_loop
from server.agent.tools.tool_manager import tool_manager

from .token_budget import TokenBudget
from .memory_manager import MemoryManager
from .token_loop_detector import TokenLoopDetector
from .system_prompt_builder import SystemPromptBuilder
from .tool_formatter import ToolFormatter
from .response_processor import ResponseProcessor

logger = logging.getLogger(__name__)


class ReActAgent:
    """
    Single-agent ReAct implementation with three-memory architecture:
    - Working Memory (WM): recent messages that fit in token budget
    - Session Memory (SM): decisions, TODOs, context from this chat session  
    - Long-term Memory (LTM): RAG + domain packs, retrieved as needed
    """

    def __init__(
        self,
        api_base_url: str,
        tools: List[Dict],
        knowledge_store: KnowledgeStore,
        max_iterations: int = 10,
        domain_pack_dir: str = "../../learning/packs",
        user_id: Optional[str] = None,
        token_budget: Optional[TokenBudget] = None,
    ) -> None:
        self.api_base_url = api_base_url
        self.tools = tools
        self.knowledge_store = knowledge_store
        self.max_iterations = max_iterations
        self.domain_pack_dir = domain_pack_dir
        self.user_id = user_id or "anonymous"
        
        # Initialize components
        self.budget = token_budget or TokenBudget()
        self.memory = MemoryManager(self.budget)
        self.loop_detector = TokenLoopDetector()
        self.prompt_builder = SystemPromptBuilder(self.memory, self.knowledge_store)
        self.tool_formatter = ToolFormatter(self.memory)
        self.response_processor = ResponseProcessor(self.knowledge_store, self.memory)
        
        logger.info(f"ReActAgent initialized with token budget: {self.budget.total_budget}")
    
    def _initialize_conversation_tracking(self, messages: List[Dict[str, str]]):
        """Initialize conversation in knowledge store and tool learning store for context tracking"""
        # Initialize tool learning conversation tracking
        conversation_id = f"react_{self.user_id}_{int(__import__('time').time())}"
        tool_learning_store.start_conversation(conversation_id)
        
        for msg in messages:
            role = msg.get("role")
            if role:
                self.knowledge_store.add_message(role, msg.get("content", ""))

    async def _llm_analysis_callback(self, analysis_prompt: str) -> str:
        """
        Callback function for LLM analysis of tool learning patterns.
        Uses the same LLM that powers the agent for consistency.
        """
        try:
            logger.info("Performing LLM analysis for tool learning pattern")
            
            analysis_messages = [
                {"role": "system", "content": "You are an expert at analyzing tool execution patterns to extract useful learnings."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            # Use the same API endpoint as the main agent
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base_url}/v1/chat/completions",
                    json={
                        "model": "gpt-4",  # Or whatever model the agent uses
                        "messages": analysis_messages,
                        "temperature": 0.1,  # Low temperature for analytical tasks
                        "max_tokens": 1000
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    logger.info("LLM learning analysis completed successfully")
                    return content
                else:
                    logger.error(f"LLM analysis request failed: {response.status_code}")
                    return "Analysis failed: HTTP error"
                    
        except Exception as e:
            logger.error(f"Error in LLM analysis callback: {e}", exc_info=True)
            return f"Analysis failed: {str(e)}"
    
    async def execute_react_loop(self, react_messages: List[Dict[str, str]], 
                                 logger: Any) -> str:
        """Execute the main ReAct reasoning loop"""
        iteration = 0
        
        while iteration < 1:#self.max_iterations:
            iteration += 1
            logger.info(f"ReAct iteration {iteration}/{self.max_iterations}")
            
            self.knowledge_store.add_context(
                f"Starting ReAct iteration {iteration}",
                ContextType.THOUGHT,
                ImportanceLevel.LOW,
                "react_agent"
            )
            
            response = (self._execute_llm_request(react_messages, logger))
            response = response['choices'][0]['message']
            tool_calls = response['tool_calls']
            if (tool_calls):
                for i in range(len(tool_calls)):
                    logger.info(f"Calling tool: {tool_calls[i]['function']['name']} with args: {tool_calls[i]['function']['arguments']}")
                    result = tool_manager.run_tool(tool_calls[i]['function']['name'], tool_calls[i]['function']['arguments'],)
                    return react_messages.append({'role' : 'function', 'name' : tool_calls[i]['function']['name'], 'content' : result}) 
            else:
                return react_messages.append({'role' : 'assistant', 'content' : response['content']})


    
    def _update_session_memory_from_tool_result(self, tool_result_summary: Dict):
        """Update session memory based on tool results"""
        tool_name = tool_result_summary.get("name", "")
        status = tool_result_summary.get("status")
        arguments = tool_result_summary.get("arguments", {})
        
        if tool_name == "create_file" and status == "success":
            file_path = arguments.get('file_path', 'unknown')
            self.memory.add_decision(f"Created file: {file_path}")
        elif tool_name == "run_git_command" and "checkout -b" in str(arguments):
            self.memory.add_decision("Created new git branch")
            self.memory.complete_todo("Create feature branch")
    
    def _handle_final_response(self, content: str) -> bool:
        """Handle final response and complete task"""
        if '[Done]' in content:
            success = "error" not in content.lower() and "failed" not in content.lower()
            learning_loop.complete_task(success, content[:200])
            
            if success:
                self.memory.add_decision("Task completed successfully")
            else:
                self.memory.add_decision("Task completed with errors")
            
            return False  # Stop loop
        else:
            logger.info("Response doesn't appear final, continuing...")
            return True  # Continue loop
    
    async def _handle_max_iterations_reached(self) -> AsyncGenerator[str, None]:
        """Handle case where max iterations is reached"""
        logger.warning(f"ReAct agent reached max iterations ({self.max_iterations})")
        final_msg = "\n\n[Task completed - reached maximum iterations]"
        learning_loop.complete_task(False, "Reached maximum iterations")
        self.memory.add_decision("Task incomplete - reached max iterations")
        yield final_msg
    
    def _sanitize_messages_for_vllm(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert function role messages to system role for vLLM compatibility"""
        sanitized_messages = []
        for msg in messages:
            msg_copy = msg.copy()
            # vLLM doesn't support 'function' role, convert to 'system' with tool result prefix
            if msg_copy.get("role") == "function":
                tool_name = msg_copy.get("name", "tool")
                content = msg_copy.get("content", "")
                msg_copy["role"] = "system"
                msg_copy["content"] = f"TOOL_RESULT from {tool_name}: {content}"
                # Remove the 'name' field as it's not needed for system messages
                msg_copy.pop("name", None)
            sanitized_messages.append(msg_copy)
        return sanitized_messages

    def _execute_llm_request(
        self,
        messages: List[Dict[str, str]],
        logger: Any
    ) -> Dict[str, Any]:
        sanitized_messages = self._sanitize_messages_for_vllm(messages)
        
        request_payload = {
            "model": "qwen3-30b-awq",
            "messages": sanitized_messages,
            "temperature": 0.6,
            "stream": False,
            "max_tokens": 2048
        }
        
        if self.tools:
            request_payload["tools"] = self.tools
            request_payload["tool_choice"] = "auto"

        vllm_url = "http://localhost:8003"
        url = f"{vllm_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer anton-vllm-key"
        }
        
        with httpx.Client(timeout=120.0) as client:
            try:
                response = client.post(url, json=request_payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                error_details = f"HTTP {e.response.status_code}: {e.response.reason_phrase}\n"
                try:
                    error_body = e.response.read().decode()
                    error_details += f"Response body: {error_body}"
                except:
                    error_details += "Could not read response body"
                
                logger.error(f"ReActAgent: HTTP error from vLLM server: {error_details}")
                raise RuntimeError(f"HTTP error from vLLM server: {error_details}") from e
            except httpx.RequestError as e:
                logger.error(f"ReActAgent: API request to vLLM server failed: {e}")
                raise RuntimeError(f"API request to vLLM server failed: {e}") from e
            except Exception as e:
                logger.error(f"ReActAgent: An unexpected error occurred: {e}", exc_info=True)
                raise RuntimeError(f"An unexpected error occurred: {e}") from e
