"""
ReAct (Reason-Act) Agent: single-loop reasoning, tool use, and response streaming.
Main agent class that coordinates all ReAct components.
Enhanced with tool learning capabilities.
"""
import asyncio
import json
import logging
import re
from typing import AsyncGenerator, List, Dict, Any, Optional

import httpx

from server.agent.agentic_flow.helpers_and_prompts import CURRENT_MODEL, call_model_for_summarization
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
        
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (roughly 4 chars per token for English)"""
        return len(text) // 4
    
    def calculate_total_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Calculate total tokens used in all messages"""
        total_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            name = msg.get("name", "")
            # Count content + role + name fields
            total_tokens += self.estimate_tokens(content + msg.get("role", "") + name)
        return total_tokens
    
    async def _summarize_text_in_batches(self, conversation_text: str) -> str:
        """
        Summarizes a given text, handling very large texts by processing them in batches.
        """
        conversation_tokens = self.estimate_tokens(conversation_text)

        if conversation_tokens <= 20000:
            logger.info("Conversation is under 20k tokens. Performing single summarization.")
            return await call_model_for_summarization(conversation_text)

        logger.info(f"Conversation is large ({conversation_tokens} tokens). Starting batched summarization...")
        
        CHUNK_SIZE = 18000 * 4 
        text_chunks = [conversation_text[i:i + CHUNK_SIZE] for i in range(0, len(conversation_text), CHUNK_SIZE)]
        
        logger.info(f"Splitting conversation into {len(text_chunks)} chunks for summarization.")

        summarization_tasks = [call_model_for_summarization(chunk) for chunk in text_chunks]
        partial_summaries = await asyncio.gather(*summarization_tasks)
        
        combined_summary_text = "\n\n---\n\n".join(partial_summaries)
        logger.info("Combining partial summaries for a final summarization pass.")

        final_summary_prompt = (
            "The following are multiple summaries from a long conversation. "
            "Please synthesize them into a single, cohesive final summary:\n\n"
            f"{combined_summary_text}"
        )
        final_summary = await call_model_for_summarization(final_summary_prompt)
        
        return final_summary
    
    async def check_and_summarize_if_needed(self, react_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Checks if the message history exceeds a token threshold and, if so, preserves the
        first two messages (system, user) and summarizes the rest of the conversation.
        """
        total_tokens = self.calculate_total_message_tokens(react_messages)
        
        # No action needed if within the limit.
        if total_tokens <= 15000:
            return react_messages
        
        # Not enough messages to perform a summary (we need to keep at least 2).
        if len(react_messages) <= 2:
            return react_messages

        logger.info(f"Message history has {total_tokens} tokens, exceeding 15k limit. Starting summarization...")
        
        # Correctly preserve the first two messages (system and initial user prompt).
        messages_to_keep = react_messages[:2]
        messages_to_summarize = react_messages[2:]
        
        # Format the conversation history for the summarization model.
        formatted_history = []
        for msg in messages_to_summarize:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "function" and "name" in msg:
                 formatted_history.append(f"[TOOL RESULT from {msg['name']}]: {content}")
            else:
                 formatted_history.append(f"[{role.upper()}]: {content}")
        
        conversation_text = "\n".join(formatted_history)
        
        # Get the summary using the new batching-aware helper method.
        summary = await self._summarize_text_in_batches(conversation_text)
        
        # Create the new summary message to insert into the history.
        summary_message = {
            "role": "assistant",
            "content": f"A portion of the conversation was summarized to save space. "
                       f"({len(messages_to_summarize)} messages replaced)\n\n"
                       f"SUMMARY OF PREVIOUS MESSAGES:\n{summary}"
        }
        
        # Construct the new, shorter message list.
        new_messages = messages_to_keep + [summary_message]
        
        new_token_count =  self.calculate_total_message_tokens(new_messages)
        logger.info(
            f"Summarization complete: {total_tokens} tokens -> {new_token_count} tokens "
            f"({len(react_messages)} messages -> {len(new_messages)} messages)"
            f"{summary_message}"
        )
        
        return new_messages
    
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
        recent_tool_calls = []  # Track recent tool calls to detect loops
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"ReAct iteration {iteration}/{self.max_iterations}")
            
            # Only check summarization occasionally for the researcher to preserve context
            if iteration % 10 == 0:  # Check every 10 iterations instead of every iteration
                react_messages = await self.check_and_summarize_if_needed(react_messages)
            
            response = (self._execute_llm_request(react_messages, logger))
            response = response['choices'][0]['message']
            tool_calls = response['tool_calls']
            if (tool_calls):
                for i in range(len(tool_calls)):
                    tool_name = tool_calls[i]['function']['name']
                    tool_args = tool_calls[i]['function']['arguments']
                    
                    # Create a signature for this tool call
                    tool_signature = f"{tool_name}:{tool_args}"
                    
                    # Check for repetitive tool calls (same tool with same args within last 3 calls - more aggressive)
                    if tool_signature in recent_tool_calls[-3:]:
                        logger.warning(f"Detected repetitive tool call: {tool_name} with {tool_args}")
                        # Add a strong guidance message instead of executing the same tool again
                        guidance_msg = f"🔄 STOP REPEATING! You've already tried {tool_name} with these arguments recently. If you have enough information to synthesize findings, CONCLUDE your research instead of searching more!"
                        react_messages.append({
                            'role': 'function', 
                            'name': tool_name, 
                            'content': f"LOOP_DETECTED: {guidance_msg}"
                        })
                        continue
                    
                    # Also check if using the same tool too frequently (same tool type within last 4 calls)
                    same_tool_count = sum(1 for sig in recent_tool_calls[-4:] if sig.startswith(f"{tool_name}:"))
                    if same_tool_count >= 3:
                        logger.warning(f"Tool {tool_name} used too frequently - forcing diversity")
                        guidance_msg = f"🚨 DIVERSIFY! You've used {tool_name} {same_tool_count} times recently. Try different tools or conclude with your current findings!"
                        react_messages.append({
                            'role': 'function', 
                            'name': tool_name, 
                            'content': f"FREQUENCY_WARNING: {guidance_msg}"
                        })
                        continue
                    
                    logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
                    try:
                        result = tool_manager.run_tool(tool_name, tool_args)
                        react_messages.append({'role' : 'function', 'name' : tool_name, 'content' : result}) 
                            
                    except Exception as e:
                        logger.error(f"Error occurred while calling tool {tool_name}: {e}")
                        react_messages.append({'role' : 'function', 'name' : tool_name, 'content' : str(e)}) 
            else:
                # Add the assistant's final response and return
                react_messages.append({'role' : 'assistant', 'content' : response['content']})
                return response['content']
        
        # If we reached max iterations, return the last response
        return "Maximum iterations reached"


    
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
        logger: Any,
    ) -> Dict[str, Any]:
        sanitized_messages = self._sanitize_messages_for_vllm(messages)
        
        request_payload = {
            "model": "anton",
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
