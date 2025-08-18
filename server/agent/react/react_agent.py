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

    async def process_request(
        self,
        initial_messages: List[Dict[str, str]],
        logger: Any
    ) -> AsyncGenerator[str, None]:
        """Process request using three-memory architecture with token budgeting"""
        logger.info("Starting ReAct agent with three-memory system...")

        user_prompt = self._extract_user_prompt(initial_messages)
        
        if user_prompt:
            learning_loop.start_task(user_prompt)

        self._update_session_context_from_prompt(user_prompt)

        # Initialize conversation tracking (adds messages to knowledge store with dedup)
        self._initialize_conversation_tracking(initial_messages)

        # Build memories from full stored conversation (persistent across requests)
        full_history = self.knowledge_store.get_messages_for_llm()
        # Ensure LLM-based summarization for overflow before building working memory
        try:
            await self.memory.update_llm_conversation_summary(full_history, self.api_base_url)
        except Exception as e:
            logger.error(f"LLM summarization failed (will fallback to heuristic): {e}")
        working_memory = self.memory.build_working_memory(full_history)
        session_memory = self.memory.build_session_memory()

        # Build messages for LLM
        react_messages = await self._build_react_messages(user_prompt, working_memory, session_memory)

        # Main ReAct loop
        async for response in self._execute_react_loop(react_messages, logger):
            yield response
    
    def _extract_user_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Extract the latest user prompt from messages"""
        for msg in reversed(messages):
            if msg.get("role") == USER_ROLE:
                return msg.get("content", "")
        return ""
    
    def _update_session_context_from_prompt(self, user_prompt: str):
        """Update session context based on user prompt keywords"""
        prompt_lower = user_prompt.lower()
        
        if "implement" in prompt_lower or "create" in prompt_lower:
            self.memory.set_session_context("Implementation task")
            self.memory.add_todo("Ensure code compiles")
        elif "analyze" in prompt_lower or "review" in prompt_lower:
            self.memory.set_session_context("Analysis task") 
        elif "fix" in prompt_lower or "debug" in prompt_lower:
            self.memory.set_session_context("Debugging task")
            self.memory.add_todo("Identify root cause")
    
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
    
    async def _build_react_messages(self, user_prompt: str, working_memory: str, 
                                   session_memory: str) -> List[Dict[str, str]]:
        """Build messages list for ReAct processing"""
        system_prompt = await self.prompt_builder.build_system_prompt(
            user_prompt, working_memory, session_memory, self.tools, self.domain_pack_dir
        )
        
        # Replace tools placeholder with formatted tools
        tools_text = self.tool_formatter.format_tools_compact(self.tools)
        system_prompt = system_prompt.replace("{tools_placeholder}", tools_text)
        
        react_messages = [{"role": SYSTEM_ROLE, "content": system_prompt}]
        
        if working_memory:
            react_messages.append({"role": SYSTEM_ROLE, "content": f"RECENT CONTEXT:\n{working_memory}"})

        if user_prompt:
            react_messages.append({"role": USER_ROLE, "content": user_prompt})

        return react_messages
    
    async def _execute_react_loop(self, react_messages: List[Dict[str, str]], 
                                 logger: Any) -> AsyncGenerator[str, None]:
        """Execute the main ReAct reasoning loop"""
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"ReAct iteration {iteration}/{self.max_iterations}")
            
            self.knowledge_store.add_context(
                f"Starting ReAct iteration {iteration}",
                ContextType.THOUGHT,
                ImportanceLevel.LOW,
                "react_agent"
            )
            
            # Execute LLM request and process response
            llm_stream = self._execute_llm_request(react_messages, logger)
            response_buffer = ""
            
            async for event in self.response_processor.process_streaming_response(llm_stream):
                if event.startswith('<token>') or event.startswith('<thought>'):
                    yield event
                elif event.startswith('response_buffer:'):
                    response_buffer = event.split('response_buffer:', 1)[1]
            
            # Process complete response
            logger.info(f"ReAct agent response: {response_buffer}")
            
            # Handle tool calls and determine next action
            async for result in self._handle_response_and_tools(
                response_buffer, react_messages, logger
            ):
                if result in ["continue", "stop"]:
                    if result == "stop":
                        return
                    # continue to next iteration
                else:
                    yield result  # Tool result or other output
        
        if iteration >= self.max_iterations:
            async for msg in self._handle_max_iterations_reached():
                yield msg
    
    async def _handle_response_and_tools(self, response_buffer: str, 
                                        react_messages: List[Dict[str, str]], 
                                        logger: Any) -> AsyncGenerator[str, None]:
        """Handle response processing and tool execution"""
        content = re.split(r'</think>', response_buffer, maxsplit=1)[-1].strip()

        react_messages.append({"role": ASSISTANT_ROLE, "content": content})
        
        # Import here to avoid circular imports
        from server.agent import config
        
        tool_results_for_ui = []
        
        async def tool_result_callback(tool_result_summary):
            """Capture tool results for UI streaming"""
            tool_results_for_ui.append(tool_result_summary)
            self._record_tool_use(tool_result_summary)
            self._update_session_memory_from_tool_result(tool_result_summary)

        made_tool_calls = await process_tool_calls(
            content, 
            config.TOOL_CALL_REGEX,
            react_messages,
            logger,
            self.knowledge_store,
            tool_result_callback,
            self._llm_analysis_callback
        )
        
        # Stream tool results to UI
        for tool_result_summary in tool_results_for_ui:
            yield f"<tool_result>{json.dumps(tool_result_summary)}</tool_result>"

        final_result = self._handle_final_response(content)
        yield "stop" if not final_result else "continue"
    
    def _record_tool_use(self, tool_result_summary: Dict):
        """Record tool use in learning loop"""
        learning_loop.record_action("tool_use", {
            "tool_name": tool_result_summary.get("name"),
            "arguments": tool_result_summary.get("arguments"),
            "success": tool_result_summary.get("status") == "success",
        })
    
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
        if 'Final Answer:' in content:
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
    
    def _sanitize_messages_for_ollama(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert function role messages to user role for Ollama compatibility"""
        sanitized_messages = []
        for msg in messages:
            msg_copy = msg.copy()
            # Ollama doesn't support 'function' role, convert to 'user' with tool result prefix
            if msg_copy.get("role") == "function":
                tool_name = msg_copy.get("name", "tool")
                content = msg_copy.get("content", "")
                msg_copy["role"] = "system"
                msg_copy["content"] = f"[Tool Result from {tool_name}]: {content}"
                # Remove the 'name' field as it's not needed for user messages
                msg_copy.pop("name", None)
            sanitized_messages.append(msg_copy)
        return sanitized_messages

    async def _execute_llm_request(
        self,
        messages: List[Dict[str, str]],
        logger: Any
    ) -> AsyncGenerator[str, None]:
        """Execute LLM request and stream response"""
        # Sanitize messages for Ollama compatibility
        sanitized_messages = self._sanitize_messages_for_ollama(messages)
        
        request_payload = {
            "messages": sanitized_messages,
            "temperature": 0.6,
            'complex': True,
        }

        logger.info(f"Sending request payload: {json.dumps(request_payload, indent=2)}")

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                url = f"{self.api_base_url}/v1/chat/stream"
                async with client.stream("POST", url, json=request_payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_text():
                        chunk_data = json.loads(chunk)
                        content = chunk_data.get("message", {}).get("content", "")
                        yield content

            except httpx.HTTPStatusError as e:
                # Log the specific HTTP error details
                error_details = f"HTTP {e.response.status_code}: {e.response.reason_phrase}"
                try:
                    error_body = await e.response.aread()
                    error_details += f"\nResponse body: {error_body.decode()}"
                except:
                    error_details += "\nCould not read response body"
                
                logger.error(f"ReActAgent: HTTP error from model server: {error_details}")
                yield f"\n[ERROR: HTTP {e.response.status_code} from model server: {e.response.reason_phrase}]\n"
            except httpx.RequestError as e:
                logger.error(f"ReActAgent: API request to model server failed: {e}")
                yield f"\n[ERROR: Could not connect to the model server: {e}]\n"
            except Exception as e:
                logger.error(f"ReActAgent: An unexpected error occurred during model streaming: {e}", exc_info=True)
                yield f"\n[ERROR: An unexpected error occurred: {e}]\n"
