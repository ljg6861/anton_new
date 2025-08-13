"""
ReAct (Reason-Act) Agent: single-loop reasoning, tool use, and response streaming.
Integrates KnowledgeStore, LearningLoop, RAG, and per-user memory.
Uses three-memory architecture with token budgeting.
"""
from __future__ import annotations

import json
import re
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
from dataclasses import dataclass

import httpx

from server.agent.knowledge_store import KnowledgeStore, ContextType, ImportanceLevel
from server.agent.config import SYSTEM_ROLE, ASSISTANT_ROLE, USER_ROLE
from server.agent.tool_executor import process_tool_calls
from server.agent.learning_loop import learning_loop
from server.agent.user_memory import (
    build_context as build_user_context,
    extract_and_update_from_message,
)

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Token budget allocation for different prompt sections"""
    total_budget: int = 8192  # Conservative budget for most models
    system_tools_pct: float = 0.15    # System + tool catalog: ~15%
    domain_bundle_pct: float = 0.30    # Domain knowledge bundle: ~30%  
    session_summary_pct: float = 0.15  # Session memory summary: ~15%
    working_memory_pct: float = 0.35   # Recent messages: ~35%
    scratchpad_pct: float = 0.05       # Reserve for think: ~5%
    
    @property
    def system_tools_budget(self) -> int:
        return int(self.total_budget * self.system_tools_pct)
    
    @property 
    def domain_bundle_budget(self) -> int:
        return int(self.total_budget * self.domain_bundle_pct)
        
    @property
    def session_summary_budget(self) -> int:
        return int(self.total_budget * self.session_summary_pct)
        
    @property
    def working_memory_budget(self) -> int:
        return int(self.total_budget * self.working_memory_pct)
        
    @property
    def scratchpad_budget(self) -> int:
        return int(self.total_budget * self.scratchpad_pct)


class MemoryManager:
    """Manages three types of memory with token budgeting"""
    
    def __init__(self, budget: TokenBudget):
        self.budget = budget
        self.session_decisions = []  # Track key decisions this session
        self.session_todos = []      # Track open TODOs
        self.session_context = ""   # Current session context
        
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars for English)"""
        return len(text) // 4
        
    def truncate_to_budget(self, text: str, budget: int) -> str:
        """Truncate text to fit within token budget"""
        estimated_tokens = self.estimate_tokens(text)
        if estimated_tokens <= budget:
            return text
        
        # Truncate to approximately fit budget
        target_chars = budget * 4
        if len(text) <= target_chars:
            return text
            
        # Truncate at word boundary
        truncated = text[:target_chars]
        last_space = truncated.rfind(' ')
        if last_space > target_chars * 0.8:  # Don't cut too much
            truncated = truncated[:last_space]
        
        return truncated + "..."
        
    def build_working_memory(self, messages: List[Dict[str, str]]) -> str:
        """Build working memory from recent messages (WM)"""
        # Take most recent messages that fit in budget
        budget = self.budget.working_memory_budget
        
        # Start from most recent and work backwards
        selected_messages = []
        total_tokens = 0
        
        for msg in reversed(messages):
            # Preserve original role, don't default to 'user' to avoid confusion
            role = msg.get('role', 'unknown')
            content = f"[{role}]: {msg.get('content', '')}"
            msg_tokens = self.estimate_tokens(content)
            
            if total_tokens + msg_tokens <= budget:
                selected_messages.insert(0, content)
                total_tokens += msg_tokens
            else:
                break
                
        return "\n".join(selected_messages)
        
    def build_session_memory(self) -> str:
        """Build session memory summary (SM)"""
        parts = []
        
        if self.session_context:
            parts.append(f"Context: {self.session_context}")
            
        if self.session_decisions:
            decisions_text = "; ".join(self.session_decisions[-5:])  # Last 5 decisions
            parts.append(f"Decisions: {decisions_text}")
            
        if self.session_todos:
            todos_text = "; ".join(self.session_todos[-3:])  # Last 3 TODOs
            parts.append(f"TODOs: {todos_text}")
            
        session_text = " | ".join(parts)
        return self.truncate_to_budget(session_text, self.budget.session_summary_budget)
        
    def add_decision(self, decision: str):
        """Add a decision to session memory"""
        self.session_decisions.append(decision)
        # Keep only recent decisions to prevent memory bloat
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
        
        # Initialize memory management
        self.budget = token_budget or TokenBudget()
        self.memory = MemoryManager(self.budget)
        
        logger.info(f"ReActAgent initialized with token budget: {self.budget.total_budget}")

    async def get_react_system_prompt(self, user_prompt: str, working_memory: str, session_memory: str) -> str:
        """Compose system prompt with strict token budgeting"""
        
        # Build core system prompt (budgeted)
        base_system = f"""You are Anton, an intelligent AI assistant using ReAct (Reason-Act) pattern.

MEMORY CONSTRAINTS:
- Keep <think> blocks concise (max {self.budget.scratchpad_budget} tokens)
- Focus on the immediate task, rely on provided context

FORMAT:
<think>Brief reasoning about next action</think>

Then either use a tool or provide final response.

CAPABILITIES:
- File operations, code analysis, web search
- Access to domain knowledge and past learnings
- Persistent memory across conversations

TOOLS:
{self._format_tools_compact()}

TOOL USAGE:
&lt;tool_call&gt;{{"name": "tool_name", "arguments": {{"param": "value"}}}}&lt;/tool_call&gt;

RULES:
- ONE tool per turn
- Wait for OBSERVATION before continuing
- Start final responses with "Final Answer:"
"""

        # Truncate base system to fit budget
        system_prompt = self.memory.truncate_to_budget(base_system, self.budget.system_tools_budget)
        
        # Add domain knowledge bundle (LTM retrieval with budget) 
        domain_bundle = ""
        if user_prompt and self.domain_pack_dir:
            selected_pack = self.knowledge_store.select_pack_by_embedding(
                user_prompt, "learning/packs/calc.v1"
            )
            self.domain_pack_dir = selected_pack or self.domain_pack_dir
            bundle = self.knowledge_store.build_domain_knowledge_context(
                query=user_prompt,
                pack_dir=self.domain_pack_dir,
                topk=6,  # Reduced for skinnier format
                expand_radius=1,
                max_nodes=6,  # Using new default from format_context  
                max_examples_per_node=1,
            )
            if bundle:
                domain_bundle = self.memory.truncate_to_budget(bundle, self.budget.domain_bundle_budget)
                
        # Add user profile (part of LTM)
        user_context = build_user_context(self.user_id) if self.user_id else ""
        if user_context:
            user_context = self.memory.truncate_to_budget(user_context, 200)  # Small budget for user context
        
        # Assemble final prompt
        prompt_parts = [system_prompt]
        
        if domain_bundle:
            prompt_parts.append(f"\nDOMAIN KNOWLEDGE:\n{domain_bundle}")
            
        if user_context:
            prompt_parts.append(f"\nUSER PROFILE:\n{user_context}")
            
        if session_memory:
            prompt_parts.append(f"\nSESSION CONTEXT:\n{session_memory}")
            
        return "\n".join(prompt_parts)

    def _format_tools_compact(self) -> str:
        """Format tools in a compact way to reduce prompt bloat"""
        if not self.tools:
            return "No tools available"
        
        # Very compact format to save tokens
        tool_summaries = []
        for tool in self.tools:
            if isinstance(tool, dict):
                name = tool.get("name", "unknown")
                if name == "unknown":
                    fn = tool.get("function", {})
                    name = fn.get("name", "unknown")
                    description = fn.get("description", "No description")
                else:
                    description = tool.get("description", "No description")
                # Ultra-compact: just name and short description
                short_desc = description[:100] + "..." if len(description) > 100 else description
                tool_summaries.append(f"{name}: {short_desc}")
            else:
                tool_summaries.append(str(tool))
        
        tools_text = "; ".join(tool_summaries)
        # Ensure tools fit within allocated budget (part of system budget)
        return self.memory.truncate_to_budget(tools_text, self.budget.system_tools_budget // 3)

    async def process_request(
        self,
        initial_messages: List[Dict[str, str]],
        logger: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Process request using three-memory architecture with token budgeting
        """
        logger.info("Starting ReAct agent with three-memory system...")

        # Extract user prompt and update per-user memory (LTM)
        user_prompt: str = ""
        for msg in reversed(initial_messages):
            if msg.get("role") == USER_ROLE:
                user_prompt = msg.get("content", "")
                break
                
        if self.user_id and user_prompt:
            try:
                extract_and_update_from_message(self.user_id, user_prompt)
            except Exception:
                pass
                
        if user_prompt:
            learning_loop.start_task(user_prompt)

        # Update session context based on user prompt
        if "implement" in user_prompt.lower() or "create" in user_prompt.lower():
            self.memory.set_session_context("Implementation task")
            self.memory.add_todo("Ensure code compiles")
        elif "analyze" in user_prompt.lower() or "review" in user_prompt.lower():
            self.memory.set_session_context("Analysis task") 
        elif "fix" in user_prompt.lower() or "debug" in user_prompt.lower():
            self.memory.set_session_context("Debugging task")
            self.memory.add_todo("Identify root cause")

        # Build three memories with budgets
        working_memory = self.memory.build_working_memory(initial_messages)
        session_memory = self.memory.build_session_memory()
        
        # Initialize conversation in knowledge store for context tracking
        for msg in initial_messages:
            role = msg.get("role")
            if role:  # Only add messages with explicit roles to avoid confusion
                self.knowledge_store.add_message(role, msg.get("content", ""))

        # Build LLM messages with memory architecture
        system_prompt = await self.get_react_system_prompt(user_prompt, working_memory, session_memory)
        react_messages: List[Dict[str, str]] = [{"role": SYSTEM_ROLE, "content": system_prompt}]
        
        # Add working memory as user context (instead of full history)
        if working_memory:
            react_messages.append({"role": SYSTEM_ROLE, "content": f"RECENT CONTEXT:\n{working_memory}"})

        # Add current user message
        if user_prompt:
            react_messages.append({"role": USER_ROLE, "content": user_prompt})

        # Log token usage
        total_tokens = sum(self.memory.estimate_tokens(msg["content"]) for msg in react_messages)
        logger.info(f"Token usage: {total_tokens}/{self.budget.total_budget} "
                   f"({(total_tokens/self.budget.total_budget)*100:.1f}%)")

        # Main ReAct loop
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"ReAct iteration {iteration}/{self.max_iterations}")
            
            response_buffer = ""
            thinking_content = ""
            answering = False
            post_think_output = ""
            suppress_tokens = False

            async for token in self._execute_llm_request(react_messages, logger):
                response_buffer += token
                
                # Handle thinking blocks
                if "<think>" in response_buffer and "</think>" in response_buffer and not thinking_content:
                    think_match = re.search(r"<think>(.*?)</think>", response_buffer, re.DOTALL)
                    if think_match:
                        thinking_content = think_match.group(1).strip()
                        if thinking_content:
                            # Check if thinking exceeds scratchpad budget
                            think_tokens = self.memory.estimate_tokens(thinking_content)
                            if think_tokens > self.budget.scratchpad_budget:
                                logger.warning(f"Thinking block exceeded budget: {think_tokens}/{self.budget.scratchpad_budget}")
                            
                            learning_loop.record_action("thinking", {"content": thinking_content[:200] + "..."})
                            yield f"<thought>{thinking_content}</thought>"
                    
                    # Handle post-think content
                    after = response_buffer.split("</think>", 1)[-1]
                    if after:
                        post_think_output += after
                        if ("<tool_call>" in post_think_output) or ("<tool_code>" in post_think_output):
                            suppress_tokens = True
                        if not suppress_tokens:
                            yield f"<token>{after}</token>"
                        if "Final Answer:" in after:
                            answering = True
                else:
                    # Before thinking block is closed, treat as thought
                    if "</think>" not in response_buffer:
                        yield f"<thought>{token}</thought>"
                    else:
                        # After thinking, accumulate and check for tool calls
                        post_think_output += token
                        if ("<tool_call>" in post_think_output) or ("<tool_code>" in post_think_output):
                            suppress_tokens = True
                        if not suppress_tokens:
                            yield f"<token>{token}</token>"
                        if "Final Answer:" in token:
                            answering = True

            # Extract content after thinking
            content_after_thinking = re.split(r"</think>", response_buffer, maxsplit=1)
            content = content_after_thinking[-1].strip() if content_after_thinking else response_buffer.strip()

            # Tool execution
            from server.agent import config
            tool_results_for_ui: List[Dict[str, Any]] = []

            async def tool_result_callback(tool_result_summary):
                tool_results_for_ui.append(tool_result_summary)
                learning_loop.record_action("tool_use", {
                    "tool_name": tool_result_summary.get("name"),
                    "arguments": tool_result_summary.get("arguments"),
                    "success": tool_result_summary.get("status") == "success",
                })
                
                # Update session memory based on tool results
                tool_name = tool_result_summary.get("name", "")
                if tool_name == "create_file" and tool_result_summary.get("status") == "success":
                    self.memory.add_decision(f"Created file: {tool_result_summary.get('arguments', {}).get('file_path', 'unknown')}")
                elif tool_name == "run_git_command" and "checkout -b" in str(tool_result_summary.get("arguments", {})):
                    self.memory.add_decision("Created new git branch")
                    self.memory.complete_todo("Create feature branch")

            made_tool_calls = await process_tool_calls(
                content, config.TOOL_CALL_REGEX, react_messages, logger, self.knowledge_store, tool_result_callback
            )

            for tool_result_summary in tool_results_for_ui:
                yield f"<tool_result>{json.dumps(tool_result_summary)}</tool_result>"

            # Record response and decide next action
            react_messages.append({"role": ASSISTANT_ROLE, "content": content})

            if made_tool_calls:
                continue
            else:
                if content.strip().startswith("Final Answer:") or answering:
                    success = "error" not in content.lower() and "failed" not in content.lower()
                    learning_loop.complete_task(success, content[:200])
                    
                    # Update session memory with completion
                    if success:
                        self.memory.add_decision("Task completed successfully")
                    else:
                        self.memory.add_decision("Task completed with errors")
                    break
                else:
                    continue

        if iteration >= self.max_iterations:
            final_msg = "\n\n[Task completed - reached maximum iterations]"
            learning_loop.complete_task(False, "Reached maximum iterations")
            self.memory.add_decision("Task incomplete - reached max iterations")
            yield final_msg

    async def _execute_llm_request(
        self, messages: List[Dict[str, str]], logger: Any
    ) -> AsyncGenerator[str, None]:
        """Execute LLM request directly and stream tokens from the model server."""
        request_payload = {"messages": messages, "temperature": 0.6, "complex": True}

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST", f"{self.api_base_url}/v1/chat/stream", json=request_payload
                ) as response:
                    response.raise_for_status()
                    last_emitted = ""
                    async for chunk in response.aiter_text():
                        for raw in chunk.split("\n"):
                            if not raw.strip():
                                continue
                            if raw.startswith("data: "):
                                payload = raw[6:]
                                if payload == "[DONE]":
                                    return
                                try:
                                    obj = json.loads(payload)
                                    piece = obj.get("delta") or obj.get("content") or obj.get("message") or ""
                                except Exception:
                                    piece = payload

                                if piece.startswith(last_emitted):
                                    delta = piece[len(last_emitted) :]
                                else:
                                    delta = piece
                                if delta:
                                    last_emitted = piece if piece.startswith(last_emitted) else last_emitted + delta
                                    yield delta
            except httpx.RequestError as e:
                logger.error(f"ReActAgent: API request to model server failed: {e}")
                yield f"\n[ERROR: Could not connect to the model server: {e}]\n"
            except Exception as e:
                logger.error(
                    f"ReActAgent: An unexpected error occurred during model streaming: {e}", exc_info=True
                )
                yield f"\n[ERROR: An unexpected error occurred: {e}]\n"