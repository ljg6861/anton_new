"""
ReAct (Reason-Act) Agent: single-loop reasoning, tool use, and response streaming.
Refactored to use structured state management following the make_state/react_run_with_state pattern.
Integrates KnowledgeStore, LearningLoop, RAG, and per-user memory.
"""
import json
import re
import time
import logging, os, re, json, httpx
from urllib import response
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional

from server.agent.knowledge_store import KnowledgeStore, ContextType, ImportanceLevel
from server.agent.tool_executor import process_tool_calls
from server.agent.config import SYSTEM_ROLE, ASSISTANT_ROLE, USER_ROLE
from server.agent.learning_loop import learning_loop
from server.agent.state import State, make_state, enforce_budgets, Budgets, AgentStatus
from server.agent.state_ops import (
    build_working_memory, build_session_memory, build_context_summary,
    format_thought_action_observation, update_state_from_tool_result,
    parse_action_from_response, is_final_response, update_session_context_from_goal,
    get_budget_status, summarize_execution, truncate_to_budget
)

logger = logging.getLogger(__name__)
MD_DEBUG = os.getenv("ANTON_MD_DEBUG", "0") == "1"

class TokenLoopDetector:
    """Detects token-level repetition and loops in model output"""
    
    def __init__(self, window_size: int = 50, min_phrase_length: int = 4, repeat_threshold: int = 3):
        self.window_size = window_size
        self.min_phrase_length = min_phrase_length
        self.repeat_threshold = repeat_threshold
        self.token_buffer = []
        self.phrase_counts = {}
        self.last_warning_position = -1
        
    def add_token(self, token: str) -> bool:
        """
        Add a token and check for loops. Returns True if loop detected.
        """
        self.token_buffer.append(token.strip())
        
        # Keep only recent tokens
        if len(self.token_buffer) > self.window_size:
            self.token_buffer.pop(0)
            
        # Only check for loops if we have enough tokens
        if len(self.token_buffer) < self.min_phrase_length * 2:
            return False
            
        # Check for repeated phrases
        return self._detect_phrase_repetition()
    
    def _detect_phrase_repetition(self) -> bool:
        """Detect if phrases are repeating"""
        # Create phrases of different lengths
        for phrase_len in range(self.min_phrase_length, min(len(self.token_buffer) // 2, 15)):
            phrases = []
            
            # Extract overlapping phrases
            for i in range(len(self.token_buffer) - phrase_len + 1):
                phrase = " ".join(self.token_buffer[i:i + phrase_len])
                phrases.append(phrase)
            
            # Count phrase occurrences in recent window
            phrase_counts = {}
            recent_phrases = phrases[-20:]  # Look at last 20 phrases
            
            for phrase in recent_phrases:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
                
            # Check if any phrase repeats too often
            for phrase, count in phrase_counts.items():
                if count >= self.repeat_threshold and len(phrase.strip()) > 10:
                    # Avoid warning about the same phrase repeatedly
                    current_pos = len(self.token_buffer)
                    if current_pos - self.last_warning_position > 20:
                        self.last_warning_position = current_pos
                        logger.warning(f"Loop detected: phrase '{phrase[:50]}...' repeated {count} times")
                        return True
                        
        return False
    
    def reset(self):
        """Reset the detector for a new request"""
        self.token_buffer.clear()
        self.phrase_counts.clear()
        self.last_warning_position = -1


class ReActAgent:
    """
    Single-agent ReAct implementation with structured state management.
    Follows the react_run_with_state pattern for clean separation of concerns.
    """

    def __init__(
        self,
        api_base_url: str,
        tools: List[Dict],
        knowledge_store: KnowledgeStore,
        max_iterations: int = 10,
        domain_pack_dir: str = "../../learning/packs",
        user_id: Optional[str] = None,
        budgets: Optional[Budgets] = None,
    ) -> None:
        self.api_base_url = api_base_url
        self.tools = tools
        self.knowledge_store = knowledge_store
        self.domain_pack_dir = domain_pack_dir
        self.user_id = user_id or "anonymous"
        
        # Initialize default budgets if none provided
        self.default_budgets = budgets or Budgets(max_iterations=max_iterations)
        
        # Initialize loop detection
        self.loop_detector = TokenLoopDetector()
        
        logger.info(f"ReActAgent initialized with token budget: {self.default_budgets.total_tokens}")

    def react_run_with_state(self, goal: str, budgets: Optional[Budgets] = None) -> State:
        """
        Main entry point following the structured state pattern.
        This is a synchronous wrapper that will be called by the async process_request.
        """
        S = make_state(goal, budgets or self.default_budgets, self.user_id)
        update_session_context_from_goal(S)
        
        # Initialize learning loop
        if goal:
            learning_loop.start_task(goal)
        
        return S

    async def get_react_system_prompt(self, state: State, user_prompt: str) -> str:
        """Compose system prompt with structured state and strict token budgeting"""
        
        # Build working and session memory from state
        working_memory = build_working_memory(state)
        session_memory = build_session_memory(state)
        
        # Build core system prompt (budgeted)
        base_system = f"""
You are Anton, an intelligent AI assistant using the ReAct (Reason-Act) pattern. You are not just an LLM; you are the core interface for a larger system with robust source code and git capabilities. You represent this entire system.

You are INCREDIBLY thorough and confident. You never second guess yourself, but your confidence is a direct result of your meticulous research.

Your primary directive is to always perform research before providing an answer. This is a non-negotiable step. If a user were to ask you to write code, you would not just start coding; you would first use your git capabilities to understand the existing code base, identify similar files, and then act based on a complete understanding.

If you cannot find a definitive answer after thorough research, you will confidently explain what you have investigated and why a conclusive answer isn't possible. You will never invent, guess, or hallucinate information. Your responses should demonstrate the care and depth of your research.

MEMORY CONSTRAINTS:
- Keep <think> blocks concise (max {state.budgets.scratchpad_budget} tokens)
- Focus on the immediate task, rely on provided context

FORMAT:
<think>Brief reasoning about next action</think>

Then either use a tool or provide final response.

CAPABILITIES:
- File operations, code analysis, web search
- Access to domain knowledge and past learnings
- Persistent memory across conversations
- Large amount of general knowledge. You can answer questions about anything.
- Anything else mentioned in the following tools

TOOLS:
{self._format_tools_compact(state)}

TOOL USAGE:
&lt;tool_call&gt;{{"name": "tool_name", "arguments": {{"param": "value"}}}}&lt;/tool_call&gt;

RULES:
- ONE tool per turn
- Following a successful tool call, an OBSERVATION will be provided to you.
- If doing a coding task, before you start you MUST ensure you are not on the master branch. If you are, you must create a new branch using the schema: anton/<short_feature_name>
- Start your final response to the user with "Final Answer:"
- You may ONLY use markdown when giving your final answer.
- When using markdown ensure that you are always using triple backticks (```) to start and end the code block, and specify the language.
Example: 
```python **Ensure there is a new line here!**
print("Hello, World!") **Ensure there is a new line here!**
```

"""

        # Truncate base system to fit budget
        system_prompt = truncate_to_budget(base_system, state.budgets.system_tools_budget)
        
        # Add domain knowledge bundle (LTM retrieval with budget)
        domain_bundle = ""
        if user_prompt and self.domain_pack_dir:
            selected_pack = self.knowledge_store.select_pack_by_embedding(
                user_prompt, "learning/packs/calc.v1"
            )
            bundle = self.knowledge_store.build_domain_knowledge_context(
                query=user_prompt,
                pack_dir=selected_pack,
                topk=3,  # Reduced for budget
                expand_radius=1,
                max_nodes=8,
                max_examples_per_node=1
            )
            if bundle:
                domain_bundle = truncate_to_budget(bundle, state.budgets.domain_bundle_budget)
                
        # Add user profile (part of LTM)
        user_context = None #build_user_context(self.user_id) if self.user_id else ""
        if user_context:
            user_context = truncate_to_budget(user_context, 200)  # Small budget for user context
        
        # Assemble final prompt
        prompt_parts = [system_prompt]
        
        if domain_bundle:
            prompt_parts.append(f"\nDOMAIN KNOWLEDGE:\n{domain_bundle}")
            
        if user_context:
            prompt_parts.append(f"\nUSER PROFILE:\n{user_context}")
            
        if session_memory:
            prompt_parts.append(f"\nSESSION CONTEXT:\n{session_memory}")
            
        return "\n".join(prompt_parts)

    def _format_tools_compact(self, state: State) -> str:
        """Format tools in a compact way to reduce prompt bloat"""
        if not self.tools:
            return "No tools available"
        
        # Very compact format to save tokens
        tool_summaries = []
        for tool in self.tools:
            if isinstance(tool, dict):
                name = tool.get('name', 'unknown')
                if name == 'unknown':
                    tool = tool.get('function')
                    name = tool.get('name', 'unknown')
                description = tool.get('description', 'No description')
                parameters = tool.get('parameters', '{}')
                # Limit description length to avoid bloat
                short_desc = description[:100] + "..." if len(description) > 100 else description
                tool_summaries.append(f"- {name}: {short_desc}, Parameters: {parameters}")
            else:
                tool_summaries.append(str(tool))
        
        tools_text = "; ".join(tool_summaries)
        # Ensure tools fit within allocated budget (part of system budget)
        return truncate_to_budget(tools_text, state.budgets.system_tools_budget // 3)

    async def process_request(
        self,
        initial_messages: List[Dict[str, str]],
        logger: Any
    ) -> AsyncGenerator[str, None]:
        """
        Process request using structured state management following react_run_with_state pattern
        """
        logger.info("Starting ReAct agent with structured state system...")

        # Extract user prompt (goal)
        user_prompt: str = ""
        for msg in reversed(initial_messages):
            if msg.get("role") == USER_ROLE:
                user_prompt = msg.get("content", "")
                break
        
        if not user_prompt:
            logger.error("No user prompt found in messages")
            yield "Error: No user prompt provided"
            return

        # Initialize structured state
        S = self.react_run_with_state(user_prompt, self.default_budgets)
        
        # Add initial messages to state working memory
        for msg in initial_messages:
            role = msg.get("role")
            if role:
                S.working_memory.append(msg)
                self.knowledge_store.add_message(role, msg.get("content", ""))

        # Build LLM messages with memory architecture from state
        system_prompt = await self.get_react_system_prompt(S, user_prompt)
        react_messages: List[Dict[str, str]] = [{"role": SYSTEM_ROLE, "content": system_prompt}]
        
        # Add working memory as user context
        working_memory = build_working_memory(S)
        if working_memory:
            react_messages.append({"role": SYSTEM_ROLE, "content": f"RECENT CONTEXT:\n{working_memory}"})

        # Add current user message
        react_messages.append({"role": USER_ROLE, "content": user_prompt})

        # Log token usage and budget status
        from server.agent.state_ops import estimate_tokens
        total_tokens = sum(estimate_tokens(msg["content"]) for msg in react_messages)
        logger.info(f"Token usage: {total_tokens}/{S.budgets.total_tokens} "
                   f"({(total_tokens/S.budgets.total_tokens)*100:.1f}%)")
        
        budget_status = get_budget_status(S)
        logger.info(f"Budget status: {budget_status}")

        # Main ReAct loop following the structured state pattern
        while S.status not in [AgentStatus.DONE, AgentStatus.FAILED]:
            S.increment_turn()
            logger.info(f"ReAct iteration {S.turns}/{S.budgets.max_iterations}")
            
            # Check budgets before continuing
            if not enforce_budgets(S):
                logger.warning(f"Budget exceeded: {get_budget_status(S)}")
                final_msg = "\n\n[Task incomplete - budget exceeded]"
                learning_loop.complete_task(False, "Budget exceeded")
                yield final_msg
                break
            
            # Add context about current iteration
            S.add_context(
                f"Starting ReAct iteration {S.turns}",
                "react_agent",
                1.0,
                "iteration_start"
            )
            
            S.set_status(AgentStatus.PLANNING)
            
            # Execute LLM request and process response
            response_buffer = ""
            thinking_content = ""
            thinking_started = False
            thinking_ended = False
            answering = False
            
            async for token in self._execute_llm_request(react_messages, logger):
                response_buffer += token
                
                # Incremental parsing for better streaming
                if not thinking_ended:
                    # Check for start of thinking block
                    if not thinking_started and "<think>" in response_buffer:
                        thinking_started = True
                        pre_think = response_buffer.split("<think>")[0]
                        think_part = response_buffer.split("<think>")[1]
                        yield f'<thought>{think_part}</thought>'
                    
                    # Check for end of thinking block
                    if thinking_started and "</think>" in response_buffer:
                        thinking_ended = True
                        # Extract thinking content
                        think_match = re.search(r'<think>(.*?)</think>', response_buffer, re.DOTALL)
                        if think_match:
                            thinking_content = think_match.group(1).strip()
                            if thinking_content:
                                # Add thinking to state and scratchpad
                                S.add_context(thinking_content, "llm_thinking", 2.0, "thought")
                                S.add_to_scratchpad(f"Thought: {thinking_content}")
                                
                                logger.info(f"Agent thinking: {thinking_content}")
                                learning_loop.record_action("thinking", {"content": thinking_content[:200] + "..."})
                        
                        # Get content after thinking block
                        content_after_thinking = response_buffer.rsplit("</think>", 1)[-1]
                        yield f'<token>{content_after_thinking}</token>'
                    elif thinking_started:
                        yield f'<thought>{token}</thought>'
                    else:
                        yield f'<token>{token}</token>'
                else:
                    # We're past thinking, stream remaining content
                    yield f'<token>{token}</token>'
                    if 'Final Answer:' in response_buffer and not answering:
                        answering = True

            # Process the complete response
            logger.info(f"ReAct agent response: {response_buffer}")
            
            # Extract content after thinking markers
            content = re.split(r'</think>', response_buffer, maxsplit=1)[-1].strip()
            logger.info(f"Content after thinking: {content}")
            
            # Check for tool calls
            S.set_status(AgentStatus.EXECUTING)
            
            # Tool result callback to update state
            tool_results_for_ui = []
            
            async def tool_result_callback(tool_result_summary):
                """Capture tool results and update state"""
                tool_results_for_ui.append(tool_result_summary)
                
                # Update state with tool result
                tool_name = tool_result_summary.get("name", "")
                tool_args = tool_result_summary.get("arguments", {})
                result = tool_result_summary.get("result", "")
                success = tool_result_summary.get("status") == "success"
                error = tool_result_summary.get("error")
                
                # Find and complete the tool call trace
                update_state_from_tool_result(S, tool_name, tool_args, result, success, 0.0, error)
                
                # Record in learning loop
                learning_loop.record_action("tool_use", {
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "success": success,
                })
                
                # Add observation to scratchpad
                observation = f"Tool {tool_name} {'succeeded' if success else 'failed'}: {str(result)[:200]}"
                S.add_to_scratchpad(f"Action: {tool_name}\nObservation: {observation}")

            # Process tool calls
            from server.agent import config
            made_tool_calls = await process_tool_calls(
                content, 
                config.TOOL_CALL_REGEX,
                react_messages,
                logger,
                self.knowledge_store,
                tool_result_callback
            )
            
            # Stream tool results to UI
            for tool_result_summary in tool_results_for_ui:
                import json
                yield f"<tool_result>{json.dumps(tool_result_summary)}</tool_result>"

            # Record response and decide next action
            react_messages.append({"role": ASSISTANT_ROLE, "content": content})
            
            if made_tool_calls:
                # Continue the loop for next iteration
                continue
            else:
                # No tool calls - check if this is a final answer
                if is_final_response(content) or answering:
                    S.set_status(AgentStatus.DONE)
                    success = "error" not in content.lower() and "failed" not in content.lower()
                    learning_loop.complete_task(success, content[:200])
                    
                    # Update state with completion
                    if success:
                        S.add_decision("Task completed successfully")
                    else:
                        S.add_decision("Task completed with errors")
                    
                    # Add final response to state
                    S.add_context(content, "final_response", 3.0, "final_answer")
                    break
                else:
                    # Response doesn't appear final, continue
                    logger.info("Response doesn't appear final, continuing...")
                    continue
        
        # Handle completion or failure
        if S.status == AgentStatus.FAILED or S.turns >= S.budgets.max_iterations:
            if S.turns >= S.budgets.max_iterations:
                logger.warning(f"ReAct agent reached max iterations ({S.budgets.max_iterations})")
                final_msg = "\n\n[Task completed - reached maximum iterations]"
                learning_loop.complete_task(False, "Reached maximum iterations")
                S.add_decision("Task incomplete - reached max iterations")
            else:
                final_msg = "\n\n[Task failed - budget exceeded]"
            yield final_msg
        
        # Log final state summary
        final_summary = summarize_execution(S)
        logger.info(f"Final execution summary:\n{final_summary}")
    
    async def _execute_llm_request(
        self,
        messages: List[Dict[str, str]],
        logger: Any
    ) -> AsyncGenerator[str, None]:
        """
        Execute LLM request directly, replacing dependency on doer.py
        Tools parameter removed as it's unused by Ollama - tools are described in system prompt instead.
        """
        request_payload = {
            "messages": messages,
            "temperature": 0.6,
            'complex': True,
        }

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                # Assuming self.api_base_url is defined elsewhere
                async with client.stream("POST", f"{self.api_base_url}/v1/chat/stream", json=request_payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_text():
                        chunk = json.loads(chunk)
                        yield chunk.get("message").get("content")

            except httpx.RequestError as e:
                logger.error(f"ReActAgent: API request to model server failed: {e}")
                yield f"\n[ERROR: Could not connect to the model server: {e}]\n"
            except Exception as e:
                logger.error(f"ReActAgent: An unexpected error occurred during model streaming: {e}", exc_info=True)
                yield f"\n[ERROR: An unexpected error occurred: {e}]\n"