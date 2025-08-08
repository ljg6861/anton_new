"""
ReAct (Reason-Act) Agent implementation that handles the complete reasoning and tool-use loop.
Centralized state management through KnowledgeStore, eliminating ConversationState redundancy.
"""
import json
import re
import time
import logging
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional

from server.agent.knowledge_store import KnowledgeStore, ContextType, ImportanceLevel
from server.agent.tool_executor import process_tool_calls
from server.agent.config import SYSTEM_ROLE, ASSISTANT_ROLE, USER_ROLE

logger = logging.getLogger(__name__)


class ReActAgent:
    """
    Single-agent ReAct implementation that combines reasoning and acting in one flow.
    Uses KnowledgeStore for centralized state management and includes direct LLM interaction.
    """
    
    def __init__(self, api_base_url: str, tools: List[Dict], knowledge_store: KnowledgeStore, max_iterations: int = 10):
        self.api_base_url = api_base_url
        self.tools = tools
        self.knowledge_store = knowledge_store
        self.max_iterations = max_iterations
        
    def get_react_system_prompt(self) -> str:
        """Get the system prompt for the ReAct agent"""
        # Query relevant knowledge from RAG to enhance the system prompt
        relevant_knowledge = self.knowledge_store.query_relevant_knowledge("react agent reasoning tools", max_results=3)
        
        base_prompt = f"""You are Anton, an intelligent AI assistant that uses the ReAct (Reason-Act) pattern.

You can REASON about problems, take ACTIONS using tools, and provide RESPONSES to users.

For each turn, you should:
1. REASON: Think about what you need to do
2. ACT: Use tools if needed to gather information or perform actions  
3. RESPOND: Provide a helpful response to the user

Format your responses as:

<think>
Your reasoning about what to do next...
</think>

Then either:
- Use a tool if you need to gather information or perform an action
- Provide a direct response if you have enough information

You have access to these capabilities:
- File operations (read, write, list directories)
- Access to your own source code via the file operations and embeddings
- Persistent memory, including the ability to recall parts of past interactions using RAG
- Code analysis and search
- Web search 
- Knowledge retrieval

Available tools:
{self._format_tools_compact()}

You can call these tools using the following format:\n"""

        base_prompt += """
<tool_code>
{"name" : "tool name", "arguments" : {"arg1" : "arg1_value", "arg2" : "arg2_value"}}
</tool_code>

IMPORTANT TOOL USAGE RULES:
- Use only ONE tool per turn to avoid dependency issues
- Always wait for tool results before deciding on next actions
- Never include multiple tool calls in the same response
- Summarize tool results before providing your final answer
- Use "Final Answer:" when you are completely done with the task
- Tool results will be provided as OBSERVATIONS - acknowledge and use them

When a tool completes, you will see an OBSERVATION message. Always process this before continuing.

Always think step by step and be helpful to the user."""

        # Add relevant knowledge if available
        if relevant_knowledge:
            base_prompt += "\n\nRelevant past knowledge:\n"
            for knowledge in relevant_knowledge[:2]:  # Limit to avoid prompt bloat
                base_prompt += f"- {knowledge[:200]}...\n"
                
        return base_prompt
    
    def _format_tools_compact(self) -> str:
        """Format tools in a compact way to reduce prompt bloat"""
        if not self.tools:
            return "No tools available"
        
        tool_summaries = []
        for tool in self.tools:
            if isinstance(tool, dict):
                name = tool.get('name', 'unknown')
                if name == 'unknown':
                    tool = tool.get('function')
                    name = tool.get('name', 'unknown')
                description = tool.get('description', 'No description')
                # Limit description length to avoid bloat
                short_desc = description[:100] + "..." if len(description) > 100 else description
                tool_summaries.append(f"- {name}: {short_desc}")
            else:
                tool_summaries.append(f"- {str(tool)}")
        
        return "\n".join(tool_summaries)

    async def process_request(
        self,
        initial_messages: List[Dict[str, str]],
        logger: Any
    ) -> AsyncGenerator[str, None]:
        """
        Process a request using the ReAct pattern with KnowledgeStore for state management.
        Handles the complete reasoning and tool-use loop without external dependencies.
        Integrates learning loop for experience tracking and improvement.
        """
        logger.info("Starting ReAct agent processing...")
        
        # Start learning task tracking
        user_prompt = None
        for msg in initial_messages:
            if msg["role"] == "user":
                user_prompt = msg["content"]
                break
        
        if user_prompt:
            self.knowledge_store.start_learning_task(user_prompt)
        
        # Initialize conversation in knowledge store
        for msg in initial_messages:
            self.knowledge_store.add_message(msg["role"], msg["content"])
        
        # Build messages for the LLM
        messages = self.knowledge_store.get_messages_for_llm()
        
        # Add system prompt
        system_prompt = self.get_react_system_prompt()
        react_messages = [{"role": SYSTEM_ROLE, "content": system_prompt}]
        
        # Add context if available
        context_summary = self.knowledge_store.build_context_summary()
        if context_summary and context_summary != "No significant context yet.":
            react_messages.append({
                "role": SYSTEM_ROLE, 
                "content": f"Current context:\n{context_summary}"
            })
        
        # Add conversation history
        react_messages.extend(messages)
        
        # Track iterations to prevent infinite loops
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
            
            # Get response from LLM with incremental streaming and parsing
            response_buffer = ""
            start_time = time.time()
            thinking_content = ""
            thinking_started = False
            thinking_ended = False
            content_after_thinking = ""
            pre_think_buffer = ""
            
            async for token in self._execute_llm_request(react_messages, logger):
                response_buffer += token
                
                # Incremental parsing for better streaming
                if not thinking_ended:
                    # Check for start of thinking block
                    if not thinking_started and "<think>" in response_buffer:
                        thinking_started = True
                        # Extract any content before <think>
                        before_think = response_buffer.split("<think>")[0]
                        if before_think:
                            pre_think_buffer += before_think
                    
                    # Check for end of thinking block
                    if thinking_started and "</think>" in response_buffer:
                        thinking_ended = True
                        # Extract thinking content
                        think_match = re.search(r'<think>(.*?)</think>', response_buffer, re.DOTALL)
                        if think_match:
                            thinking_content = (pre_think_buffer + think_match.group(1)).strip()
                            if thinking_content:
                                # Yield structured thinking event for Chainlit UI
                                yield f"<thought>{thinking_content}</thought>"
                                logger.info(f"Agent thinking: {thinking_content}")
                                self.knowledge_store.add_context(
                                    thinking_content,
                                    ContextType.THOUGHT,
                                    ImportanceLevel.MEDIUM,
                                    "react_agent"
                                )
                        
                        # Get content after thinking block
                        content_after_thinking = response_buffer.rsplit("</think>", 1)[-1]
                else:
                    # We're past thinking, accumulate remaining content
                    content_after_thinking = response_buffer.rsplit("</think>", 1)[-1] if "</think>" in response_buffer else response_buffer
            
            # Process the complete response
            logger.info(f"ReAct agent response: {response_buffer}")
            
            # Use extracted thinking content if available, otherwise try to extract it
            if not thinking_content:
                thinking_match = re.search(r'<think>(.*?)</think>', response_buffer, re.DOTALL)
                if thinking_match:
                    thinking_content = (pre_think_buffer + thinking_match.group(1)).strip()
                    self.knowledge_store.add_context(
                        thinking_content,
                        ContextType.THOUGHT,
                        ImportanceLevel.MEDIUM,
                        "react_agent"
                    )
                    logger.info(f"Agent thinking: {thinking_content}")
                    # Yield structured thinking event for Chainlit UI
                    yield f"<thought>{thinking_content}</thought>"
            
            # Extract content after thinking markers and tool code blocks
            content = content_after_thinking or re.split(r'</think>', response_buffer, maxsplit=1)[-1].strip()
            
            # Check if agent made tool calls before yielding the content
            from server.agent import config
            
            # Collect tool results for streaming to UI
            tool_results_for_ui = []
            
            # Create callback to capture actual tool results
            async def tool_result_callback(tool_result_summary):
                """Capture actual tool results for UI streaming"""
                tool_results_for_ui.append(tool_result_summary)
            
            made_tool_calls = await process_tool_calls_with_knowledge_store(
                content, 
                config.TOOL_CALL_REGEX,  # Use existing tool call regex
                react_messages,
                logger,
                self.knowledge_store,
                tool_result_callback
            )
            
            # Stream captured tool results to UI
            for tool_result_summary in tool_results_for_ui:
                import json
                yield f"<tool_result>{json.dumps(tool_result_summary)}</tool_result>"
            
            if made_tool_calls:
                # Record tool actions in learning loop
                for tool_result_summary in tool_results_for_ui:
                    self.knowledge_store.add_learning_action(
                        "tool_execution", 
                        {
                            "tool_name": tool_result_summary["name"],
                            "arguments": tool_result_summary["arguments"],
                            "status": tool_result_summary["status"],
                            "result_preview": tool_result_summary["brief_result"][:100]
                        }
                    )
                
                # Remove tool code blocks from content before yielding clean response
                clean_content = config.TOOL_CALL_REGEX.sub('', content).strip()
                if clean_content:
                    # Yield clean content as tokens
                    for char in clean_content:
                        yield f"<token>{char}</token>"
            else:
                # No tool calls - yield the clean content as tokens
                for char in content:
                    yield f"<token>{char}</token>"
            
            # Add to conversation (use original content with tool calls for internal tracking)
            self.knowledge_store.add_message(ASSISTANT_ROLE, content)
            react_messages.append({"role": ASSISTANT_ROLE, "content": content})
            
            if made_tool_calls:
                # If tools were called, continue the loop to let agent process results
                logger.info("Tools were executed, continuing ReAct loop...")
                self.knowledge_store.add_context(
                    "Tool calls completed, processing results...",
                    ContextType.ACTION,
                    ImportanceLevel.MEDIUM,
                    "react_agent"
                )
                continue
            else:
                # No tool calls, check if this looks like a final response
                if self._is_final_response(content):
                    logger.info("Agent provided final response, ending ReAct loop")
                    self.knowledge_store.mark_complete(content)
                    
                    # Complete learning task
                    success = "error" not in content.lower() and "failed" not in content.lower()
                    self.knowledge_store.complete_learning_task(success, content[:200])
                    break
                else:
                    # Agent might need another iteration to complete the task
                    logger.info("Response doesn't appear final, continuing...")
                    continue
        
        if iteration >= self.max_iterations:
            logger.warning(f"ReAct agent reached max iterations ({self.max_iterations})")
            final_msg = "\n\n[Task completed - reached maximum iterations]"
            self.knowledge_store.add_context(
                "Reached maximum iterations",
                ContextType.THOUGHT,
                ImportanceLevel.HIGH,
                "react_agent",
                {"reason": "max_iterations"}
            )
            
            # Complete learning task as partially successful (timeout)
            self.knowledge_store.complete_learning_task(False, "Reached maximum iterations without completion")
            yield final_msg
    
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
            "temperature": 0.7,
            'complex': True,
        }

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", f"{self.api_base_url}/v1/chat/stream", json=request_payload) as response:
                    response.raise_for_status()
                    # Parse SSE format (data: prefix)
                    async for chunk in response.aiter_text():
                        for line in chunk.split('\n'):
                            if line.startswith('data: '):
                                content = line[6:]  # Remove 'data: ' prefix
                                if content == '[DONE]':
                                    return
                                yield content
                            elif line.strip():
                                # Fallback for non-SSE format
                                yield line
            except httpx.RequestError as e:
                logger.error(f"ReActAgent: API request to model server failed: {e}")
                yield f"\n[ERROR: Could not connect to the model server: {e}]\n"
            except Exception as e:
                logger.error(f"ReActAgent: An unexpected error occurred during model streaming: {e}", exc_info=True)
                yield f"\n[ERROR: An unexpected error occurred: {e}]\n"
    
    def _is_final_response(self, content: str) -> bool:
        """
        Determines if the agent's response is final using explicit finality markers.
        Now defaults to NOT final unless explicit indicators are found.
        """
        content_lower = content.lower().strip()

        if not content:
            return False

        # --- Rule 1: Check for explicit final answer markers ---
        # These are strong indicators that the task is finished.
        final_markers = [
            'final answer:',
            '<final_answer>',
            'task completed',
            'i have finished',
            'done.',
            'that completes'
        ]
        if any(marker in content_lower for marker in final_markers):
            return True

        # --- Rule 2: Check for conversational closing statements ---
        # If these are present, the response IS final.
        closing_signals = [
            'let me know if you need anything else',
            'is there anything else',
            'how else can i help',
            'hope that helps',
            'feel free to ask'
        ]
        if any(signal in content_lower for signal in closing_signals):
            return True

        # --- Rule 3: Check for explicit signs of continuation or tool use ---
        # If these are present, the response is definitely NOT final.
        continuation_signals = [
            '<tool_call>',
            '<tool_code>',
            'i need to use',
            'i will now',
            'the next step is to',
            'let me first',
            'let me check',
            'i should',
            'i need to'
        ]
        if any(signal in content_lower for signal in continuation_signals):
            return False

        # --- Rule 4: Default to NOT final unless explicit finality is indicated ---
        # This is the key change - be conservative and continue unless clearly final
        return False


async def process_tool_calls_with_knowledge_store(
    content: str,
    tool_call_regex,  # Already compiled regex from config
    messages: List[Dict],
    logger: Any,
    knowledge_store: KnowledgeStore,
    result_callback = None
) -> bool:
    """
    Process tool calls in the agent's response using KnowledgeStore for state management.
    """
    # Use existing tool executor directly with knowledge_store
    return await process_tool_calls(
        content,
        tool_call_regex,  # Pass the compiled regex directly
        messages,
        logger,
        knowledge_store,
        result_callback
    )