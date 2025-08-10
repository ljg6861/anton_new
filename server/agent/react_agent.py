# filepath: /home/lucas/anton_new/server/agent/react_agent.py
"""
ReAct (Reason-Act) Agent implementation that handles the complete reasoning and tool-use loop.
Centralized state management through KnowledgeStore, eliminating ConversationState redundancy.
"""
import json
import re
import time
import logging
from urllib import response
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional

from torch import chunk

from server.agent.knowledge_store import KnowledgeStore, ContextType, ImportanceLevel
from server.agent.tool_executor import process_tool_calls
from server.agent.config import SYSTEM_ROLE, ASSISTANT_ROLE, USER_ROLE
from server.agent.learning_loop import learning_loop

logger = logging.getLogger(__name__)


class ReActAgent:
    """
    Single-agent ReAct implementation that combines reasoning and acting in one flow.
    Uses KnowledgeStore for centralized state management and includes direct LLM interaction.
    """
    
    def __init__(self, api_base_url: str, tools: List[Dict], knowledge_store: KnowledgeStore, max_iterations: int = 10, domain_pack_dir: str = '../../learning/packs'):
        self.api_base_url = api_base_url
        self.tools = tools
        self.knowledge_store = knowledge_store
        self.max_iterations = max_iterations
        self.domain_pack_dir = domain_pack_dir
        
    async def get_react_system_prompt(self, user_prompt:str) -> str:
        """Get the system prompt for the ReAct agent with relevant learnings"""
        # Query relevant knowledge from RAG to enhance the system prompt
        relevant_knowledge = self.knowledge_store.query_relevant_knowledge("react agent reasoning tools", max_results=3)
        
        # Get relevant learnings from the learning loop
        relevant_learnings = await learning_loop.get_relevant_learnings(user_prompt or "")
        
        base_prompt = f"""You are Anton, an intelligent, helpful AI assistant that uses the ReAct (Reason-Act) pattern.
        As Anton, you are NOT just an LLM, you actually have a large amount of source code that helps power your responses and your capabilities. 
        You should NEVER tell the user that you are an LLM or that you have limitations. As Anton, it is your responsibility to provide accurate and helpful information to the best of your abilities, and if you identify potential limitations, you should work to find solutions or workarounds and propose these to the user before acting on them.

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
- A large amount of general knowledge. You can answer questions about anything!

Available tools:
{self._format_tools_compact()}

To call a tool, output a JSON object wrapped in tool tags. Do NOT emit the example literally:

Example (do not copy as-is):
&lt;tool_call&gt;
{{
  "name": "read_file",
  "arguments": {{"file_path": "server/agent/react_agent.py"}}
}}
&lt;/tool_call&gt;

Rules:
- Use only ONE tool per turn
- Always wait for tool results before deciding next actions
- Never include multiple tool calls in the same response
- Summarize tool results before providing your final answer
- Use "Final Answer:" when you are ready to reply to the user. Example: User Prompt: "Hello!" Response: "Final Answer: Hello! How can I assist you today?"
- Tool results will be provided as OBSERVATIONS - acknowledge and use them

When a tool completes, you will see an OBSERVATION message. Always process this before continuing.

You MUST always remember that when you are ready to reply to the user, start your response with "Final Answer:"

Always think step by step and be helpful to the user.
"""
        # Add relevant knowledge if available
        if relevant_knowledge:
            base_prompt += "\n\nRelevant past knowledge:\n"
            for knowledge in relevant_knowledge[:2]:
                base_prompt += f"- {knowledge[:200]}...\n"
                
        # Add relevant learnings if available
        if relevant_learnings:
            base_prompt += "\n\nRelevant past learnings and capabilities:\n"
            for learning in relevant_learnings[:3]:
                base_prompt += f"- {learning[:200]}...\n"

        if user_prompt and self.domain_pack_dir:
            bundle = self.knowledge_store.build_domain_knowledge_context(
                query=user_prompt,
                pack_dir=self.domain_pack_dir,
                topk=5,
                expand_radius=1,
                max_nodes=8,
                max_examples_per_node=1
            )
            if bundle:
                base_prompt += "\n\n# Domain knowledge\n"
                base_prompt += (
                    "You have access to the following formal rules and concepts relevant to the user's request.\n"
                    "Use these rules directly when solving; prefer formal definitions over prose.\n\n"
                    + bundle
                )
                
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
                parameters = tool.get('parameters', '{}')
                # Limit description length to avoid bloat
                short_desc = description[:100] + "..." if len(description) > 100 else description
                tool_summaries.append(f"- {name}: {short_desc}, Parameters: {parameters}")
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
            learning_loop.start_task(user_prompt)
        
        # Initialize conversation in knowledge store
        for msg in initial_messages:
            self.knowledge_store.add_message(msg["role"], msg["content"])
        
        # Build messages for the LLM
        messages = self.knowledge_store.get_messages_for_llm()
        
        # Add system prompt
        system_prompt = await self.get_react_system_prompt(user_prompt)
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
            thinking_content = ""
            thinking_started = True
            thinking_ended = False
            answering = False
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
                        yield f'<thought>{response_buffer.split("<think>")[1]}</thought>'
                    
                    # Check for end of thinking block
                    if thinking_started and "</think>" in response_buffer:
                        thinking_ended = True
                        # Extract thinking content
                        think_match = re.search(r'<think>(.*?)</think>', response_buffer, re.DOTALL)
                        if think_match:
                            thinking_content = (pre_think_buffer + think_match.group(1)).strip()
                            if thinking_content:
                                # Yield structured thinking event for Chainlit UI
                                logger.info(f"Agent thinking: {thinking_content}")
                                self.knowledge_store.add_context(
                                    thinking_content,
                                    ContextType.THOUGHT,
                                    ImportanceLevel.MEDIUM,
                                    "react_agent"
                                )
                                # Record thinking in learning loop
                                learning_loop.record_action("thinking", {"content": thinking_content[:200] + "..."})
                        
                        # Get content after thinking block
                        content_after_thinking = response_buffer.rsplit("</think>", 1)[-1]
                        yield f'<token>{content_after_thinking}</token>'
                    else:
                        yield f'<thought>{token}</thought>'
                else:
                    # We're past thinking, accumulate remaining content
                    content_after_thinking = response_buffer.rsplit("</think>", 1)[-1]
                    if answering:
                        yield f'<token>{token}</token>'
                    if 'Final Answer:' in content_after_thinking and not answering:
                        answering = True

            
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
                    # Record thinking in learning loop
                    learning_loop.record_action("thinking", {"content": thinking_content[:200] + "..."})
                    # Yield structured thinking event for Chainlit UI
                    yield f"<thought>{thinking_content}</thought>"
            
            # Extract content after thinking markers and tool code blocks
            content = content_after_thinking or re.split(r'</think>', response_buffer, maxsplit=1)[-1].strip()
            logger.info(f"Content after thinking: {content}")
            # Check if agent made tool calls before yielding the content
            from server.agent import config
            
            # Collect tool results for streaming to UI
            tool_results_for_ui = []
            
            # Create callback to capture actual tool results
            async def tool_result_callback(tool_result_summary):
                """Capture actual tool results for UI streaming"""
                tool_results_for_ui.append(tool_result_summary)
                # Record tool use in learning loop
                learning_loop.record_action("tool_use", {
                    "tool_name": tool_result_summary["name"],
                    "arguments": tool_result_summary["arguments"],
                    "success": tool_result_summary["status"] == "success"
                })
            
            made_tool_calls = await process_tool_calls(
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
                self.knowledge_store.add_context(
                    "Tool calls made",
                    ContextType.ACTION,
                    ImportanceLevel.MEDIUM,
                    "react_agent"
                )
                pass
            
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
                if content.strip().startswith('Final Answer:'):
                    logger.info("Agent provided final response, ending ReAct loop")
                    self.knowledge_store.mark_complete(content)
                    
                    # Complete learning task
                    success = "error" not in content.lower() and "failed" not in content.lower()
                    learning_loop.complete_task(success, content[:200])
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
            learning_loop.complete_task(False, "Reached maximum iterations without completion")
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
            "temperature": 0.6,
            'complex': True,
        }

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", f"{self.api_base_url}/v1/chat/stream", json=request_payload) as response:
                    response.raise_for_status()
                    last_emitted = ""  # track cumulative text we've already emitted
                    async for chunk in response.aiter_text():
                        for raw in chunk.split("\n"):
                            if not raw.strip():
                                continue
                            if raw.startswith("data: "):
                                payload = raw[6:]
                                if payload == "[DONE]":
                                    return
                                # Try JSON; fall back to raw text
                                try:
                                    obj = json.loads(payload)
                                    # common keys: "delta", "content", "message"
                                    piece = obj.get("delta") or obj.get("content") or obj.get("message") or ""
                                except Exception:
                                    piece = payload

                                # If the server sends cumulative text, only emit the new suffix
                                if piece.startswith(last_emitted):
                                    delta = piece[len(last_emitted):]
                                else:
                                    # not cumulative; treat as incremental
                                    delta = piece
                                if delta:
                                    last_emitted = piece if piece.startswith(last_emitted) else last_emitted + delta
                                    yield delta

            except httpx.RequestError as e:
                logger.error(f"ReActAgent: API request to model server failed: {e}")
                yield f"\n[ERROR: Could not connect to the model server: {e}]\n"
            except Exception as e:
                logger.error(f"ReActAgent: An unexpected error occurred during model streaming: {e}", exc_info=True)
                yield f"\n[ERROR: An unexpected error occurred: {e}]\n"