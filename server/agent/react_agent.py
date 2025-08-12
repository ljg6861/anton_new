"""
ReAct (Reason-Act) Agent: single-loop reasoning, tool use, and response streaming.
Integrates KnowledgeStore, LearningLoop, RAG, and per-user memory.
"""
from __future__ import annotations

import json
import re
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional

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


class ReActAgent:
    """
    Single-agent ReAct implementation that combines reasoning and acting in one flow.
    Uses KnowledgeStore for centralized state management and includes direct LLM interaction.
    """

    def __init__(
        self,
        api_base_url: str,
        tools: List[Dict],
        knowledge_store: KnowledgeStore,
        max_iterations: int = 10,
        domain_pack_dir: str = "../../learning/packs",
        user_id: Optional[str] = None,
    ) -> None:
        self.api_base_url = api_base_url
        self.tools = tools
        self.knowledge_store = knowledge_store
        self.max_iterations = max_iterations
        self.domain_pack_dir = domain_pack_dir
        self.user_id = user_id or "anonymous"

    async def get_react_system_prompt(self, user_prompt: str) -> str:
        """Compose the system prompt including capabilities, tools, learnings, RAG, and user profile."""
        # Per-user memory context
        user_context = build_user_context(self.user_id) if self.user_id else ""

        # Query relevant knowledge from RAG to enhance the system prompt
        relevant_knowledge = self.knowledge_store.query_relevant_knowledge(
            "react agent reasoning tools", max_results=3
        )

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

Coding Rules:
- Under NO circumstances are you to do any coding tasks before checking out to a new branch. Call this branch feature/<feature_name>
- You MUST ensure that your code compiles

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
        if user_context:
            base_prompt += "\n\n# User profile\n" + user_context + "\n"

        if relevant_knowledge:
            base_prompt += "\n\nRelevant past knowledge:\n"
            for knowledge in relevant_knowledge[:2]:
                base_prompt += f"- {knowledge[:200]}...\n"

        if relevant_learnings:
            base_prompt += "\n\nRelevant past learnings and capabilities:\n"
            for learning in relevant_learnings[:3]:
                base_prompt += f"- {learning[:200]}...\n"

        if user_prompt and self.domain_pack_dir:
            # Select and build domain knowledge context
            selected_pack = self.knowledge_store.select_pack_by_embedding(
                user_prompt, "learning/packs/calc.v1"
            )
            self.domain_pack_dir = selected_pack or self.domain_pack_dir
            bundle = self.knowledge_store.build_domain_knowledge_context(
                query=user_prompt,
                pack_dir=self.domain_pack_dir,
                topk=5,
                expand_radius=1,
                max_nodes=8,
                max_examples_per_node=1,
            )
            if bundle:
                base_prompt += "\n\n# Domain knowledge\n"
                base_prompt += (
                    "You have access to the following formal rules and concepts relevant to the user's request.\n"
                    "Use these rules directly when solving; prefer formal definitions over prose.\n\n" + bundle
                )

        return base_prompt

    def _format_tools_compact(self) -> str:
        """Format tools in a compact way to reduce prompt bloat"""
        if not self.tools:
            return "No tools available"
        tool_summaries = []
        for tool in self.tools:
            if isinstance(tool, dict):
                name = tool.get("name", "unknown")
                if name == "unknown":
                    fn = tool.get("function", {})
                    name = fn.get("name", "unknown")
                    description = fn.get("description", "No description")
                    params = fn.get("parameters", {})
                else:
                    description = tool.get("description", "No description")
                    params = tool.get("parameters", {})
                short_desc = description[:100] + "..." if len(description) > 100 else description
                tool_summaries.append(f"- {name}: {short_desc}, Parameters: {json.dumps(params)[:120]}")
            else:
                tool_summaries.append(f"- {str(tool)}")
        return "\n".join(tool_summaries)

    async def process_request(
        self,
        initial_messages: List[Dict[str, str]],
        logger: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Process a request using the ReAct pattern with KnowledgeStore for state management.
        Handles the complete reasoning and tool-use loop without external dependencies.
        Integrates learning loop for experience tracking and improvement.
        """
        logger.info("Starting ReAct agent processing...")

        # Find latest user message and update per-user memory
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

        # Initialize conversation in knowledge store
        for msg in initial_messages:
            self.knowledge_store.add_message(msg.get("role", USER_ROLE), msg.get("content", ""))

        # Build messages for the LLM: pass through the exact incoming chat history
        messages = [
            {"role": msg.get("role", USER_ROLE), "content": msg.get("content", "")}
            for msg in initial_messages
        ]

        # Add system prompt and optional context
        system_prompt = await self.get_react_system_prompt(user_prompt)
        react_messages: List[Dict[str, str]] = [{"role": SYSTEM_ROLE, "content": system_prompt}]
        context_summary = self.knowledge_store.build_context_summary()
        if context_summary and context_summary != "No significant context yet.":
            react_messages.append({"role": SYSTEM_ROLE, "content": f"Current context:\n{context_summary}"})

        # Add conversation history
        react_messages.extend(messages)

        # Iterative loop
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"ReAct iteration {iteration}/{self.max_iterations}")
            self.knowledge_store.add_context(
                f"Starting ReAct iteration {iteration}", ContextType.THOUGHT, ImportanceLevel.LOW, "react_agent"
            )

            response_buffer = ""
            thinking_content = ""
            answering = False
            pre_think_buffer = ""

            async for token in self._execute_llm_request(react_messages, logger):
                response_buffer += token
                # Stream thoughts and then tokens
                if "<think>" in response_buffer and "</think>" in response_buffer and not thinking_content:
                    think_match = re.search(r"<think>(.*?)</think>", response_buffer, re.DOTALL)
                    if think_match:
                        thinking_content = (pre_think_buffer + think_match.group(1)).strip()
                        if thinking_content:
                            self.knowledge_store.add_context(
                                thinking_content, ContextType.THOUGHT, ImportanceLevel.MEDIUM, "react_agent"
                            )
                            learning_loop.record_action("thinking", {"content": thinking_content[:200] + "..."})
                            yield f"<thought>{thinking_content}</thought>"
                    # Emit the content after the thinking block as normal tokens
                    after = response_buffer.split("</think>", 1)[-1]
                    if after:
                        yield f"<token>{after}</token>"
                        if "Final Answer:" in after:
                            answering = True
                else:
                    # Before thinking block is closed, treat as thought to the UI
                    if "</think>" not in response_buffer:
                        yield f"<thought>{token}</thought>"
                    else:
                        # After thinking, stream tokens immediately (no gating on Final Answer)
                        yield f"<token>{token}</token>"
                        if "Final Answer:" in token:
                            answering = True

            # Post-process response
            if not thinking_content:
                think_match = re.search(r"<think>(.*?)</think>", response_buffer, re.DOTALL)
                if think_match:
                    thinking_content = (pre_think_buffer + think_match.group(1)).strip()
                    if thinking_content:
                        self.knowledge_store.add_context(
                            thinking_content, ContextType.THOUGHT, ImportanceLevel.MEDIUM, "react_agent"
                        )
                        learning_loop.record_action("thinking", {"content": thinking_content[:200] + "..."})
                        yield f"<thought>{thinking_content}</thought>"

            # Extract content after thinking markers
            content_after_thinking = re.split(r"</think>", response_buffer, maxsplit=1)
            content = content_after_thinking[-1].strip() if content_after_thinking else response_buffer.strip()

            # Tool handling
            from server.agent import config

            tool_results_for_ui: List[Dict[str, Any]] = []

            async def tool_result_callback(tool_result_summary):
                tool_results_for_ui.append(tool_result_summary)
                learning_loop.record_action(
                    "tool_use",
                    {
                        "tool_name": tool_result_summary.get("name"),
                        "arguments": tool_result_summary.get("arguments"),
                        "success": tool_result_summary.get("status") == "success",
                    },
                )

            made_tool_calls = await process_tool_calls(
                content, config.TOOL_CALL_REGEX, react_messages, logger, self.knowledge_store, tool_result_callback
            )

            for tool_result_summary in tool_results_for_ui:
                yield f"<tool_result>{json.dumps(tool_result_summary)}</tool_result>"

            # Record assistant content and continue or finish
            self.knowledge_store.add_message(ASSISTANT_ROLE, content)
            react_messages.append({"role": ASSISTANT_ROLE, "content": content})

            if made_tool_calls:
                self.knowledge_store.add_context(
                    "Tool calls made", ContextType.ACTION, ImportanceLevel.MEDIUM, "react_agent"
                )
                continue
            else:
                if content.strip().startswith("Final Answer:") or answering:
                    self.knowledge_store.mark_complete(content)
                    success = "error" not in content.lower() and "failed" not in content.lower()
                    learning_loop.complete_task(success, content[:200])
                    break
                else:
                    continue

        if iteration >= self.max_iterations:
            final_msg = "\n\n[Task completed - reached maximum iterations]"
            self.knowledge_store.add_context(
                "Reached maximum iterations", ContextType.THOUGHT, ImportanceLevel.HIGH, "react_agent", {"reason": "max_iterations"}
            )
            learning_loop.complete_task(False, "Reached maximum iterations without completion")
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