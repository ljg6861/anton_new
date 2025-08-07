"""
ReAct (Reason-Act) Agent implementation that replaces the rigid Planner-Doer-Evaluator loop
with a more flexible single-agent model that decides next steps in a single LLM call.
"""
import json
import re
import time
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional

from server.agent.conversation_state import ConversationState, StateType
from server.agent.doer import execute_turn
from server.agent.tool_executor import process_tool_calls
from server.agent.config import SYSTEM_ROLE, ASSISTANT_ROLE, USER_ROLE

logger = logging.getLogger(__name__)


class ReActAgent:
    """
    Single-agent ReAct implementation that combines reasoning and acting in one flow.
    Significantly reduces latency compared to the multi-agent Planner-Doer-Evaluator pattern.
    """
    
    def __init__(self, api_base_url: str, tools: List[Dict], max_iterations: int = 10):
        self.api_base_url = api_base_url
        self.tools = tools
        self.max_iterations = max_iterations
        
    def get_react_system_prompt(self) -> str:
        """Get the system prompt for the ReAct agent"""
        return """You are Anton, an intelligent AI assistant that uses the ReAct (Reason-Act) pattern.

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
- Code analysis and search
- Web search 
- Knowledge retrieval

Always think step by step and be helpful to the user."""

    async def process_request(
        self,
        conversation_state: ConversationState,
        logger: Any
    ) -> AsyncGenerator[str, None]:
        """
        Process a request using the ReAct pattern.
        This replaces the complex Planner-Doer-Evaluator loop.
        """
        logger.info("Starting ReAct agent processing...")
        
        # Build messages for the LLM
        messages = conversation_state.get_messages_for_llm()
        
        # Add system prompt
        system_prompt = self.get_react_system_prompt()
        react_messages = [{"role": SYSTEM_ROLE, "content": system_prompt}]
        
        # Add context if available
        context_summary = conversation_state.build_context_summary()
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
            
            conversation_state.add_state_entry(
                f"Starting ReAct iteration {iteration}",
                StateType.THOUGHT
            )
            
            # Get response from LLM
            response_buffer = ""
            start_time = time.time()
            
            async for token in execute_turn(
                api_base_url=self.api_base_url,
                messages=react_messages,
                logger=logger,
                tools=self.tools,
                temperature=0.7,
                complex=True
            ):
                response_buffer += token
                yield token
            
            # Process the response
            logger.info(f"ReAct agent response: {response_buffer}")
            
            # Extract thinking from response
            thinking_match = re.search(r'<think>(.*?)</think>', response_buffer, re.DOTALL)
            if thinking_match:
                thinking = thinking_match.group(1).strip()
                conversation_state.add_state_entry(thinking, StateType.THOUGHT)
                logger.info(f"Agent thinking: {thinking}")
            
            # Extract content after thinking markers
            content = re.split(r'</think>', response_buffer, maxsplit=1)[-1].strip()
            
            # Add to conversation
            conversation_state.add_message(ASSISTANT_ROLE, content)
            react_messages.append({"role": ASSISTANT_ROLE, "content": content})
            
            # Check if agent made tool calls
            from server.agent import config
            made_tool_calls = await process_tool_calls_with_state(
                content, 
                config.TOOL_CALL_REGEX,  # Use existing tool call regex
                react_messages,
                logger,
                conversation_state
            )
            
            if made_tool_calls:
                # If tools were called, continue the loop to let agent process results
                logger.info("Tools were executed, continuing ReAct loop...")
                conversation_state.add_state_entry(
                    "Tool calls completed, processing results...",
                    StateType.ACTION
                )
                continue
            else:
                # No tool calls, check if this looks like a final response
                if self._is_final_response(content):
                    logger.info("Agent provided final response, ending ReAct loop")
                    conversation_state.mark_complete(content)
                    break
                else:
                    # Agent might need another iteration to complete the task
                    logger.info("Response doesn't appear final, continuing...")
                    continue
        
        if iteration >= self.max_iterations:
            logger.warning(f"ReAct agent reached max iterations ({self.max_iterations})")
            final_msg = "\n\n[Task completed - reached maximum iterations]"
            conversation_state.add_state_entry(
                "Reached maximum iterations",
                StateType.THOUGHT,
                {"reason": "max_iterations"}
            )
            yield final_msg
    
    def _is_final_response(self, content: str) -> bool:
        """
        Heuristic to determine if the agent's response is final.
        This replaces the evaluator component.
        """
        # Check for tool calls - if present, not final
        if '<function_calls>' in content or 'I need to' in content.lower():
            return False
            
        # Check for completion indicators
        completion_indicators = [
            'task completed',
            'finished',
            'done',
            'here is the result',
            'the answer is',
            'i have completed',
            'successfully',
            'final result'
        ]
        
        content_lower = content.lower()
        for indicator in completion_indicators:
            if indicator in content_lower:
                return True
        
        # If response is substantial and doesn't indicate more work needed
        if len(content.strip()) > 50 and not any(phrase in content_lower for phrase in [
            'let me', 'i should', 'i will', 'next i', 'first i', 'i need to'
        ]):
            return True
            
        return False


async def process_tool_calls_with_state(
    content: str,
    tool_call_regex,  # Already compiled regex from config
    messages: List[Dict],
    logger: Any,
    conversation_state: ConversationState
) -> bool:
    """
    Process tool calls in the agent's response using the existing tool_executor.
    Updated to work with ConversationState.
    """
    from server.agent.tool_executor import process_tool_calls
    
    # Use existing tool executor but adapt it for ConversationState
    class StateAdapter:
        """Adapter to make ConversationState compatible with knowledge_store interface"""
        def __init__(self, conv_state: ConversationState):
            self.conv_state = conv_state
            
        def update_from_tool_execution(self, tool_name: str, tool_args: dict, tool_result: str):
            self.conv_state.add_tool_output(tool_name, tool_result, {"args": tool_args})
            
            # Track file explorations for file-related tools
            if tool_name in ['read_file', 'list_files', 'write_file'] and 'path' in tool_args:
                self.conv_state.add_file_exploration(tool_args['path'])
    
    state_adapter = StateAdapter(conversation_state)
    
    # Use the existing tool executor
    return await process_tool_calls(
        content,
        tool_call_regex,  # Pass the compiled regex directly
        messages,
        logger,
        state_adapter
    )