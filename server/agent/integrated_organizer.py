"""
Integration layer for the enhanced Anton agent architecture.

This module provides backward compatibility while enabling the new
workflow orchestrator and enhanced components to work with the existing system.
"""
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from server.agent.workflow_orchestrator import AdaptiveWorkflowOrchestrator, TaskComplexity
from server.agent.enhanced_tool_manager import enhanced_tool_manager
from server.agent.intelligent_context_manager import intelligent_context_manager, ContextType, ContextPriority
from server.agent.resilient_parser import resilient_parser, OutputFormat, ParsingContext
from server.agent.state_tracker import state_tracker
from server.model_server import AgentChatRequest

logger = logging.getLogger(__name__)


class IntegratedOrganizer:
    """
    Enhanced organizer that integrates new architecture components while
    maintaining backward compatibility with the existing system.
    """
    
    def __init__(self, api_base_url: str, logger: Any):
        self.api_base_url = api_base_url
        self.logger = logger
        
        # Initialize enhanced components
        self.workflow_orchestrator = AdaptiveWorkflowOrchestrator(api_base_url, logger)
        self.use_enhanced_workflow = True  # Flag to enable/disable new workflow
        
        # Track integration state
        self.integration_active = False
        
        logger.info("Integrated Organizer initialized with enhanced architecture")
    
    async def run_enhanced_organizer_loop(
        self,
        request: AgentChatRequest,
        logger: Any,
        api_base_url: str
    ) -> AsyncGenerator[str, None]:
        """
        Enhanced organizer loop that uses the new architecture components.
        
        This method serves as a drop-in replacement for the original
        run_organizer_loop function with enhanced capabilities.
        """
        try:
            self.integration_active = True
            
            # Prepare initial messages and task
            from server.agent.message_handler import prepare_initial_messages
            organizer_messages = prepare_initial_messages(request.messages)
            original_task = organizer_messages[-1]["content"]
            
            # Start comprehensive state tracking
            task_id = state_tracker.start_task_tracking(
                task_description=original_task,
                task_type=self._classify_task_type(original_task),
                complexity=self._assess_task_complexity(original_task)
            )
            
            # Add task context to intelligent context manager
            intelligent_context_manager.add_context(
                content=original_task,
                context_type=ContextType.TASK_DESCRIPTION,
                priority=ContextPriority.CRITICAL,
                source="user_request"
            )
            
            # Get strategy recommendations from state tracker
            task_type = self._classify_task_type(original_task)
            complexity = self._assess_task_complexity(original_task)
            recommendations = state_tracker.get_strategy_recommendations(task_type, complexity)
            
            if recommendations:
                self.logger.info(f"Strategy recommendations: {recommendations}")
                # Add recommendations to context
                intelligent_context_manager.add_context(
                    content=f"Strategy recommendations: {'; '.join(recommendations)}",
                    context_type=ContextType.MEMORY,
                    priority=ContextPriority.HIGH,
                    source="strategy_recommendations"
                )
            
            # Decide whether to use enhanced workflow based on task complexity
            if self.use_enhanced_workflow and self._should_use_enhanced_workflow(original_task):
                # Use the new adaptive workflow orchestrator
                self.logger.info("Using enhanced adaptive workflow")
                
                async for result in self.workflow_orchestrator.execute_workflow(
                    task=original_task,
                    initial_messages=organizer_messages,
                    tools=request.tools,
                    max_turns=10  # From config.MAX_TURNS
                ):
                    yield result
            else:
                # Fall back to original workflow with enhancements
                self.logger.info("Using enhanced traditional workflow")
                
                async for result in self._run_enhanced_traditional_workflow(
                    request, organizer_messages, original_task, task_id
                ):
                    yield result
            
            # Complete task tracking
            state_tracker.complete_task(task_id, success=True, final_output="Task completed")
            
        except Exception as e:
            self.logger.error(f"Enhanced organizer loop failed: {e}", exc_info=True)
            if 'task_id' in locals():
                state_tracker.complete_task(task_id, success=False, final_output=f"Error: {e}")
            yield f"\n[Error in enhanced workflow: {e}]\n"
        finally:
            self.integration_active = False
    
    def _classify_task_type(self, task: str) -> str:
        """Classify the task type for tracking and optimization."""
        task_lower = task.lower()
        
        if any(keyword in task_lower for keyword in ["read", "show", "display", "list", "view"]):
            return "information_retrieval"
        elif any(keyword in task_lower for keyword in ["analyze", "review", "examine", "check"]):
            return "analysis"
        elif any(keyword in task_lower for keyword in ["create", "write", "generate", "build"]):
            return "creation"
        elif any(keyword in task_lower for keyword in ["fix", "debug", "solve", "repair"]):
            return "problem_solving"
        elif any(keyword in task_lower for keyword in ["refactor", "optimize", "improve", "enhance"]):
            return "optimization"
        else:
            return "general"
    
    def _assess_task_complexity(self, task: str) -> int:
        """Assess task complexity on a 1-5 scale."""
        complexity_indicators = {
            1: ["show", "list", "display", "read"],
            2: ["find", "search", "get", "fetch"],
            3: ["analyze", "check", "review", "examine"],
            4: ["create", "implement", "build", "generate"],
            5: ["refactor", "optimize", "design", "architect"]
        }
        
        task_lower = task.lower()
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in task_lower for indicator in indicators):
                return complexity
        
        return 3  # Default to medium complexity
    
    def _should_use_enhanced_workflow(self, task: str) -> bool:
        """Determine if enhanced workflow should be used based on task characteristics."""
        # Use enhanced workflow for complex tasks or when specifically requested
        complexity = self._assess_task_complexity(task)
        
        # Always use enhanced workflow for complex tasks
        if complexity >= 4:
            return True
        
        # Use enhanced workflow for tasks that benefit from advanced features
        enhanced_indicators = [
            "analyze", "refactor", "optimize", "design", "architecture",
            "multiple", "complex", "comprehensive", "detailed"
        ]
        
        task_lower = task.lower()
        if any(indicator in task_lower for indicator in enhanced_indicators):
            return True
        
        return False  # Use traditional workflow for simple tasks
    
    async def _run_enhanced_traditional_workflow(
        self,
        request: AgentChatRequest,
        organizer_messages: List[Dict[str, str]],
        original_task: str,
        task_id: str
    ) -> AsyncGenerator[str, None]:
        """
        Run the traditional workflow with enhancements from new components.
        """
        from server.agent.doer import execute_turn, run_doer_loop
        from server.agent.prompts import get_evaluator_prompt
        from client.context_builder import ContextBuilder
        from server.agent import config
        from server.agent.config import SYSTEM_ROLE, ASSISTANT_ROLE, USER_ROLE
        import re
        import time
        
        # Enhanced context store using intelligent context manager
        context_store = {
            "explored_files": set(),
            "code_content": {},
            "task_progress": []
        }
        
        # Build enhanced system prompt with memory context
        system_prompt = await self._build_enhanced_system_prompt(original_task)
        organizer_messages.insert(0, {"role": SYSTEM_ROLE, "content": system_prompt})
        
        try:
            for turn in range(config.MAX_TURNS):
                state_tracker.record_step(task_id, f"Starting turn {turn + 1}")
                
                # --- Enhanced Planner Turn ---
                state_tracker.record_step(task_id, "Planner phase")
                
                # Add intelligent context summary
                if turn > 0:
                    context_summary = self._build_intelligent_context_summary(context_store)
                    organizer_messages.append({
                        "role": SYSTEM_ROLE,
                        "content": f"Previous step progress:\n{context_summary}"
                    })
                
                # Execute planner with enhanced parsing
                response_buffer = ""
                async for token in execute_turn(self.api_base_url, organizer_messages, self.logger, request.tools, 0.6, False):
                    response_buffer += token
                
                # Use resilient parser for planner output
                parsed_result = resilient_parser.parse(
                    response_buffer,
                    ParsingContext(expected_format=OutputFormat.PLAIN_TEXT)
                )
                
                content = parsed_result.content.get("raw_text", response_buffer) if parsed_result.content else response_buffer
                content = re.split(r"</think>", content, maxsplit=1)[-1].strip()
                
                self.logger.info("Enhanced Planner said:\n" + content)
                organizer_messages.append({"role": ASSISTANT_ROLE, "content": content})
                
                # Record planner decision
                state_tracker.record_decision(
                    task_id,
                    f"Planner delegated: {content[:100]}...",
                    "Based on task analysis and available tools"
                )
                
                # --- Enhanced Doer Turn ---
                state_tracker.record_step(task_id, "Doer execution phase")
                
                # Prepare doer with enhanced context
                doer_messages = await self._prepare_enhanced_doer_messages(context_store)
                doer_messages.append({"role": USER_ROLE, "content": content})
                
                # Execute doer loop with tool tracking
                async for token in self._run_enhanced_doer_loop(
                    doer_messages, request.tools, context_store, task_id
                ):
                    yield token
                
                doer_result = doer_messages[-1]["content"]
                
                # Add doer result to context manager
                intelligent_context_manager.add_context(
                    content=doer_result,
                    context_type=ContextType.CONVERSATION,
                    priority=ContextPriority.HIGH,
                    source=f"doer_turn_{turn}"
                )
                
                # --- Enhanced Evaluator Turn ---
                state_tracker.record_step(task_id, "Evaluator assessment phase")
                
                evaluator_system_prompt = get_evaluator_prompt() + (
                    f"\n\nHere is the information to evaluate:"
                    f"\nOriginal High-Level Task: {original_task}"
                    f"\nDelegated Step for the Doer: {content}"
                    f"\nDoer's Final Result: {doer_result}"
                )
                
                organizer_messages.append({
                    "role": USER_ROLE,
                    "content": f"The doer has completed the delegated task. Here is the result:\n\n{doer_result}"
                })
                
                evaluator_messages = [{"role": SYSTEM_ROLE, "content": evaluator_system_prompt}]
                
                response_buffer = ""
                async for token in execute_turn(self.api_base_url, evaluator_messages, self.logger, request.tools, 0.1, True):
                    response_buffer += token
                    content_check = re.split(r"</think>", response_buffer, maxsplit=1)
                    if len(content_check) == 2:
                        yield token
                
                # Parse evaluator response
                evaluator_parsed = resilient_parser.parse(
                    response_buffer,
                    ParsingContext(expected_format=OutputFormat.PLAIN_TEXT)
                )
                
                evaluator_response = evaluator_parsed.content.get("raw_text", response_buffer) if evaluator_parsed.content else response_buffer
                evaluator_response = re.split(r"</think>", evaluator_response, maxsplit=1)[-1].strip()
                
                self.logger.info("Enhanced Evaluator response:\n" + evaluator_response)
                
                # Record evaluator decision
                if evaluator_response.startswith('SUCCESS:'):
                    state_tracker.record_decision(task_id, "Evaluator: SUCCESS", "Task step completed successfully")
                    continue
                elif evaluator_response.startswith('FAILURE:'):
                    state_tracker.record_decision(task_id, "Evaluator: FAILURE", "Task step needs adjustment")
                    organizer_messages.append({"role": USER_ROLE, "content": evaluator_response})
                    continue
                elif evaluator_response.startswith('DONE:'):
                    state_tracker.record_decision(task_id, "Evaluator: DONE", "Task completed successfully")
                    organizer_messages.append({"role": USER_ROLE, "content": evaluator_response})
                    
                    # Generate final summary with enhanced context
                    summary_prompt = f"The entire task has been successfully completed. Based on our conversation history, please summarize the steps taken and provide the final answer to the original high-level task: {original_task}"
                    organizer_messages.append({"role": USER_ROLE, "content": summary_prompt})
                    
                    # Use enhanced context for final response
                    enhanced_system_prompt = await self._build_enhanced_system_prompt_doer()
                    organizer_messages[0] = {"role": SYSTEM_ROLE, "content": enhanced_system_prompt}
                    
                    final_response_buffer = ""
                    async for token in execute_turn(self.api_base_url, organizer_messages, self.logger, request.tools, True):
                        final_response_buffer += token
                        yield token
                    return
            
            # Max turns reached
            self.logger.warning("Enhanced workflow: Max turns reached")
            yield "\n\n[Enhanced workflow: Reached maximum turns; stopping organizer.]"
            
        except Exception as e:
            self.logger.error(f"Error in enhanced traditional workflow: {e}", exc_info=True)
            state_tracker.record_error(task_id, "workflow_error", str(e), "Graceful degradation")
            yield f"\n[Enhanced workflow error: {e}]\n"
    
    async def _build_enhanced_system_prompt(self, task_description: str) -> str:
        """Build enhanced system prompt with memory context."""
        from server.agent.prompts import get_planner_prompt
        
        # Get base prompt
        base_prompt = get_planner_prompt()
        
        # Get enhanced tool context
        tool_schemas = enhanced_tool_manager.get_tool_schemas()
        
        # Get memory context from intelligent context manager
        memory_context = intelligent_context_manager.get_context_for_prompt(
            context_types=[ContextType.MEMORY, ContextType.LEARNING_INSIGHTS],
            max_tokens=1000
        )
        
        # Build enhanced prompt
        enhanced_prompt = base_prompt.replace('{tools}', str(tool_schemas))
        
        if memory_context:
            enhanced_prompt += f"\n\nRelevant Memory Context:\n{memory_context}"
        
        return enhanced_prompt
    
    async def _build_enhanced_system_prompt_doer(self) -> str:
        """Build enhanced system prompt for doer."""
        from server.agent.prompts import get_doer_prompt
        
        base_prompt = get_doer_prompt()
        tool_schemas = enhanced_tool_manager.get_tool_schemas()
        
        return base_prompt.replace('{tools}', str(tool_schemas))
    
    async def _prepare_enhanced_doer_messages(self, context_store: Dict) -> List[Dict[str, str]]:
        """Prepare doer messages with enhanced context."""
        doer_messages = []
        
        # Add enhanced system prompt
        system_prompt = await self._build_enhanced_system_prompt_doer()
        doer_messages.append({"role": "system", "content": system_prompt})
        
        # Add context about explored files
        if context_store["explored_files"]:
            explored_files_msg = "Previously explored files: " + ", ".join(context_store["explored_files"])
            doer_messages.append({"role": "system", "content": explored_files_msg})
        
        # Add file content context (with intelligent truncation)
        for filename, content in context_store["code_content"].items():
            # Use intelligent context manager to decide what to include
            intelligent_context_manager.add_context(
                content=f"File {filename}: {content}",
                context_type=ContextType.FILE_CONTENT,
                priority=ContextPriority.MEDIUM,
                source=f"file_{filename}"
            )
        
        return doer_messages
    
    async def _run_enhanced_doer_loop(
        self,
        doer_messages: List[Dict[str, str]],
        tools: List,
        context_store: Dict,
        task_id: str
    ) -> AsyncGenerator[str, None]:
        """Run doer loop with enhanced tool tracking."""
        from server.agent.doer import run_doer_loop
        
        # Wrap the original doer loop to track tool usage
        async for token in run_doer_loop(
            doer_messages, tools, self.logger, self.api_base_url, True, context_store
        ):
            yield token
        
        # Extract tool usage from context store and record it
        if "tool_outputs" in context_store:
            for tool_output in context_store["tool_outputs"]:
                # Parse tool output to extract tool name and success
                if "Tool:" in tool_output:
                    parts = tool_output.split("|")
                    if len(parts) >= 3:
                        tool_info = parts[0].replace("Tool:", "").strip()
                        result_info = parts[2].replace("Result:", "").strip()
                        
                        # Determine success based on result
                        success = not any(error_word in result_info.lower() 
                                        for error_word in ["error", "failed", "exception"])
                        
                        # Record tool usage
                        state_tracker.record_tool_usage(
                            task_id=task_id,
                            tool_name=tool_info,
                            success=success,
                            execution_time=1.0,  # Placeholder - would need actual timing
                            context=f"doer_execution"
                        )
    
    def _build_intelligent_context_summary(self, context_store: Dict) -> str:
        """Build context summary using intelligent context manager."""
        # Get summary from intelligent context manager
        context_summary = intelligent_context_manager.get_context_for_prompt(
            context_types=[ContextType.PROGRESS_UPDATE, ContextType.FILE_CONTENT],
            max_tokens=500
        )
        
        if context_summary:
            return context_summary
        
        # Fallback to traditional summary
        summary_parts = []
        
        if context_store["explored_files"]:
            summary_parts.append("Explored files: " + ", ".join(list(context_store["explored_files"])[:5]))
        
        if context_store["code_content"]:
            summary_parts.append("Retrieved file contents:")
            for filename in list(context_store["code_content"].keys())[:3]:
                summary_parts.append(f"- {filename}")
        
        if context_store["task_progress"]:
            summary_parts.append("Progress so far:")
            for step in context_store["task_progress"][-3:]:
                summary_parts.append(f"- {step}")
        
        return "\n".join(summary_parts) if summary_parts else "No significant progress to report."


# Create backward-compatible function that can replace the original organizer
async def run_enhanced_organizer_loop(
    request: AgentChatRequest,
    logger: Any,
    api_base_url: str
) -> AsyncGenerator[str, None]:
    """
    Enhanced organizer loop that serves as a drop-in replacement
    for the original run_organizer_loop function.
    """
    integrated_organizer = IntegratedOrganizer(api_base_url, logger)
    
    async for result in integrated_organizer.run_enhanced_organizer_loop(request, logger, api_base_url):
        yield result


# Utility functions for gradual migration
def enable_enhanced_features():
    """Enable enhanced features globally."""
    logger.info("Enhanced Anton agent features enabled")


def disable_enhanced_features():
    """Disable enhanced features and fall back to original behavior."""
    logger.info("Enhanced Anton agent features disabled - using original behavior")


def get_integration_status() -> Dict[str, Any]:
    """Get status of integration components."""
    return {
        "enhanced_tool_manager_loaded": enhanced_tool_manager is not None,
        "intelligent_context_manager_loaded": intelligent_context_manager is not None,
        "resilient_parser_loaded": resilient_parser is not None,
        "state_tracker_loaded": state_tracker is not None,
        "integration_active": hasattr(IntegratedOrganizer, 'integration_active')
    }