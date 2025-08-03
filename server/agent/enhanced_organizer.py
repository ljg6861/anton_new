"""
Enhanced Organizer with Comprehensive Workflow Optimization

Integrates all optimization components:
- Loop detection and prevention
- Enhanced Doer with structured tool usage
- Sophisticated evaluator with three verdict levels
- Central coordination with comprehensive state tracking
- Performance monitoring with actionable insights
"""

import json
import re
import time
from typing import AsyncGenerator, Any, Dict, List

from client.context_builder import ContextBuilder
from server.agent import config
from server.agent.config import SYSTEM_ROLE, ASSISTANT_ROLE, USER_ROLE
from server.agent.agent_coordinator import AgentCoordinator, AgentType, TaskStatus
from server.agent.enhanced_doer import EnhancedDoer, ResponseType, ExecutionStatus
from server.agent.enhanced_evaluator import EnhancedEvaluator, EvaluationLevel
from server.agent.performance_monitor import PerformanceMonitor
from server.agent.doer import execute_turn
from server.agent.learning_identifier import learning_identifier
from server.agent.rag_manager import rag_manager
from server.model_server import AgentChatRequest


class EnhancedOrganizer:
    """
    Enhanced orchestrator implementing comprehensive workflow optimization.
    
    Features:
    - Robust loop detection and prevention (0.85 similarity threshold)
    - Structured tool usage enforcement in Doer
    - Three-level evaluation (SUCCESS/PARTIAL/FAILURE)
    - Maximum 10 iterations with comprehensive tracking
    - Performance monitoring with 15-second timeouts
    - Automatic optimization suggestions
    """
    
    def __init__(self):
        # Initialize all components
        self.coordinator = AgentCoordinator(
            max_iterations=10,
            similarity_threshold=0.85,
            enable_performance_monitoring=True
        )
        
        self.enhanced_doer = EnhancedDoer(
            max_response_time=15.0,
            require_structured_output=True,
            validate_tools_before_execution=True
        )
        
        self.enhanced_evaluator = EnhancedEvaluator()
        
        self.performance_monitor = PerformanceMonitor(
            max_operation_duration=30.0,
            max_memory_percent=80.0,
            max_cpu_percent=90.0,
            monitoring_interval=5.0
        )
        
        # State tracking
        self.task_active = False
        self.current_task_id = None
    
    async def run_enhanced_organizer_loop(self,
                                        request: AgentChatRequest,
                                        logger: Any,
                                        api_base_url: str) -> AsyncGenerator[str, None]:
        """
        Main enhanced organizer loop with comprehensive optimization.
        """
        try:
            # Start performance monitoring
            self.performance_monitor.start_monitoring()
            
            # Initialize task
            task_id = f"task_{int(time.time())}"
            original_task = request.messages[-1]["content"]
            
            # Prepare context store
            context_store = {
                "explored_files": set(),
                "code_content": {},
                "task_progress": [],
                "tool_outputs": []
            }
            
            # Initialize coordination
            coordinator_state = self.coordinator.initialize_task(
                task_id=task_id,
                original_task=original_task,
                context_store=context_store
            )
            
            self.task_active = True
            self.current_task_id = task_id
            
            logger.info(f"Starting enhanced organizer loop for task: {task_id}")
            logger.info(f"Max iterations: {self.coordinator.max_iterations}")
            
            # Prepare initial messages for Planner
            organizer_messages = self._prepare_initial_messages(request.messages, original_task)
            
            # Main coordination loop
            while True:
                # Check if we should continue
                should_continue, reason = self.coordinator.should_continue()
                if not should_continue:
                    logger.info(f"Stopping coordination: {reason}")
                    yield f"\n\n[Task completed: {reason}]\n"
                    break
                
                # Advance turn
                current_turn = self.coordinator.advance_turn()
                logger.info(f"Starting turn {current_turn}/{self.coordinator.max_iterations}")
                
                # Check for loops before starting turn
                is_loop, pattern_breaking_instruction, loop_info = self.coordinator.check_for_loops()
                if is_loop and pattern_breaking_instruction:
                    logger.warning(f"Loop detected: {loop_info}")
                    yield f"\n[LOOP DETECTED - Breaking pattern]\n"
                    
                    # Inject pattern-breaking instruction
                    organizer_messages.append({
                        "role": SYSTEM_ROLE,
                        "content": pattern_breaking_instruction
                    })
                
                # Execute Planner turn
                planner_result = await self._execute_planner_turn(
                    organizer_messages, context_store, api_base_url, logger, current_turn
                )
                
                if not planner_result:
                    logger.error("Planner turn failed")
                    break
                
                planner_instruction = planner_result["instruction"]
                
                # Execute Doer turn
                doer_result = await self._execute_doer_turn(
                    planner_instruction, request.tools, context_store, api_base_url, logger, current_turn
                )
                
                if not doer_result:
                    logger.error("Doer turn failed")
                    break
                
                # Yield Doer response to user
                yield doer_result["response_content"]
                
                # Execute Evaluator turn
                evaluation_result = await self._execute_evaluator_turn(
                    original_task, planner_instruction, doer_result, context_store, logger, current_turn
                )
                
                if not evaluation_result:
                    logger.error("Evaluator turn failed")
                    break
                
                # Process evaluation result
                evaluation_level = evaluation_result["level"]
                evaluation_reason = evaluation_result["reason"]
                
                logger.info(f"Evaluation: {evaluation_level.value} - {evaluation_reason}")
                
                # Update organizer messages based on evaluation
                if evaluation_level == EvaluationLevel.DONE:
                    logger.info("Task completed successfully")
                    
                    # Generate final summary
                    final_summary = await self._generate_final_summary(
                        original_task, organizer_messages, api_base_url, logger
                    )
                    yield final_summary
                    break
                
                elif evaluation_level == EvaluationLevel.SUCCESS:
                    # Continue with next iteration
                    organizer_messages.append({
                        "role": USER_ROLE,
                        "content": f"SUCCESS: {evaluation_reason}. Continue with the next step."
                    })
                
                elif evaluation_level == EvaluationLevel.PARTIAL:
                    # Provide feedback and continue
                    feedback = f"PARTIAL: {evaluation_reason}. "
                    if evaluation_result.get("suggestions"):
                        feedback += f"Suggestions: {'; '.join(evaluation_result['suggestions'])}"
                    
                    organizer_messages.append({
                        "role": USER_ROLE,
                        "content": feedback
                    })
                
                else:  # FAILURE
                    # Provide detailed feedback for improvement
                    feedback = f"FAILURE: {evaluation_reason}. "
                    if evaluation_result.get("suggestions"):
                        feedback += f"Please address these issues: {'; '.join(evaluation_result['suggestions'])}"
                    
                    organizer_messages.append({
                        "role": USER_ROLE,
                        "content": feedback
                    })
                
                # Store learning insights if appropriate
                await self._analyze_and_store_learning(
                    planner_instruction, doer_result, context_store, original_task, logger
                )
            
            # Finalize task
            final_state = self.coordinator.finalize_task()
            
            # Generate performance report
            performance_report = self.performance_monitor.get_performance_report()
            logger.info("Performance Summary:")
            logger.info(f"- Overall performance level: {performance_report.get('performance_level', 'unknown')}")
            logger.info(f"- Total operations: {performance_report.get('operation_statistics', {}).get('total_operations', 0)}")
            logger.info(f"- Success rate: {performance_report.get('operation_statistics', {}).get('success_rate', 0):.1%}")
            
            # Log optimization suggestions
            suggestions = performance_report.get('optimization_suggestions', [])
            if suggestions:
                logger.info("Optimization suggestions:")
                for suggestion in suggestions[:3]:  # Top 3 suggestions
                    logger.info(f"- {suggestion.get('description', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error in enhanced organizer loop: {e}", exc_info=True)
            yield f"\n[ERROR: {str(e)}]\n"
        
        finally:
            # Cleanup
            self.performance_monitor.stop_monitoring()
            self.task_active = False
            self.current_task_id = None
    
    async def _execute_planner_turn(self, 
                                  organizer_messages: List[Dict],
                                  context_store: Dict,
                                  api_base_url: str,
                                  logger: Any,
                                  turn: int) -> Dict:
        """Execute a Planner turn with performance tracking."""
        operation_id = self.performance_monitor.start_operation(
            operation_name=f"planner_turn_{turn}",
            operation_type="planner",
            metadata={"turn": turn}
        )
        
        self.coordinator.start_agent_interaction(
            AgentType.PLANNER, 
            organizer_messages[-1]["content"] if organizer_messages else ""
        )
        
        try:
            # Add context summary to planner
            context_summary = self._build_context_summary(context_store)
            if turn > 1 and context_summary:
                organizer_messages.append({
                    "role": SYSTEM_ROLE,
                    "content": f"Previous step progress:\n{context_summary}"
                })
            
            # Execute planner
            response_buffer = ""
            async for token in execute_turn(api_base_url, organizer_messages, logger, [], 0.6, False):
                response_buffer += token
            
            # Extract instruction
            content = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
            logger.info(f"Planner instruction: {content}")
            
            # Update messages
            organizer_messages.append({"role": ASSISTANT_ROLE, "content": content})
            
            # Complete tracking
            self.coordinator.complete_agent_interaction(response_buffer)
            self.performance_monitor.end_operation(operation_id, success=True)
            
            return {
                "instruction": content,
                "full_response": response_buffer
            }
            
        except Exception as e:
            logger.error(f"Planner turn failed: {e}")
            self.coordinator.complete_agent_interaction("", tool_calls_made=False)
            self.performance_monitor.end_operation(operation_id, success=False)
            return None
    
    async def _execute_doer_turn(self,
                               instruction: str,
                               tools: List,
                               context_store: Dict,
                               api_base_url: str,
                               logger: Any,
                               turn: int) -> Dict:
        """Execute a Doer turn with enhanced structure enforcement."""
        operation_id = self.performance_monitor.start_operation(
            operation_name=f"doer_turn_{turn}",
            operation_type="doer",
            metadata={"turn": turn, "instruction_length": len(instruction)}
        )
        
        self.coordinator.start_agent_interaction(AgentType.DOER, instruction)
        
        try:
            # Prepare Doer messages
            doer_messages = await self._prepare_doer_messages(instruction, context_store)
            
            # Execute enhanced Doer
            doer_response = await self.enhanced_doer.execute_structured_turn(
                api_base_url=api_base_url,
                messages=doer_messages,
                logger=logger,
                tools=tools,
                context_store=context_store
            )
            
            # Validate response
            if doer_response.execution_status != ExecutionStatus.SUCCESS:
                logger.warning(f"Doer execution issues: {doer_response.validation_errors}")
            
            # Extract tool results for tracking
            tool_results = [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "status": tc.status.value,
                    "duration": tc.duration
                }
                for tc in doer_response.tool_calls
            ]
            
            # Complete tracking
            self.coordinator.complete_agent_interaction(
                response=doer_response.content,
                tool_calls_made=len(doer_response.tool_calls) > 0,
                tool_results=tool_results
            )
            self.performance_monitor.end_operation(
                operation_id, 
                success=doer_response.execution_status == ExecutionStatus.SUCCESS
            )
            
            return {
                "doer_response": doer_response,
                "response_content": doer_response.content,
                "tool_calls": doer_response.tool_calls,
                "execution_status": doer_response.execution_status
            }
            
        except Exception as e:
            logger.error(f"Doer turn failed: {e}")
            self.coordinator.complete_agent_interaction("", tool_calls_made=False)
            self.performance_monitor.end_operation(operation_id, success=False)
            return None
    
    async def _execute_evaluator_turn(self,
                                    original_task: str,
                                    instruction: str,
                                    doer_result: Dict,
                                    context_store: Dict,
                                    logger: Any,
                                    turn: int) -> Dict:
        """Execute an Evaluator turn with sophisticated assessment."""
        operation_id = self.performance_monitor.start_operation(
            operation_name=f"evaluator_turn_{turn}",
            operation_type="evaluator",
            metadata={"turn": turn}
        )
        
        evaluation_input = f"Original task: {original_task}\nInstruction: {instruction}\nResult: {doer_result['response_content']}"
        self.coordinator.start_agent_interaction(AgentType.EVALUATOR, evaluation_input)
        
        try:
            # Execute enhanced evaluation
            evaluation_result = await self.enhanced_evaluator.evaluate_doer_result(
                original_task=original_task,
                delegated_instruction=instruction,
                doer_response=doer_result["doer_response"],
                context_store=context_store,
                logger=logger
            )
            
            # Complete tracking
            evaluation_text = f"{evaluation_result.level.value}: {evaluation_result.reason}"
            self.coordinator.complete_agent_interaction(
                response=evaluation_text,
                evaluation_result=evaluation_text
            )
            self.performance_monitor.end_operation(operation_id, success=True)
            
            return {
                "level": evaluation_result.level,
                "reason": evaluation_result.reason,
                "progress_score": evaluation_result.progress_score,
                "information_value": evaluation_result.information_value,
                "suggestions": evaluation_result.suggestions,
                "evaluation_result": evaluation_result
            }
            
        except Exception as e:
            logger.error(f"Evaluator turn failed: {e}")
            self.coordinator.complete_agent_interaction("", evaluation_result="FAILURE: Evaluation error")
            self.performance_monitor.end_operation(operation_id, success=False)
            return None
    
    async def _prepare_doer_messages(self, instruction: str, context_store: Dict) -> List[Dict]:
        """Prepare messages for the Doer with enhanced context."""
        # Get Doer system prompt
        system_prompt = await ContextBuilder().build_system_prompt_doer()
        messages = [{"role": SYSTEM_ROLE, "content": system_prompt}]
        
        # Add context about explored files
        if context_store.get("explored_files"):
            explored_files_msg = "Previously explored files: " + ", ".join(context_store["explored_files"])
            messages.append({"role": SYSTEM_ROLE, "content": explored_files_msg})
        
        # Add file contents (limited to prevent overflow)
        for filename, content in list(context_store.get("code_content", {}).items())[:5]:  # Limit to 5 files
            messages.append({
                "role": SYSTEM_ROLE,
                "content": f"Content of file {filename}:\n```\n{content}\n```"
            })
        
        # Add the instruction
        messages.append({"role": USER_ROLE, "content": instruction})
        
        return messages
    
    def _prepare_initial_messages(self, request_messages: List[Dict], original_task: str) -> List[Dict]:
        """Prepare initial messages for the organizer."""
        # Start with user messages
        organizer_messages = []
        for msg in request_messages:
            if msg["role"] in [USER_ROLE, ASSISTANT_ROLE]:
                organizer_messages.append(msg)
        
        return organizer_messages
    
    async def _generate_final_summary(self,
                                    original_task: str,
                                    organizer_messages: List[Dict],
                                    api_base_url: str,
                                    logger: Any) -> str:
        """Generate final summary when task is completed."""
        try:
            summary_prompt = (
                f"The entire task has been successfully completed. "
                f"Based on our conversation history, please summarize the steps taken and "
                f"provide the final answer to the original task: {original_task}"
            )
            
            # Create summary messages
            summary_messages = organizer_messages.copy()
            summary_messages.append({"role": USER_ROLE, "content": summary_prompt})
            
            # Get system prompt for final summary
            system_prompt = await ContextBuilder().build_system_prompt_doer()
            summary_messages.insert(0, {"role": SYSTEM_ROLE, "content": system_prompt})
            
            # Generate summary
            final_response = ""
            async for token in execute_turn(api_base_url, summary_messages, logger, [], 0.3, True):
                final_response += token
            
            return final_response
            
        except Exception as e:
            logger.error(f"Failed to generate final summary: {e}")
            return "\n\n[Task completed - summary generation failed]\n"
    
    async def _analyze_and_store_learning(self,
                                        instruction: str,
                                        doer_result: Dict,
                                        context_store: Dict,
                                        original_task: str,
                                        logger: Any):
        """Analyze interaction for learning opportunities."""
        try:
            doer_response = doer_result["doer_response"]
            
            # Extract tool outputs for learning analysis
            tool_outputs = context_store.get("tool_outputs", [])
            
            # Analyze for insights
            insights = learning_identifier.analyze_interaction(
                task_description=instruction,
                doer_response=doer_response.content,
                tool_outputs=tool_outputs,
                context={"original_task": original_task}
            )
            
            if not insights:
                return
            
            # Determine if learning should be triggered
            task_success = doer_response.response_type == ResponseType.FINAL_ANSWER
            novel_info_discovered = bool(context_store.get("explored_files")) or bool(context_store.get("code_content"))
            
            should_learn = learning_identifier.should_trigger_learning(
                insights=insights,
                task_success=task_success,
                novel_information_discovered=novel_info_discovered
            )
            
            if should_learn:
                logger.info(f"Learning triggered: Found {len(insights)} valuable insights")
                
                # Store top insights
                for insight in insights[:2]:
                    knowledge_text = (
                        f"Context: {instruction}\n"
                        f"Insight: {insight.insight_text}\n"
                        f"Keywords: {', '.join(insight.keywords)}\n"
                        f"Related to: {original_task}"
                    )
                    
                    rag_manager.add_knowledge(
                        text=knowledge_text,
                        source=f"learning_{insight.source}_{int(time.time())}"
                    )
                    logger.info(f"Stored learning insight: {insight.insight_text[:100]}...")
                
                rag_manager.save()
            
        except Exception as e:
            logger.error(f"Error in learning analysis: {e}", exc_info=True)
    
    def _build_context_summary(self, context_store: Dict) -> str:
        """Build a summary of context for the planner."""
        summary_parts = []
        
        if context_store.get("explored_files"):
            summary_parts.append("Explored files: " + ", ".join(list(context_store["explored_files"])[:10]))
        
        if context_store.get("code_content"):
            summary_parts.append("Retrieved file contents:")
            for filename in list(context_store["code_content"].keys())[:5]:
                summary_parts.append(f"- {filename}")
        
        if context_store.get("task_progress"):
            summary_parts.append("Progress so far:")
            for step in context_store["task_progress"][-5:]:  # Last 5 steps
                summary_parts.append(f"- {step}")
        
        return "\n".join(summary_parts)
    
    def get_current_status(self) -> Dict:
        """Get current status of the enhanced organizer."""
        if not self.task_active:
            return {"status": "inactive"}
        
        coordinator_debug = self.coordinator.get_debugging_info()
        performance_report = self.performance_monitor.get_performance_report()
        doer_stats = self.enhanced_doer.get_execution_statistics()
        evaluator_stats = self.enhanced_evaluator.get_evaluation_statistics()
        
        return {
            "status": "active",
            "task_id": self.current_task_id,
            "coordination": coordinator_debug,
            "performance": performance_report,
            "doer_statistics": doer_stats,
            "evaluator_statistics": evaluator_stats
        }


# Global instance for use in other modules
enhanced_organizer = EnhancedOrganizer()


# Wrapper function for backward compatibility
async def run_enhanced_organizer_loop(request: AgentChatRequest, logger: Any, api_base_url: str) -> AsyncGenerator[str, None]:
    """Run enhanced organizer loop with comprehensive optimization."""
    async for result in enhanced_organizer.run_enhanced_organizer_loop(request, logger, api_base_url):
        yield result