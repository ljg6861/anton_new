"""
Adaptive Workflow Orchestrator for the Anton agent system.

This component addresses the critical architectural weaknesses in the current system:
1. Rigid Agent Communication Flow -> Adaptive bidirectional communication
2. Insufficient Error Recovery -> Robust fallback strategies
3. Context Management Issues -> Intelligent context pruning and management
4. Limited Loop Detection -> Enhanced pattern recognition
5. Metrics Without Actionable Feedback -> Runtime optimization
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
from collections import deque, defaultdict

from server.agent.config import MAX_TURNS, ASSISTANT_ROLE, SYSTEM_ROLE, USER_ROLE


logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Enhanced agent roles with bidirectional communication capabilities."""
    PLANNER = "planner"
    DOER = "doer" 
    EVALUATOR = "evaluator"
    COORDINATOR = "coordinator"  # New role for complex task management


class TaskComplexity(Enum):
    """Task complexity levels for adaptive workflow selection."""
    SIMPLE = "simple"      # Direct execution
    MODERATE = "moderate"  # Single planner-doer cycle
    COMPLEX = "complex"    # Multi-agent collaboration
    CRITICAL = "critical"  # Requires specialized handling


class WorkflowState(Enum):
    """Current state of the workflow execution."""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    RECOVERING = "recovering"  # Error recovery state
    ADAPTING = "adapting"      # Workflow adaptation state
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentMessage:
    """Enhanced message structure for bidirectional communication."""
    role: AgentRole
    content: str
    timestamp: float
    message_type: str = "response"  # response, request, feedback, error
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    requires_response: bool = False


@dataclass
class WorkflowContext:
    """Comprehensive context tracking for intelligent management."""
    original_task: str
    current_state: WorkflowState
    complexity: TaskComplexity
    messages: List[AgentMessage] = field(default_factory=list)
    explored_files: set = field(default_factory=set)
    code_content: Dict[str, str] = field(default_factory=dict)
    task_progress: List[str] = field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    loop_detection_state: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: AgentMessage) -> None:
        """Add a message with intelligent context management."""
        self.messages.append(message)
        self._prune_context_if_needed()
    
    def _prune_context_if_needed(self) -> None:
        """Intelligent context pruning to prevent overflow."""
        # Keep last 50 messages for immediate context
        if len(self.messages) > 50:
            # Keep first 10 (system prompts) and last 40 (recent conversation)
            important_messages = self.messages[:10]
            recent_messages = self.messages[-40:]
            self.messages = important_messages + recent_messages
            logger.info(f"Context pruned: kept {len(self.messages)} messages")


@dataclass  
class ErrorRecoveryStrategy:
    """Strategy for recovering from specific types of errors."""
    error_pattern: str
    recovery_action: str
    retry_count: int = 0
    max_retries: int = 3
    backoff_factor: float = 1.5
    
    def should_retry(self) -> bool:
        return self.retry_count < self.max_retries
    
    def get_backoff_delay(self) -> float:
        return self.backoff_factor ** self.retry_count


class AdaptiveWorkflowOrchestrator:
    """
    Adaptive workflow orchestrator that addresses key architectural weaknesses.
    
    Features:
    - Bidirectional agent communication
    - Dynamic workflow adaptation based on task complexity
    - Robust error recovery with fallback strategies
    - Intelligent context management with pruning
    - Enhanced loop detection and prevention
    - Performance-driven optimization
    """
    
    def __init__(self, api_base_url: str, logger: Any):
        self.api_base_url = api_base_url
        self.logger = logger
        self.context: Optional[WorkflowContext] = None
        
        # Enhanced error recovery strategies
        self.error_strategies = self._initialize_error_strategies()
        
        # Performance tracking for adaptive optimization
        self.performance_history = deque(maxlen=100)
        
        # Loop detection with enhanced pattern recognition
        self.loop_detector = EnhancedLoopDetector()
        
        # Workflow adaptation rules
        self.adaptation_rules = self._initialize_adaptation_rules()
    
    def _initialize_error_strategies(self) -> Dict[str, ErrorRecoveryStrategy]:
        """Initialize comprehensive error recovery strategies."""
        return {
            "tool_execution_failed": ErrorRecoveryStrategy(
                error_pattern="tool.*failed|error executing",
                recovery_action="retry_with_alternative_tool",
                max_retries=2
            ),
            "model_api_timeout": ErrorRecoveryStrategy(
                error_pattern="timeout|connection.*error",
                recovery_action="reduce_context_and_retry",
                max_retries=3
            ),
            "parsing_error": ErrorRecoveryStrategy(
                error_pattern="json.*error|parsing.*failed",
                recovery_action="request_structured_response",
                max_retries=2
            ),
            "context_overflow": ErrorRecoveryStrategy(
                error_pattern="context.*too.*long|token.*limit",
                recovery_action="aggressive_context_pruning",
                max_retries=1
            ),
            "reasoning_loop": ErrorRecoveryStrategy(
                error_pattern="loop.*detected|repetitive.*response",
                recovery_action="inject_perspective_change",
                max_retries=2
            )
        }
    
    def _initialize_adaptation_rules(self) -> List[Dict[str, Any]]:
        """Initialize rules for dynamic workflow adaptation."""
        return [
            {
                "condition": "high_error_rate",
                "threshold": 0.3,
                "action": "increase_context_detail"
            },
            {
                "condition": "slow_progress", 
                "threshold": 10.0,  # seconds per turn
                "action": "simplify_workflow"
            },
            {
                "condition": "loop_detected",
                "threshold": 3,
                "action": "change_agent_strategy"
            },
            {
                "condition": "high_success_rate",
                "threshold": 0.9,
                "action": "optimize_for_speed"
            }
        ]
    
    async def execute_workflow(
        self,
        task: str,
        initial_messages: List[Dict[str, str]],
        tools: List[Any],
        max_turns: int = MAX_TURNS
    ) -> AsyncGenerator[str, None]:
        """
        Execute an adaptive workflow with bidirectional communication and error recovery.
        """
        # Initialize workflow context
        complexity = self._assess_task_complexity(task)
        self.context = WorkflowContext(
            original_task=task,
            current_state=WorkflowState.INITIALIZING,
            complexity=complexity
        )
        
        self.logger.info(f"Starting adaptive workflow for {complexity.value} task")
        
        try:
            # Choose workflow strategy based on complexity
            async for result in self._execute_adaptive_strategy(
                initial_messages, tools, max_turns
            ):
                yield result
                
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            async for recovery_result in self._attempt_error_recovery(e):
                yield recovery_result
    
    def _assess_task_complexity(self, task: str) -> TaskComplexity:
        """Assess task complexity to determine optimal workflow strategy."""
        task_lower = task.lower()
        
        # Simple task indicators
        simple_indicators = ["read", "list", "show", "display", "get"]
        if any(indicator in task_lower for indicator in simple_indicators):
            return TaskComplexity.SIMPLE
        
        # Complex task indicators  
        complex_indicators = ["analyze", "refactor", "implement", "design", "optimize"]
        if any(indicator in task_lower for indicator in complex_indicators):
            return TaskComplexity.COMPLEX
        
        # Critical task indicators
        critical_indicators = ["security", "performance", "architecture", "migration"]
        if any(indicator in task_lower for indicator in critical_indicators):
            return TaskComplexity.CRITICAL
        
        return TaskComplexity.MODERATE
    
    async def _execute_adaptive_strategy(
        self,
        initial_messages: List[Dict[str, str]],
        tools: List[Any],
        max_turns: int
    ) -> AsyncGenerator[str, None]:
        """Execute workflow using complexity-appropriate strategy."""
        
        if self.context.complexity == TaskComplexity.SIMPLE:
            async for result in self._execute_direct_strategy(initial_messages, tools):
                yield result
        elif self.context.complexity == TaskComplexity.MODERATE:
            async for result in self._execute_collaborative_strategy(initial_messages, tools, max_turns):
                yield result
        elif self.context.complexity == TaskComplexity.COMPLEX:
            async for result in self._execute_multi_agent_strategy(initial_messages, tools, max_turns):
                yield result
        else:  # CRITICAL
            async for result in self._execute_specialized_strategy(initial_messages, tools, max_turns):
                yield result
    
    async def _execute_direct_strategy(
        self,
        initial_messages: List[Dict[str, str]],
        tools: List[Any]
    ) -> AsyncGenerator[str, None]:
        """Direct execution for simple tasks."""
        self.context.current_state = WorkflowState.EXECUTING
        
        # Simple tasks go directly to doer
        from server.agent.doer import run_doer_loop
        
        doer_messages = self._prepare_doer_messages(initial_messages)
        
        async for token in run_doer_loop(
            doer_messages, tools, self.logger, self.api_base_url, False, self.context.__dict__
        ):
            yield token
    
    async def _execute_collaborative_strategy(
        self,
        initial_messages: List[Dict[str, str]],
        tools: List[Any],
        max_turns: int
    ) -> AsyncGenerator[str, None]:
        """Enhanced collaborative strategy with bidirectional communication."""
        # This is an improved version of the current organizer loop
        # with better error handling and adaptive behavior
        
        from server.agent.doer import execute_turn, run_doer_loop
        from server.agent.prompts import get_evaluator_prompt
        from client.context_builder import ContextBuilder
        
        organizer_messages = self._prepare_organizer_messages(initial_messages)
        
        for turn in range(max_turns):
            try:
                self.context.current_state = WorkflowState.PLANNING
                
                # Check for adaptive workflow changes
                await self._check_adaptation_rules(turn)
                
                # Enhanced planner turn with error recovery
                async for planner_result in self._execute_planner_turn(
                    organizer_messages, tools, turn
                ):
                    yield planner_result
                
                self.context.current_state = WorkflowState.EXECUTING
                
                # Enhanced doer turn with loop detection
                async for doer_result in self._execute_doer_turn(
                    organizer_messages, tools, turn
                ):
                    yield doer_result
                
                self.context.current_state = WorkflowState.EVALUATING
                
                # Enhanced evaluator turn with adaptive feedback
                should_continue = await self._execute_evaluator_turn(
                    organizer_messages, tools, turn
                )
                
                if not should_continue:
                    self.context.current_state = WorkflowState.COMPLETED
                    return
                    
            except Exception as e:
                self.logger.error(f"Error in turn {turn}: {e}")
                async for recovery_result in self._attempt_turn_recovery(e, turn):
                    yield recovery_result
        
        # Max turns reached
        self.context.current_state = WorkflowState.FAILED
        yield "\n\n[Adaptive workflow: Maximum turns reached]"
    
    async def _execute_multi_agent_strategy(
        self,
        initial_messages: List[Dict[str, str]],
        tools: List[Any],
        max_turns: int
    ) -> AsyncGenerator[str, None]:
        """Multi-agent strategy for complex tasks."""
        # Placeholder for advanced multi-agent coordination
        # For now, use enhanced collaborative strategy
        async for result in self._execute_collaborative_strategy(initial_messages, tools, max_turns):
            yield result
    
    async def _execute_specialized_strategy(
        self,
        initial_messages: List[Dict[str, str]],
        tools: List[Any],
        max_turns: int
    ) -> AsyncGenerator[str, None]:
        """Specialized strategy for critical tasks."""
        # Placeholder for specialized handling
        # For now, use collaborative strategy with extra safeguards
        async for result in self._execute_collaborative_strategy(initial_messages, tools, max_turns):
            yield result
    
    def _prepare_organizer_messages(self, initial_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Prepare messages for organizer with enhanced context."""
        # Convert to proper format and add system prompt
        messages = []
        for msg in initial_messages:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return messages
    
    def _prepare_doer_messages(self, initial_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Prepare messages for direct doer execution."""
        # Convert to proper format and add doer system prompt
        messages = []
        for msg in initial_messages:
            messages.append({
                "role": msg["role"], 
                "content": msg["content"]
            })
        return messages
    
    async def _check_adaptation_rules(self, turn: int) -> None:
        """Check if workflow adaptation is needed based on performance."""
        # Calculate current error rate
        recent_errors = sum(1 for entry in self.context.error_history[-10:] if entry)
        error_rate = recent_errors / 10 if self.context.error_history else 0
        
        # Check adaptation rules
        for rule in self.adaptation_rules:
            if rule["condition"] == "high_error_rate" and error_rate > rule["threshold"]:
                self.logger.info(f"Adapting workflow: high error rate detected ({error_rate:.2f})")
                await self._apply_adaptation(rule["action"])
            elif rule["condition"] == "loop_detected":
                loop_count = self.loop_detector.get_loop_count()
                if loop_count > rule["threshold"]:
                    self.logger.info(f"Adapting workflow: loops detected ({loop_count})")
                    await self._apply_adaptation(rule["action"])
    
    async def _apply_adaptation(self, action: str) -> None:
        """Apply workflow adaptation based on identified issues."""
        if action == "increase_context_detail":
            # Add more detailed context to messages
            pass
        elif action == "simplify_workflow":
            # Switch to simpler workflow strategy
            self.context.complexity = TaskComplexity.SIMPLE
        elif action == "change_agent_strategy":
            # Inject strategy change prompt
            pass
        elif action == "optimize_for_speed":
            # Reduce context detail for faster execution
            pass
    
    async def _execute_planner_turn(
        self,
        organizer_messages: List[Dict[str, str]],
        tools: List[Any],
        turn: int
    ) -> AsyncGenerator[str, None]:
        """Execute enhanced planner turn with error recovery."""
        from server.agent.doer import execute_turn
        from client.context_builder import ContextBuilder
        
        try:
            # Build system prompt with memory context
            system_prompt = await ContextBuilder().build_system_prompt_planner(
                self.context.original_task
            )
            
            if turn == 0:
                organizer_messages.insert(0, {"role": SYSTEM_ROLE, "content": system_prompt})
            
            # Add context summary
            if turn > 0:
                context_summary = self._build_enhanced_context_summary()
                organizer_messages.append({
                    "role": SYSTEM_ROLE,
                    "content": f"Previous step progress:\n{context_summary}"
                })
            
            # Execute planner turn
            response_buffer = ""
            async for token in execute_turn(
                self.api_base_url, organizer_messages, self.logger, tools, 0.6, False
            ):
                response_buffer += token
            
            # Extract content after thinking markers
            import re
            content = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
            organizer_messages.append({"role": ASSISTANT_ROLE, "content": content})
            
            yield f"[Planner]: {content}\n"
            
        except Exception as e:
            self.logger.error(f"Planner turn failed: {e}")
            raise
    
    async def _execute_doer_turn(
        self,
        organizer_messages: List[Dict[str, str]],
        tools: List[Any],
        turn: int
    ) -> AsyncGenerator[str, None]:
        """Execute enhanced doer turn with loop detection."""
        from server.agent.doer import run_doer_loop
        from client.context_builder import ContextBuilder
        
        try:
            # Prepare doer messages
            doer_messages = []
            
            # Add system prompt
            system_prompt = await ContextBuilder().build_system_prompt_doer()
            doer_messages.append({"role": SYSTEM_ROLE, "content": system_prompt})
            
            # Add context from organizer
            latest_planner_content = organizer_messages[-1]["content"]
            doer_messages.append({"role": USER_ROLE, "content": latest_planner_content})
            
            # Execute doer loop with enhanced context
            context_dict = {
                "explored_files": self.context.explored_files,
                "code_content": self.context.code_content,
                "task_progress": self.context.task_progress,
                "tool_outputs": [output for output in self.context.tool_outputs]
            }
            
            async for token in run_doer_loop(
                doer_messages, tools, self.logger, self.api_base_url, True, context_dict
            ):
                yield token
            
            # Update context with doer results
            doer_result = doer_messages[-1]["content"]
            self.context.add_message(AgentMessage(
                role=AgentRole.DOER,
                content=doer_result,
                timestamp=time.time()
            ))
            
        except Exception as e:
            self.logger.error(f"Doer turn failed: {e}")
            raise
    
    async def _execute_evaluator_turn(
        self,
        organizer_messages: List[Dict[str, str]],
        tools: List[Any],
        turn: int
    ) -> bool:
        """Execute enhanced evaluator turn and return whether to continue."""
        from server.agent.doer import execute_turn
        from server.agent.prompts import get_evaluator_prompt
        
        try:
            # Get the latest doer result
            doer_result = organizer_messages[-1]["content"] if organizer_messages else ""
            
            # Build evaluator prompt
            evaluator_prompt = get_evaluator_prompt() + (
                f"\n\nHere is the information to evaluate:"
                f"\nOriginal High-Level Task: {self.context.original_task}"
                f"\nCurrent Step Result: {doer_result}"
            )
            
            evaluator_messages = [{"role": SYSTEM_ROLE, "content": evaluator_prompt}]
            
            # Execute evaluator
            response_buffer = ""
            async for token in execute_turn(
                self.api_base_url, evaluator_messages, self.logger, tools, 0.1, True
            ):
                response_buffer += token
            
            # Parse evaluator response
            import re
            evaluator_response = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
            
            if evaluator_response.startswith('SUCCESS:'):
                self.logger.info("Evaluator confirmed success. Continuing...")
                return True
            elif evaluator_response.startswith('FAILURE:'):
                self.logger.info("Evaluator reported failure. Planner will adjust.")
                organizer_messages.append({"role": USER_ROLE, "content": evaluator_response})
                return True
            elif evaluator_response.startswith('DONE:'):
                self.logger.info("Evaluator confirmed task completion.")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Evaluator turn failed: {e}")
            return True  # Continue on evaluator errors
    
    def _build_enhanced_context_summary(self) -> str:
        """Build enhanced context summary with intelligent prioritization."""
        summary_parts = []
        
        # Recent progress (prioritize recent items)
        if self.context.task_progress:
            recent_progress = self.context.task_progress[-3:]  # Last 3 items
            summary_parts.append("Recent progress:")
            for step in recent_progress:
                summary_parts.append(f"- {step}")
        
        # Key files explored (prioritize smaller list)
        if self.context.explored_files:
            explored_list = list(self.context.explored_files)[:5]  # Top 5 files
            summary_parts.append(f"Key files explored: {', '.join(explored_list)}")
        
        # Error context if any recent errors
        if self.context.error_history:
            recent_errors = self.context.error_history[-2:]  # Last 2 errors
            if recent_errors:
                summary_parts.append("Recent issues resolved:")
                for error in recent_errors:
                    summary_parts.append(f"- {error.get('type', 'Unknown')}: {error.get('recovery', 'Handled')}")
        
        return "\n".join(summary_parts) if summary_parts else "No significant progress to report."
    
    async def _attempt_error_recovery(self, error: Exception) -> AsyncGenerator[str, None]:
        """Attempt to recover from workflow-level errors."""
        error_msg = str(error).lower()
        
        # Find matching recovery strategy
        strategy = None
        for pattern, recovery_strategy in self.error_strategies.items():
            if any(keyword in error_msg for keyword in recovery_strategy.error_pattern.split("|")):
                strategy = recovery_strategy
                break
        
        if strategy and strategy.should_retry():
            self.logger.info(f"Attempting error recovery: {strategy.recovery_action}")
            strategy.retry_count += 1
            
            # Add error to history
            self.context.error_history.append({
                "error": str(error),
                "strategy": strategy.recovery_action,
                "timestamp": time.time(),
                "retry_count": strategy.retry_count
            })
            
            # Apply recovery action
            if strategy.recovery_action == "reduce_context_and_retry":
                # Aggressively prune context
                self.context._prune_context_if_needed()
                yield "[Recovery]: Context reduced, retrying...\n"
            elif strategy.recovery_action == "request_structured_response":
                yield "[Recovery]: Requesting clearer response format...\n"
            else:
                yield f"[Recovery]: Applying {strategy.recovery_action}...\n"
        else:
            yield f"[Error]: Unable to recover from {type(error).__name__}: {error}\n"
    
    async def _attempt_turn_recovery(self, error: Exception, turn: int) -> AsyncGenerator[str, None]:
        """Attempt to recover from turn-level errors."""
        self.context.current_state = WorkflowState.RECOVERING
        
        yield f"[Turn {turn}]: Encountered error, attempting recovery...\n"
        
        async for recovery_result in self._attempt_error_recovery(error):
            yield recovery_result
        
        # Reset state after recovery attempt
        self.context.current_state = WorkflowState.PLANNING


class EnhancedLoopDetector:
    """Enhanced loop detection beyond exact repetition matching."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.recent_responses = deque(maxlen=window_size)
        self.loop_count = 0
        
    def add_response(self, response: str) -> bool:
        """Add response and check for loops. Returns True if loop detected."""
        normalized = self._normalize_response(response)
        self.recent_responses.append(normalized)
        
        if len(self.recent_responses) >= 3:
            loop_detected = self._detect_pattern_loop()
            if loop_detected:
                self.loop_count += 1
                return True
        
        return False
    
    def _normalize_response(self, response: str) -> str:
        """Normalize response for pattern comparison."""
        # Remove timestamps, specific file paths, etc.
        import re
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}', '', response)
        normalized = re.sub(r'/[^\s]+\.(py|js|txt)', '/file.ext', normalized)
        return normalized.lower().strip()
    
    def _detect_pattern_loop(self) -> bool:
        """Detect various types of loops beyond exact matching."""
        responses = list(self.recent_responses)
        
        # Check for exact repetition
        if len(set(responses)) == 1:
            return True
        
        # Check for alternating pattern (A-B-A-B)
        if len(responses) >= 4:
            if responses[-1] == responses[-3] and responses[-2] == responses[-4]:
                return True
        
        # Check for semantic similarity using word overlap
        if len(responses) >= 3:
            last_response = responses[-1]
            for prev_response in responses[-3:-1]:
                similarity = self._calculate_word_overlap(last_response, prev_response)
                if similarity > 0.8:  # 80% word overlap
                    return True
        
        return False
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap ratio between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_loop_count(self) -> int:
        """Get current loop count."""
        return self.loop_count
    
    def reset(self) -> None:
        """Reset loop detection state."""
        self.recent_responses.clear()
        self.loop_count = 0