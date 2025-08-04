"""
Agent Coordination System

Central coordinator that manages interactions between Planner, Doer, and Evaluator
with comprehensive state tracking, loop detection, and performance monitoring.
"""

import time
import json
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .loop_detector import LoopDetector


class AgentType(Enum):
    PLANNER = "planner"
    DOER = "doer" 
    EVALUATOR = "evaluator"


class TaskStatus(Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"


@dataclass
class AgentInteraction:
    """Represents a single agent interaction."""
    agent_type: AgentType
    turn: int
    instruction: str
    response: str
    timestamp: float
    duration: float
    tool_calls_made: bool = False
    tool_results: List[Dict] = None
    evaluation_result: Optional[str] = None
    
    def __post_init__(self):
        if self.tool_results is None:
            self.tool_results = []


@dataclass
class CoordinatorState:
    """Complete state of the coordination system."""
    task_id: str
    original_task: str
    current_turn: int
    max_iterations: int
    status: TaskStatus
    interactions: List[AgentInteraction]
    context_store: Dict
    loop_detector_status: Dict
    start_time: float
    total_duration: float = 0.0
    loop_breaks_applied: int = 0
    performance_metrics: Dict = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


class AgentCoordinator:
    """
    Central coordinator managing all agent interactions with comprehensive tracking.
    
    Features:
    - Maximum iteration enforcement (configurable, default 10)
    - Comprehensive state tracking and history
    - Loop detection and pattern breaking
    - Performance monitoring and optimization
    - Detailed debugging information
    """
    
    def __init__(self, 
                 max_iterations: int = 10,
                 similarity_threshold: float = 0.85,
                 enable_performance_monitoring: bool = True):
        self.max_iterations = max_iterations
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # Initialize loop detector
        self.loop_detector = LoopDetector(
            similarity_threshold=similarity_threshold,
            max_history_size=20,
            loop_detection_window=5
        )
        
        # State tracking
        self.state: Optional[CoordinatorState] = None
        self.performance_tracker = {}
        self.current_interaction: Optional[AgentInteraction] = None
    
    def initialize_task(self, task_id: str, original_task: str, context_store: Dict = None) -> CoordinatorState:
        """Initialize a new task coordination session."""
        self.state = CoordinatorState(
            task_id=task_id,
            original_task=original_task,
            current_turn=0,
            max_iterations=self.max_iterations,
            status=TaskStatus.IN_PROGRESS,
            interactions=[],
            context_store=context_store or {},
            loop_detector_status={},
            start_time=time.time()
        )
        
        # Reset loop detector for new task
        self.loop_detector.reset()
        
        return self.state
    
    def start_agent_interaction(self, agent_type: AgentType, instruction: str) -> None:
        """Start tracking a new agent interaction."""
        if not self.state:
            raise ValueError("Task not initialized. Call initialize_task first.")
        
        self.current_interaction = AgentInteraction(
            agent_type=agent_type,
            turn=self.state.current_turn,
            instruction=instruction,
            response="",
            timestamp=time.time(),
            duration=0.0
        )
        
        # Add to loop detector
        self.loop_detector.add_instruction(
            content=instruction,
            turn=self.state.current_turn,
            timestamp=time.time(),
            agent=agent_type.value
        )
    
    def complete_agent_interaction(self, response: str, 
                                 tool_calls_made: bool = False,
                                 tool_results: List[Dict] = None,
                                 evaluation_result: str = None) -> AgentInteraction:
        """Complete the current agent interaction and add it to history."""
        if not self.current_interaction:
            raise ValueError("No active interaction to complete.")
        
        self.current_interaction.response = response
        self.current_interaction.duration = time.time() - self.current_interaction.timestamp
        self.current_interaction.tool_calls_made = tool_calls_made
        self.current_interaction.tool_results = tool_results or []
        self.current_interaction.evaluation_result = evaluation_result
        
        # Add to state history
        self.state.interactions.append(self.current_interaction)
        
        # Track performance if enabled
        if self.enable_performance_monitoring:
            self._track_performance(self.current_interaction)
        
        completed_interaction = self.current_interaction
        self.current_interaction = None
        
        return completed_interaction
    
    def check_for_loops(self) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Check for reasoning loops and get pattern-breaking instruction if needed."""
        is_loop, pattern_breaking_instruction, loop_info = self.loop_detector.detect_loop()
        
        if is_loop:
            self.state.loop_breaks_applied += 1
            self.state.loop_detector_status = self.loop_detector.get_status()
        
        return is_loop, pattern_breaking_instruction, loop_info
    
    def should_continue(self) -> Tuple[bool, str]:
        """
        Determine if the task should continue or stop.
        
        Returns:
            Tuple of (should_continue, reason)
        """
        if not self.state:
            return False, "Task not initialized"
        
        # Check iteration limit
        if self.state.current_turn >= self.max_iterations:
            self.state.status = TaskStatus.MAX_ITERATIONS_REACHED
            return False, f"Maximum iterations ({self.max_iterations}) reached"
        
        # Check if task is completed based on last evaluator response
        if self.state.interactions:
            last_interaction = self.state.interactions[-1]
            if (last_interaction.agent_type == AgentType.EVALUATOR and 
                last_interaction.evaluation_result and
                last_interaction.evaluation_result.startswith('DONE:')):
                self.state.status = TaskStatus.COMPLETED
                return False, "Task completed successfully"
        
        return True, "Task can continue"
    
    def advance_turn(self) -> int:
        """Advance to the next turn and return the new turn number."""
        if self.state:
            self.state.current_turn += 1
        return self.state.current_turn if self.state else 0
    
    def finalize_task(self, status: TaskStatus = None) -> CoordinatorState:
        """Finalize the task and calculate final metrics."""
        if not self.state:
            raise ValueError("No task to finalize")
        
        if status:
            self.state.status = status
        
        self.state.total_duration = time.time() - self.state.start_time
        self.state.loop_detector_status = self.loop_detector.get_status()
        
        if self.enable_performance_monitoring:
            self.state.performance_metrics = self._generate_performance_report()
        
        return self.state
    
    def get_task_history(self) -> List[Dict]:
        """Get complete task history in a serializable format."""
        if not self.state:
            return []
        
        return [
            {
                'agent_type': interaction.agent_type.value,
                'turn': interaction.turn,
                'instruction': interaction.instruction,
                'response': interaction.response,
                'timestamp': interaction.timestamp,
                'duration': interaction.duration,
                'tool_calls_made': interaction.tool_calls_made,
                'tool_results': interaction.tool_results,
                'evaluation_result': interaction.evaluation_result
            }
            for interaction in self.state.interactions
        ]
    
    def get_debugging_info(self) -> Dict:
        """Get comprehensive debugging information."""
        if not self.state:
            return {}
        
        return {
            'task_summary': {
                'task_id': self.state.task_id,
                'original_task': self.state.original_task,
                'status': self.state.status.value,
                'current_turn': self.state.current_turn,
                'total_duration': self.state.total_duration,
                'interaction_count': len(self.state.interactions)
            },
            'loop_detection': self.state.loop_detector_status,
            'performance_metrics': self.state.performance_metrics,
            'recent_interactions': self.get_task_history()[-5:],  # Last 5 interactions
            'context_store_summary': {
                'explored_files_count': len(self.state.context_store.get('explored_files', set())),
                'code_content_count': len(self.state.context_store.get('code_content', {})),
                'tool_outputs_count': len(self.state.context_store.get('tool_outputs', []))
            }
        }
    
    def _track_performance(self, interaction: AgentInteraction):
        """Track performance metrics for an interaction."""
        agent_type = interaction.agent_type.value
        
        if agent_type not in self.performance_tracker:
            self.performance_tracker[agent_type] = {
                'total_duration': 0.0,
                'interaction_count': 0,
                'tool_calls_count': 0,
                'average_duration': 0.0
            }
        
        metrics = self.performance_tracker[agent_type]
        metrics['total_duration'] += interaction.duration
        metrics['interaction_count'] += 1
        metrics['average_duration'] = metrics['total_duration'] / metrics['interaction_count']
        
        if interaction.tool_calls_made:
            metrics['tool_calls_count'] += 1
    
    def _generate_performance_report(self) -> Dict:
        """Generate a comprehensive performance report."""
        total_interactions = len(self.state.interactions)
        
        if total_interactions == 0:
            return {}
        
        # Calculate overall metrics
        total_duration = sum(i.duration for i in self.state.interactions)
        avg_interaction_duration = total_duration / total_interactions
        
        # Agent-specific metrics
        agent_metrics = {}
        for agent_type in AgentType:
            agent_interactions = [i for i in self.state.interactions if i.agent_type == agent_type]
            if agent_interactions:
                agent_metrics[agent_type.value] = {
                    'interaction_count': len(agent_interactions),
                    'total_duration': sum(i.duration for i in agent_interactions),
                    'average_duration': sum(i.duration for i in agent_interactions) / len(agent_interactions),
                    'tool_calls_made': sum(1 for i in agent_interactions if i.tool_calls_made)
                }
        
        # Performance analysis
        analysis = {
            'bottlenecks': [],
            'optimization_suggestions': [],
            'efficiency_score': 0.0
        }
        
        # Identify bottlenecks
        for agent_type, metrics in agent_metrics.items():
            if metrics['average_duration'] > 30.0:  # More than 30 seconds average
                analysis['bottlenecks'].append(f"{agent_type} agent averaging {metrics['average_duration']:.1f}s per interaction")
        
        # Generate optimization suggestions
        if self.state.loop_breaks_applied > 0:
            analysis['optimization_suggestions'].append(f"Reduce reasoning loops (detected {self.state.loop_breaks_applied} loop breaks)")
        
        if total_duration > 300:  # More than 5 minutes total
            analysis['optimization_suggestions'].append("Consider breaking down complex tasks into smaller steps")
        
        # Calculate efficiency score (0-100)
        base_score = 100
        base_score -= min(self.state.loop_breaks_applied * 10, 50)  # Penalize loops
        base_score -= min((total_duration - 60) / 10, 30) if total_duration > 60 else 0  # Penalize long duration
        analysis['efficiency_score'] = max(base_score, 0)
        
        return {
            'overall': {
                'total_interactions': total_interactions,
                'total_duration': total_duration,
                'average_interaction_duration': avg_interaction_duration,
                'task_completion_time': self.state.total_duration
            },
            'agent_metrics': agent_metrics,
            'analysis': analysis
        }