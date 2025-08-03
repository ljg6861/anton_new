"""
Enhanced Evaluator Component

Implements sophisticated progress assessment with SUCCESS/PARTIAL/FAILURE levels,
tool execution validation, and comprehensive evaluation tracking.
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .enhanced_doer import DoerResponse, ExecutionStatus, ResponseType


class EvaluationLevel(Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL" 
    FAILURE = "FAILURE"
    DONE = "DONE"


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating progress."""
    tool_execution_required: bool = False
    min_information_gained: bool = False
    progress_toward_goal: bool = False
    relevant_to_instruction: bool = False
    builds_context: bool = False


@dataclass
class EvaluationResult:
    """Comprehensive evaluation result."""
    level: EvaluationLevel
    reason: str
    progress_score: float  # 0.0 to 1.0
    information_value: float  # 0.0 to 1.0
    tool_execution_success: bool
    criteria_met: EvaluationCriteria
    suggestions: List[str]
    evaluation_duration: float
    patterns_detected: List[str]


class EnhancedEvaluator:
    """
    Enhanced evaluator with sophisticated progress assessment.
    
    Features:
    - Three-level verdict system (SUCCESS/PARTIAL/FAILURE)
    - Tool execution validation
    - Progress measurement toward overall goal
    - Context building assessment
    - Evaluation history tracking
    - Pattern detection for failure modes
    """
    
    def __init__(self):
        # Evaluation tracking
        self.evaluation_history: List[EvaluationResult] = []
        self.pattern_tracker: Dict[str, int] = {}
        
        # Progress patterns
        self.progress_indicators = [
            r'(successfully|completed|found|discovered|identified|retrieved)',
            r'(file|directory|function|class|method|variable) (found|located|read|analyzed)',
            r'(error|issue|problem) (found|identified|detected)',
            r'(understand|learned|determined|confirmed)',
        ]
        
        # Failure patterns
        self.failure_indicators = [
            r'(failed|error|unable|cannot|could not)',
            r'(not found|does not exist|file not found)',
            r'(permission denied|access denied)',
            r'(timeout|timed out)',
        ]
        
        # Information value patterns
        self.high_value_patterns = [
            r'(source code|configuration|implementation|structure)',
            r'(function definition|class definition|method implementation)',
            r'(error message|stack trace|log entry)',
            r'(system information|version|status)',
        ]
    
    async def evaluate_doer_result(self,
                                 original_task: str,
                                 delegated_instruction: str,
                                 doer_response: DoerResponse,
                                 context_store: Dict,
                                 logger: Any) -> EvaluationResult:
        """
        Perform comprehensive evaluation of Doer result.
        
        Returns detailed evaluation with level, reasoning, and suggestions.
        """
        start_time = time.time()
        
        try:
            # Analyze tool execution
            tool_execution_success = self._analyze_tool_execution(doer_response)
            
            # Assess progress and information value
            progress_score = self._calculate_progress_score(
                delegated_instruction, doer_response, context_store
            )
            information_value = self._calculate_information_value(doer_response)
            
            # Evaluate criteria
            criteria = self._evaluate_criteria(
                original_task, delegated_instruction, doer_response, context_store
            )
            
            # Determine evaluation level
            level, reason = self._determine_evaluation_level(
                progress_score, information_value, tool_execution_success, criteria, doer_response
            )
            
            # Generate suggestions
            suggestions = self._generate_improvement_suggestions(
                level, criteria, doer_response, progress_score
            )
            
            # Detect patterns
            patterns = self._detect_evaluation_patterns(doer_response, level)
            
            # Create evaluation result
            evaluation_result = EvaluationResult(
                level=level,
                reason=reason,
                progress_score=progress_score,
                information_value=information_value,
                tool_execution_success=tool_execution_success,
                criteria_met=criteria,
                suggestions=suggestions,
                evaluation_duration=time.time() - start_time,
                patterns_detected=patterns
            )
            
            # Track evaluation history
            self.evaluation_history.append(evaluation_result)
            
            # Update pattern tracking
            for pattern in patterns:
                self.pattern_tracker[pattern] = self.pattern_tracker.get(pattern, 0) + 1
            
            logger.info(f"Evaluation completed: {level.value} (score: {progress_score:.2f})")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}", exc_info=True)
            
            # Return failure evaluation on error
            return EvaluationResult(
                level=EvaluationLevel.FAILURE,
                reason=f"Evaluation error: {str(e)}",
                progress_score=0.0,
                information_value=0.0,
                tool_execution_success=False,
                criteria_met=EvaluationCriteria(),
                suggestions=["Fix evaluation system error"],
                evaluation_duration=time.time() - start_time,
                patterns_detected=["evaluation_error"]
            )
    
    def _analyze_tool_execution(self, doer_response: DoerResponse) -> bool:
        """Analyze if tool execution was successful."""
        if not doer_response.tool_calls:
            # No tools called - this is okay if it's a final answer
            return doer_response.response_type == ResponseType.FINAL_ANSWER
        
        # Check if all tool calls succeeded
        return all(
            tool_call.status == ExecutionStatus.SUCCESS 
            for tool_call in doer_response.tool_calls
        )
    
    def _calculate_progress_score(self, 
                                delegated_instruction: str,
                                doer_response: DoerResponse,
                                context_store: Dict) -> float:
        """Calculate progress score toward the delegated instruction (0.0 to 1.0)."""
        score = 0.0
        
        # Base score for successful execution
        if doer_response.execution_status == ExecutionStatus.SUCCESS:
            score += 0.3
        
        # Score for tool execution
        if doer_response.tool_calls:
            successful_tools = sum(
                1 for tool_call in doer_response.tool_calls 
                if tool_call.status == ExecutionStatus.SUCCESS
            )
            tool_score = successful_tools / len(doer_response.tool_calls)
            score += 0.3 * tool_score
        
        # Score for information content
        content_lower = doer_response.content.lower()
        
        # Check for progress indicators
        progress_matches = sum(
            1 for pattern in self.progress_indicators 
            if re.search(pattern, content_lower, re.IGNORECASE)
        )
        score += min(progress_matches * 0.1, 0.2)
        
        # Check for high-value information
        value_matches = sum(
            1 for pattern in self.high_value_patterns 
            if re.search(pattern, content_lower, re.IGNORECASE)
        )
        score += min(value_matches * 0.05, 0.2)
        
        # Penalty for failure indicators
        failure_matches = sum(
            1 for pattern in self.failure_indicators 
            if re.search(pattern, content_lower, re.IGNORECASE)
        )
        score -= min(failure_matches * 0.1, 0.3)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_information_value(self, doer_response: DoerResponse) -> float:
        """Calculate information value of the response (0.0 to 1.0)."""
        if not doer_response.content:
            return 0.0
        
        content_lower = doer_response.content.lower()
        value_score = 0.0
        
        # Length-based value (longer responses generally contain more info)
        length_score = min(len(doer_response.content) / 1000, 0.3)  # Max 0.3 for length
        value_score += length_score
        
        # High-value pattern detection
        for pattern in self.high_value_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                value_score += 0.2
        
        # Tool result value
        if doer_response.tool_calls:
            for tool_call in doer_response.tool_calls:
                if tool_call.status == ExecutionStatus.SUCCESS and tool_call.result:
                    # Successful tool calls with results are valuable
                    value_score += 0.1
        
        return min(1.0, value_score)
    
    def _evaluate_criteria(self,
                          original_task: str,
                          delegated_instruction: str, 
                          doer_response: DoerResponse,
                          context_store: Dict) -> EvaluationCriteria:
        """Evaluate specific criteria for progress assessment."""
        criteria = EvaluationCriteria()
        
        # Tool execution required and successful
        if doer_response.tool_calls:
            criteria.tool_execution_required = True
            # Successful if at least one tool succeeded
            criteria.progress_toward_goal = any(
                tool_call.status == ExecutionStatus.SUCCESS 
                for tool_call in doer_response.tool_calls
            )
        else:
            # No tools required if it's a final answer
            criteria.tool_execution_required = False
            criteria.progress_toward_goal = doer_response.response_type == ResponseType.FINAL_ANSWER
        
        # Information gained assessment
        criteria.min_information_gained = self._calculate_information_value(doer_response) > 0.2
        
        # Relevance to instruction
        criteria.relevant_to_instruction = self._assess_relevance(
            delegated_instruction, doer_response
        )
        
        # Context building
        criteria.builds_context = self._assess_context_building(doer_response, context_store)
        
        return criteria
    
    def _assess_relevance(self, delegated_instruction: str, doer_response: DoerResponse) -> bool:
        """Assess if the response is relevant to the delegated instruction."""
        instruction_words = set(delegated_instruction.lower().split())
        response_words = set(doer_response.content.lower().split())
        
        # Calculate word overlap
        overlap = len(instruction_words.intersection(response_words))
        overlap_ratio = overlap / len(instruction_words) if instruction_words else 0
        
        return overlap_ratio > 0.2  # At least 20% word overlap
    
    def _assess_context_building(self, doer_response: DoerResponse, context_store: Dict) -> bool:
        """Assess if the response builds useful context."""
        # Check if new files were explored
        if any(tool_call.tool_name in ['read_file', 'list_directory'] 
               for tool_call in doer_response.tool_calls):
            return True
        
        # Check if substantial information was provided
        return len(doer_response.content) > 200  # At least 200 characters of content
    
    def _determine_evaluation_level(self,
                                  progress_score: float,
                                  information_value: float,
                                  tool_execution_success: bool,
                                  criteria: EvaluationCriteria,
                                  doer_response: DoerResponse) -> Tuple[EvaluationLevel, str]:
        """Determine the evaluation level and provide reasoning."""
        
        # Check for DONE (task completion)
        if (doer_response.response_type == ResponseType.FINAL_ANSWER and 
            progress_score > 0.7 and 
            information_value > 0.5):
            return EvaluationLevel.DONE, "Task appears to be completed with comprehensive final answer"
        
        # Check for SUCCESS
        if (progress_score >= 0.7 and 
            tool_execution_success and 
            criteria.min_information_gained and 
            criteria.relevant_to_instruction):
            return EvaluationLevel.SUCCESS, f"Strong progress made with score {progress_score:.2f}"
        
        # Check for PARTIAL success
        if (progress_score >= 0.4 or 
            (tool_execution_success and criteria.builds_context) or
            (information_value > 0.4 and criteria.relevant_to_instruction)):
            reasons = []
            if progress_score >= 0.4:
                reasons.append(f"moderate progress (score: {progress_score:.2f})")
            if tool_execution_success and criteria.builds_context:
                reasons.append("successful context building")
            if information_value > 0.4:
                reasons.append("valuable information gathered")
            
            return EvaluationLevel.PARTIAL, f"Partial progress: {', '.join(reasons)}"
        
        # Default to FAILURE
        failure_reasons = []
        if progress_score < 0.4:
            failure_reasons.append(f"low progress score ({progress_score:.2f})")
        if not tool_execution_success and doer_response.tool_calls:
            failure_reasons.append("tool execution failures")
        if not criteria.min_information_gained:
            failure_reasons.append("insufficient information gained")
        if not criteria.relevant_to_instruction:
            failure_reasons.append("response not relevant to instruction")
        
        return EvaluationLevel.FAILURE, f"Insufficient progress: {', '.join(failure_reasons)}"
    
    def _generate_improvement_suggestions(self,
                                        level: EvaluationLevel,
                                        criteria: EvaluationCriteria,
                                        doer_response: DoerResponse,
                                        progress_score: float) -> List[str]:
        """Generate specific suggestions for improvement."""
        suggestions = []
        
        if level == EvaluationLevel.FAILURE:
            if not criteria.tool_execution_required or doer_response.execution_status != ExecutionStatus.SUCCESS:
                suggestions.append("Use specific tools to gather information before providing results")
            
            if not criteria.min_information_gained:
                suggestions.append("Provide more detailed information or analysis")
            
            if not criteria.relevant_to_instruction:
                suggestions.append("Focus response more directly on the specific instruction given")
            
            if doer_response.tool_calls:
                failed_tools = [tc.tool_name for tc in doer_response.tool_calls 
                               if tc.status != ExecutionStatus.SUCCESS]
                if failed_tools:
                    suggestions.append(f"Fix tool execution errors for: {', '.join(failed_tools)}")
        
        elif level == EvaluationLevel.PARTIAL:
            suggestions.append("Build on current progress with additional investigation or detail")
            
            if progress_score < 0.6:
                suggestions.append("Gather more comprehensive information to reach full success")
        
        elif level == EvaluationLevel.SUCCESS:
            suggestions.append("Continue with the next logical step in the task")
        
        return suggestions
    
    def _detect_evaluation_patterns(self, doer_response: DoerResponse, level: EvaluationLevel) -> List[str]:
        """Detect patterns in evaluation results."""
        patterns = []
        
        # Response type patterns
        patterns.append(f"response_type_{doer_response.response_type.value}")
        patterns.append(f"evaluation_level_{level.value}")
        
        # Tool usage patterns
        if doer_response.tool_calls:
            tool_names = [tc.tool_name for tc in doer_response.tool_calls]
            patterns.append(f"tools_used_{len(tool_names)}")
            patterns.extend(f"tool_{name}" for name in set(tool_names))
        else:
            patterns.append("no_tools_used")
        
        # Execution status patterns
        patterns.append(f"execution_{doer_response.execution_status.value}")
        
        return patterns
    
    def get_evaluation_statistics(self) -> Dict:
        """Get comprehensive evaluation statistics."""
        if not self.evaluation_history:
            return {}
        
        total_evaluations = len(self.evaluation_history)
        
        # Level distribution
        level_counts = {}
        for level in EvaluationLevel:
            count = sum(1 for eval_result in self.evaluation_history if eval_result.level == level)
            level_counts[level.value] = count
        
        # Average scores
        avg_progress_score = sum(e.progress_score for e in self.evaluation_history) / total_evaluations
        avg_information_value = sum(e.information_value for e in self.evaluation_history) / total_evaluations
        avg_duration = sum(e.evaluation_duration for e in self.evaluation_history) / total_evaluations
        
        # Success rates
        success_rate = level_counts.get('SUCCESS', 0) / total_evaluations
        partial_rate = level_counts.get('PARTIAL', 0) / total_evaluations
        failure_rate = level_counts.get('FAILURE', 0) / total_evaluations
        
        return {
            'total_evaluations': total_evaluations,
            'level_distribution': level_counts,
            'average_scores': {
                'progress_score': avg_progress_score,
                'information_value': avg_information_value,
                'evaluation_duration': avg_duration
            },
            'success_rates': {
                'success_rate': success_rate,
                'partial_rate': partial_rate, 
                'failure_rate': failure_rate,
                'combined_success_rate': success_rate + partial_rate
            },
            'pattern_frequency': self.pattern_tracker.copy()
        }
    
    def reset_statistics(self):
        """Reset all evaluation statistics."""
        self.evaluation_history.clear()
        self.pattern_tracker.clear()