"""
Minimal State Machine Orchestrator

Implements a tiny state machine that runs one linear node sequence:
PLANNING → EXECUTING → CRITIQUING → DONE/FAILED

Now uses Swarm-style deterministic execution instead of ReAct looping.
This provides deterministic task execution with explicit handoffs.
"""
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from server.agent.state import State, make_state, Budgets, AgentStatus, Evidence
from server.agent.swarm_execution import branch_executor, ExecutionFlow
from server.agent.evaluator_node import EvaluatorNode, AcceptanceCriteria, get_evaluator_node

logger = logging.getLogger(__name__)


class OrchestrationStatus(Enum):
    """Status values for the orchestration state machine"""
    PLANNING = "planning"
    EXECUTING = "executing"
    CRITIQUING = "critiquing"
    DONE = "done"
    FAILED = "failed"


class RouteType(Enum):
    """Route types for different orchestration patterns"""
    MINIMAL_FLOW = "minimal_flow"
    # Future: COMPLEX_FLOW, MULTI_AGENT_FLOW, etc.


@dataclass
class OrchestrationResult:
    """Result from the orchestration state machine"""
    answer: Optional[str] = None
    state: Optional[State] = None
    error: Optional[str] = None
    status: OrchestrationStatus = OrchestrationStatus.FAILED
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TaskOrchestrator:
    """
    Minimal state machine orchestrator that coordinates task execution
    through a linear sequence of states.
    """
    
    def __init__(self, 
                 api_base: str = "http://localhost:8000/v1",
                 model_name: str = "gpt-4",
                 llm_client=None):
        self.api_base = api_base
        self.model_name = model_name
        self.llm_client = llm_client
        
    async def run_task(self,
                      goal: str,
                      budgets: Budgets,
                      acceptance: AcceptanceCriteria,
                      user_id: str = "default_user") -> OrchestrationResult:
        """
        Run a task through the state machine orchestrator.
        
        Implements the exact flow specified:
        PLANNING → EXECUTING → CRITIQUING → DONE/FAILED
        
        Args:
            goal: The goal to achieve
            budgets: Token and iteration budgets
            acceptance: Acceptance criteria for evaluation
            user_id: User identifier
            
        Returns:
            OrchestrationResult with answer, state, and metadata
        """
        logger.info(f"Starting task orchestration for goal: {goal[:100]}...")
        
        # Initialize state
        S = make_state(goal, budgets, user_id)
        S.status = AgentStatus.PLANNING  # We'll track orchestration status separately
        orchestration_status = OrchestrationStatus.PLANNING
        
        # Variables for state machine
        tmp_answer = None
        evaluator = None
        
        # Main state machine loop
        iteration = 0
        max_iterations = 10  # Safety limit
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Orchestration iteration {iteration}, status: {orchestration_status.value}")
            
            if orchestration_status == OrchestrationStatus.PLANNING:
                logger.info("PLANNING: Setting up Swarm-style deterministic execution")
                
                # Set subgoals and route for deterministic flow
                S.subgoals = ["prepare_inputs", "execute", "postprocess"]
                S.route = RouteType.MINIMAL_FLOW.value
                
                # Transition to EXECUTING
                orchestration_status = OrchestrationStatus.EXECUTING
                logger.info("Transitioning to EXECUTING with deterministic flow")
                
            elif orchestration_status == OrchestrationStatus.EXECUTING:
                logger.info("EXECUTING: Running Swarm-style deterministic execution")
                
                try:
                    # Use the new branch_executor for deterministic execution
                    execution_flow = await branch_executor(
                        state=S,
                        acceptance_criteria=acceptance,
                        llm_client=self.llm_client
                    )
                    
                    if execution_flow.success:
                        tmp_answer = execution_flow.final_output
                        S.cost += execution_flow.total_cost
                        
                        # Add evidence from execution flow
                        if tmp_answer:
                            evidence_entry = Evidence(
                                type="DOC",
                                content=tmp_answer,
                                source="swarm_execution",
                                metadata={
                                    "ref": "swarm_answer", 
                                    "summary": tmp_answer[:200] if len(tmp_answer) > 200 else tmp_answer,
                                    "execution_steps": len(execution_flow.step_results),
                                    "strategies_used": execution_flow.metadata.get("strategies_used", [])
                                }
                            )
                            S.evidence.append(evidence_entry)
                        
                        # Transition to CRITIQUING
                        orchestration_status = OrchestrationStatus.CRITIQUING
                        logger.info(f"Swarm execution successful, answer length: {len(tmp_answer) if tmp_answer else 0}")
                        logger.info("Transitioning to CRITIQUING")
                        
                    else:
                        logger.error(f"Swarm execution failed: {execution_flow.error}")
                        return OrchestrationResult(
                            error=f"swarm_execution_failed: {execution_flow.error}",
                            state=S,
                            status=OrchestrationStatus.FAILED,
                            metadata={
                                "iteration": iteration, 
                                "failed_in": "EXECUTING",
                                "execution_flow": execution_flow.metadata
                            }
                        )
                    
                except Exception as e:
                    logger.error(f"Error in EXECUTING state: {e}")
                    return OrchestrationResult(
                        error=f"execution_exception: {str(e)}",
                        state=S,
                        status=OrchestrationStatus.FAILED,
                        metadata={"iteration": iteration, "failed_in": "EXECUTING"}
                    )
                    
            elif orchestration_status == OrchestrationStatus.CRITIQUING:
                logger.info("CRITIQUING: Evaluating answer")
                
                if not tmp_answer:
                    logger.error("No answer to critique")
                    return OrchestrationResult(
                        error="no_answer_to_critique",
                        state=S,
                        status=OrchestrationStatus.FAILED,
                        metadata={"iteration": iteration, "failed_in": "CRITIQUING"}
                    )
                
                try:
                    # Create evaluator if not exists
                    if evaluator is None:
                        if self.llm_client is None:
                            logger.error("No LLM client provided for evaluation")
                            return OrchestrationResult(
                                error="no_llm_client_for_evaluation",
                                state=S,
                                status=OrchestrationStatus.FAILED,
                                metadata={"iteration": iteration, "failed_in": "CRITIQUING"}
                            )
                        evaluator = get_evaluator_node(self.llm_client)
                    
                    # Evaluate the answer
                    evaluation = await evaluator.evaluate(tmp_answer, S, acceptance)
                    score = evaluation.overall_score
                    
                    logger.info(f"Evaluation score: {score:.2f}, threshold: {acceptance.min_score}")
                    
                    # Decision based on score
                    if score >= acceptance.min_score:
                        logger.info("Score meets threshold - task DONE")
                        orchestration_status = OrchestrationStatus.DONE
                        S.status = AgentStatus.DONE
                        return OrchestrationResult(
                            answer=tmp_answer,
                            state=S,
                            status=OrchestrationStatus.DONE,
                            metadata={
                                "iteration": iteration,
                                "final_score": score,
                                "evaluation_feedback": evaluation.feedback
                            }
                        )
                    else:
                        logger.info("Score below threshold - task FAILED")
                        orchestration_status = OrchestrationStatus.FAILED
                        S.status = AgentStatus.FAILED
                        return OrchestrationResult(
                            error="low_score",
                            state=S,
                            status=OrchestrationStatus.FAILED,
                            metadata={
                                "iteration": iteration,
                                "final_score": score,
                                "threshold": acceptance.min_score,
                                "evaluation_feedback": evaluation.feedback
                            }
                        )
                        
                except Exception as e:
                    logger.error(f"Error in CRITIQUING state: {e}")
                    return OrchestrationResult(
                        error=f"evaluation_failed: {str(e)}",
                        state=S,
                        status=OrchestrationStatus.FAILED,
                        metadata={"iteration": iteration, "failed_in": "CRITIQUING"}
                    )
                    
            else:
                # Should not reach here in minimal flow
                logger.error(f"Unexpected orchestration status: {orchestration_status}")
                return OrchestrationResult(
                    error=f"unexpected_status: {orchestration_status.value}",
                    state=S,
                    status=OrchestrationStatus.FAILED,
                    metadata={"iteration": iteration, "unexpected_status": orchestration_status.value}
                )
        
        # Max iterations reached
        logger.error(f"Max iterations ({max_iterations}) reached")
        return OrchestrationResult(
            error="max_iterations_reached",
            state=S,
            status=OrchestrationStatus.FAILED,
            metadata={"iteration": iteration, "max_iterations": max_iterations}
        )

# Convenience functions for common use cases

async def run_minimal_task(goal: str, 
                          budgets: Optional[Budgets] = None,
                          acceptance: Optional[AcceptanceCriteria] = None,
                          llm_client=None,
                          user_id: str = "default_user") -> OrchestrationResult:
    """
    Convenience function to run a task with minimal configuration.
    
    Args:
        goal: The goal to achieve
        budgets: Token and iteration budgets (uses defaults if None)
        acceptance: Acceptance criteria (uses defaults if None)
        llm_client: LLM client for evaluation
        user_id: User identifier
        
    Returns:
        OrchestrationResult
    """
    if budgets is None:
        budgets = Budgets()
    
    if acceptance is None:
        acceptance = AcceptanceCriteria(min_score=0.7)
    
    orchestrator = TaskOrchestrator(llm_client=llm_client)
    return await orchestrator.run_task(goal, budgets, acceptance, user_id)


async def run_task_with_evaluation(goal: str,
                                  min_score: float = 0.7,
                                  required_elements: list[str] = None,
                                  llm_client=None) -> OrchestrationResult:
    """
    Run a task with basic evaluation criteria.
    
    Args:
        goal: The goal to achieve
        min_score: Minimum score threshold
        required_elements: List of required elements in the answer
        llm_client: LLM client for evaluation
        
    Returns:
        OrchestrationResult
    """
    acceptance = AcceptanceCriteria(
        min_score=min_score,
        required_elements=required_elements or []
    )
    
    return await run_minimal_task(goal, acceptance=acceptance, llm_client=llm_client)
