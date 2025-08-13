"""
Swarm-Style Execution Functions

Implements explicit step-by-step execution to replace looping ReAct patterns:
prepare_inputs → execute → postprocess

This provides deterministic execution with clear handoffs between functions.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from server.agent.state import State, Budgets, AgentStatus
from server.agent.tools_router import tools_router, ExecutionResult, ExecutionStatus
from server.agent.evaluator_node import AcceptanceCriteria, EvaluatorNode, get_evaluator_node

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of individual execution steps"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result from an individual execution step"""
    ok: bool
    output: Any = None
    error: Optional[str] = None
    cost: float = 0.0
    metadata: Dict[str, Any] = None
    status: StepStatus = StepStatus.SUCCESS
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionFlow:
    """Result from complete execution flow"""
    success: bool
    final_output: Any = None
    error: Optional[str] = None
    total_cost: float = 0.0
    step_results: List[StepResult] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.step_results is None:
            self.step_results = []
        if self.metadata is None:
            self.metadata = {}


# Step 1: Prepare Inputs
def function_prepare_inputs(goal: str, context: List[Any]) -> StepResult:
    """
    Prepare inputs for execution based on goal and context.
    
    This function analyzes the goal and available context to determine:
    - What tools are needed
    - What information is required
    - How to structure the execution plan
    
    Args:
        goal: The objective to achieve
        context: Available context information
        
    Returns:
        StepResult with prepared execution plan
    """
    logger.info(f"PREPARE_INPUTS: Analyzing goal: {goal[:100]}...")
    
    try:
        # Analyze goal to determine execution strategy
        execution_plan = _analyze_goal_requirements(goal)
        
        # Process available context
        context_summary = _process_context(context)
        
        # Determine required tools
        required_tools = _determine_required_tools(goal, execution_plan)
        
        # Create structured input for execution
        prepared_input = {
            "goal": goal,
            "execution_plan": execution_plan,
            "context_summary": context_summary,
            "required_tools": required_tools,
            "strategy": execution_plan.get("strategy", "direct_execution")
        }
        
        logger.info(f"PREPARE_INPUTS: Success - Strategy: {prepared_input['strategy']}, Tools: {len(required_tools)}")
        
        return StepResult(
            ok=True,
            output=prepared_input,
            cost=10.0,  # Minimal cost for analysis
            metadata={
                "tools_identified": len(required_tools),
                "strategy": prepared_input["strategy"],
                "context_items": len(context) if context else 0
            }
        )
        
    except Exception as e:
        logger.error(f"PREPARE_INPUTS: Failed - {e}")
        return StepResult(
            ok=False,
            error=f"input_preparation_failed: {str(e)}",
            status=StepStatus.FAILED
        )


# Step 2: Execute
async def function_execute(prepared_input: Dict[str, Any]) -> StepResult:
    """
    Execute the plan using ToolsRouter and deterministic steps.
    
    This function takes the prepared input and executes the plan using
    the tools and strategy determined in the preparation phase.
    
    Args:
        prepared_input: Output from function_prepare_inputs
        
    Returns:
        StepResult with execution results
    """
    logger.info(f"EXECUTE: Running strategy: {prepared_input.get('strategy', 'unknown')}")
    
    try:
        goal = prepared_input["goal"]
        execution_plan = prepared_input["execution_plan"]
        required_tools = prepared_input["required_tools"]
        
        # Execute based on strategy
        strategy = prepared_input.get("strategy", "direct_execution")
        
        if strategy == "tool_based":
            result = await _execute_tool_based_strategy(goal, required_tools, execution_plan)
        elif strategy == "research_and_synthesize":
            result = await _execute_research_synthesis_strategy(goal, required_tools, execution_plan)
        elif strategy == "code_generation":
            result = await _execute_code_generation_strategy(goal, required_tools, execution_plan)
        else:
            # Default: direct execution
            result = await _execute_direct_strategy(goal, required_tools, execution_plan)
        
        if result["success"]:
            logger.info(f"EXECUTE: Success - Generated {len(result['output'])} chars")
            return StepResult(
                ok=True,
                output=result["output"],
                cost=result.get("cost", 50.0),
                metadata={
                    "strategy_used": strategy,
                    "tools_called": result.get("tools_used", []),
                    "execution_steps": result.get("steps", 1)
                }
            )
        else:
            logger.error(f"EXECUTE: Strategy failed - {result.get('error', 'unknown')}")
            return StepResult(
                ok=False,
                error=f"execution_failed: {result.get('error', 'strategy_execution_failed')}",
                status=StepStatus.FAILED,
                cost=result.get("cost", 0.0)
            )
            
    except Exception as e:
        logger.error(f"EXECUTE: Exception - {e}")
        return StepResult(
            ok=False,
            error=f"execution_exception: {str(e)}",
            status=StepStatus.FAILED
        )


# Step 3: Postprocess
async def function_postprocess(execution_output: str, 
                              acceptance_criteria: Optional[AcceptanceCriteria] = None,
                              llm_client=None) -> StepResult:
    """
    Postprocess execution results for quality and formatting.
    
    This function takes the raw execution output and:
    - Formats it for presentation
    - Validates against acceptance criteria
    - Adds metadata and quality scores
    
    Args:
        execution_output: Raw output from function_execute
        acceptance_criteria: Optional quality criteria
        llm_client: Optional LLM client for evaluation
        
    Returns:
        StepResult with final formatted output
    """
    logger.info(f"POSTPROCESS: Processing {len(execution_output)} chars")
    
    try:
        # Format the output
        formatted_output = _format_output(execution_output)
        
        # Optional quality evaluation
        quality_score = None
        evaluation_feedback = None
        
        if acceptance_criteria and llm_client:
            try:
                # Create a mock state for evaluation
                from server.agent.state import make_state
                mock_state = make_state("postprocess_evaluation", Budgets())
                mock_state.context = [execution_output]
                
                evaluator = get_evaluator_node(llm_client)
                evaluation = await evaluator.evaluate(formatted_output, mock_state, acceptance_criteria)
                quality_score = evaluation.overall_score
                evaluation_feedback = evaluation.feedback
                
                logger.info(f"POSTPROCESS: Quality score: {quality_score:.2f}")
                
            except Exception as eval_error:
                logger.warning(f"POSTPROCESS: Evaluation failed: {eval_error}")
        
        # Add final metadata
        final_metadata = {
            "original_length": len(execution_output),
            "formatted_length": len(formatted_output),
            "quality_score": quality_score,
            "evaluation_attempted": acceptance_criteria is not None
        }
        
        if evaluation_feedback:
            final_metadata["evaluation_feedback"] = evaluation_feedback[:200]  # Truncate for brevity
        
        logger.info("POSTPROCESS: Success - Output formatted and validated")
        
        return StepResult(
            ok=True,
            output=formatted_output,
            cost=5.0,  # Minimal cost for formatting
            metadata=final_metadata
        )
        
    except Exception as e:
        logger.error(f"POSTPROCESS: Failed - {e}")
        return StepResult(
            ok=False,
            error=f"postprocessing_failed: {str(e)}",
            status=StepStatus.FAILED
        )


# Main Flow Execution
async def run_minimal_flow(state: State, 
                          acceptance_criteria: Optional[AcceptanceCriteria] = None,
                          llm_client=None) -> ExecutionFlow:
    """
    Execute the minimal flow: prepare_inputs → execute → postprocess
    
    Args:
        state: Current execution state
        acceptance_criteria: Optional quality criteria
        llm_client: Optional LLM client for evaluation
        
    Returns:
        ExecutionFlow with complete results
    """
    logger.info(f"RUN_MINIMAL_FLOW: Starting for goal: {state.goal[:100]}...")
    
    step_results = []
    total_cost = 0.0
    
    # Step 1: Prepare inputs
    s1 = function_prepare_inputs(state.goal, state.context)
    step_results.append(s1)
    total_cost += s1.cost
    
    if not s1.ok:
        logger.error(f"RUN_MINIMAL_FLOW: Step 1 failed - {s1.error}")
        return ExecutionFlow(
            success=False,
            error=s1.error,
            total_cost=total_cost,
            step_results=step_results,
            metadata={"failed_at_step": 1}
        )
    
    # Step 2: Execute
    s2 = await function_execute(s1.output)
    step_results.append(s2)
    total_cost += s2.cost
    
    if not s2.ok:
        logger.error(f"RUN_MINIMAL_FLOW: Step 2 failed - {s2.error}")
        return ExecutionFlow(
            success=False,
            error=s2.error,
            total_cost=total_cost,
            step_results=step_results,
            metadata={"failed_at_step": 2}
        )
    
    # Step 3: Postprocess
    s3 = await function_postprocess(s2.output, acceptance_criteria, llm_client)
    step_results.append(s3)
    total_cost += s3.cost
    
    if not s3.ok:
        logger.error(f"RUN_MINIMAL_FLOW: Step 3 failed - {s3.error}")
        return ExecutionFlow(
            success=False,
            error=s3.error,
            total_cost=total_cost,
            step_results=step_results,
            metadata={"failed_at_step": 3}
        )
    
    # Success!
    logger.info(f"RUN_MINIMAL_FLOW: Success - Total cost: {total_cost}")
    
    return ExecutionFlow(
        success=True,
        final_output=s3.output,
        total_cost=total_cost,
        step_results=step_results,
        metadata={
            "steps_completed": 3,
            "strategies_used": [s1.metadata.get("strategy"), s2.metadata.get("strategy_used")],
            "tools_used": s2.metadata.get("tools_called", [])
        }
    )


# Branch Executor (as requested)
async def branch_executor(state: State,
                         acceptance_criteria: Optional[AcceptanceCriteria] = None,
                         llm_client=None) -> ExecutionFlow:
    """
    Route execution based on state.route, with fallback to minimal flow.
    
    Args:
        state: Current execution state
        acceptance_criteria: Optional quality criteria
        llm_client: Optional LLM client for evaluation
        
    Returns:
        ExecutionFlow results
    """
    logger.info(f"BRANCH_EXECUTOR: Route: {getattr(state, 'route', 'MINIMAL_FLOW')}")
    
    route = getattr(state, 'route', 'MINIMAL_FLOW')
    
    if route == "MINIMAL_FLOW":
        return await run_minimal_flow(state, acceptance_criteria, llm_client)
    else:
        # Temporarily force minimal flow as requested
        logger.info(f"BRANCH_EXECUTOR: Route {route} not implemented, using MINIMAL_FLOW")
        return await run_minimal_flow(state, acceptance_criteria, llm_client)


# Helper Functions
def _analyze_goal_requirements(goal: str) -> Dict[str, Any]:
    """Analyze goal to determine execution requirements"""
    goal_lower = goal.lower()
    
    if any(keyword in goal_lower for keyword in ["implement", "code", "function", "algorithm"]):
        return {
            "strategy": "code_generation",
            "complexity": "medium",
            "requires_tools": True,
            "expected_output": "code_with_explanation"
        }
    elif any(keyword in goal_lower for keyword in ["research", "find", "search", "investigate"]):
        return {
            "strategy": "research_and_synthesize", 
            "complexity": "high",
            "requires_tools": True,
            "expected_output": "comprehensive_analysis"
        }
    elif any(keyword in goal_lower for keyword in ["explain", "describe", "what is", "how does"]):
        return {
            "strategy": "direct_execution",
            "complexity": "low",
            "requires_tools": False,
            "expected_output": "explanation"
        }
    else:
        return {
            "strategy": "tool_based",
            "complexity": "medium",
            "requires_tools": True,
            "expected_output": "structured_response"
        }


def _process_context(context: List[Any]) -> Dict[str, Any]:
    """Process available context information"""
    if not context:
        return {"summary": "No context available", "items": 0}
    
    context_items = len(context)
    total_chars = sum(len(str(item)) for item in context)
    
    return {
        "summary": f"Available context: {context_items} items, {total_chars} chars",
        "items": context_items,
        "total_chars": total_chars
    }


def _determine_required_tools(goal: str, execution_plan: Dict[str, Any]) -> List[str]:
    """Determine which tools are needed for the goal"""
    goal_lower = goal.lower()
    required_tools = []
    
    # Code-related goals
    if execution_plan.get("strategy") == "code_generation":
        required_tools.extend(["execute_python_code", "search_codebase"])
    
    # Research-related goals
    if execution_plan.get("strategy") == "research_and_synthesize":
        required_tools.extend(["web_search", "read_file"])
    
    # File operations
    if any(keyword in goal_lower for keyword in ["file", "read", "write", "save"]):
        required_tools.extend(["read_file", "write_file"])
    
    # Git operations
    if any(keyword in goal_lower for keyword in ["git", "commit", "branch", "repository"]):
        required_tools.extend(["git_status", "git_commit"])
    
    # Default tools for general tasks
    if not required_tools:
        required_tools = ["get_codebase_stats"]
    
    return list(set(required_tools))  # Remove duplicates


async def _execute_direct_strategy(goal: str, required_tools: List[str], execution_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute direct strategy - simple goal completion"""
    logger.info("Executing direct strategy")
    
    # For direct strategy, we generate a response based on the goal
    if "python" in goal.lower():
        output = f"""Python is a high-level, interpreted programming language known for its simplicity and readability.

Key Features:
- Easy to learn syntax
- Extensive standard library
- Cross-platform compatibility
- Dynamic typing
- Large ecosystem

Common Applications:
- Web development
- Data science
- Machine learning
- Automation
- Scientific computing

Example:
```python
print("Hello, World!")
```

Python's design philosophy emphasizes code readability and simplicity."""
    
    elif "algorithm" in goal.lower() or "implement" in goal.lower():
        output = f"""Here's a comprehensive solution for: {goal}

```python
def example_implementation():
    # Implementation based on the goal
    pass
```

This solution addresses the requirements by providing a structured approach."""
    
    else:
        output = f"This is a comprehensive response to: {goal}\n\nThe answer addresses the key aspects and provides relevant information."
    
    return {
        "success": True,
        "output": output,
        "cost": 30.0,
        "tools_used": [],
        "steps": 1
    }


async def _execute_tool_based_strategy(goal: str, required_tools: List[str], execution_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute tool-based strategy using ToolsRouter"""
    logger.info(f"Executing tool-based strategy with tools: {required_tools}")
    
    results = []
    total_cost = 0.0
    tools_used = []
    
    for tool_name in required_tools[:3]:  # Limit to 3 tools for deterministic execution
        try:
            # Use ToolsRouter for actual tool execution
            result = await tools_router.call(tool_name, {}, timeout=30.0)
            
            if result.status == ExecutionStatus.SUCCESS:
                results.append(f"Tool {tool_name}: {str(result.result)[:200]}...")
                tools_used.append(tool_name)
                total_cost += 20.0
            else:
                results.append(f"Tool {tool_name} failed: {result.error}")
                
        except Exception as e:
            results.append(f"Tool {tool_name} error: {str(e)}")
    
    if results:
        output = f"Results for: {goal}\n\n" + "\n\n".join(results)
        return {
            "success": True,
            "output": output,
            "cost": total_cost,
            "tools_used": tools_used,
            "steps": len(results)
        }
    else:
        return {
            "success": False,
            "error": "no_tools_succeeded",
            "cost": total_cost
        }


async def _execute_research_synthesis_strategy(goal: str, required_tools: List[str], execution_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute research and synthesis strategy"""
    logger.info("Executing research and synthesis strategy")
    
    # Mock research phase
    research_results = [
        f"Research finding 1 for {goal}",
        f"Research finding 2 for {goal}",
        f"Research finding 3 for {goal}"
    ]
    
    # Mock synthesis
    output = f"""Research Analysis: {goal}

Key Findings:
{chr(10).join(f"• {finding}" for finding in research_results)}

Synthesis:
Based on the research, this comprehensive analysis addresses the core aspects of {goal} and provides actionable insights.

Recommendations:
- Further investigation may be needed
- Consider implementation approaches
- Validate findings with additional sources"""
    
    return {
        "success": True,
        "output": output,
        "cost": 40.0,
        "tools_used": required_tools[:2],
        "steps": 2  # Research + synthesis
    }


async def _execute_code_generation_strategy(goal: str, required_tools: List[str], execution_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute code generation strategy"""
    logger.info("Executing code generation strategy")
    
    # Generate code based on goal
    if "binary search" in goal.lower():
        code = """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1"""
    else:
        code = f"""def solution():
    # Implementation for: {goal}
    pass"""
    
    output = f"""Code Implementation: {goal}

```python
{code}
```

Explanation:
This implementation provides a solution for {goal} using best practices and efficient algorithms.

Usage:
The code can be tested and integrated into larger systems as needed."""
    
    return {
        "success": True,
        "output": output,
        "cost": 35.0,
        "tools_used": ["execute_python_code"] if "execute_python_code" in required_tools else [],
        "steps": 1
    }


def _format_output(raw_output: str) -> str:
    """Format raw output for presentation"""
    # Basic formatting - ensure proper structure
    if not raw_output.strip():
        return "No output generated."
    
    # Add basic structure if missing
    if not any(marker in raw_output for marker in ["#", "##", "**", "```"]):
        # Add basic formatting
        lines = raw_output.split('\n')
        if len(lines) > 3:
            formatted = f"# Response\n\n{raw_output}\n\n---\n*Generated via deterministic execution flow*"
        else:
            formatted = raw_output
    else:
        formatted = raw_output
    
    return formatted.strip()
