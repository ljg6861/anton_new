import asyncio
import datetime
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from server.agent.tool_learning_store import tool_learning_store
from server.agent.agentic_flow.helpers_and_prompts import ADAPTIVE_PLANNING_PROMPT, ASSESSMENT_PROMPT, EXECUTOR_PROMPT, FINAL_MESSAGE_GENERATOR_PROMPT, PLAN_RESULT_EVALUATOR_PROMPT, REPLANNING_PROMPT_ADDENDUM, RESEARCHER_PROMPT, SIMPLE_PLANNER_PROMPT, call_model_server
from server.agent.config import MODEL_SERVER_URL, USER_ROLE
from server.agent.knowledge_store import KnowledgeStore
from server.agent.react.react_agent import ReActAgent
from server.agent.react.token_budget import TokenBudget
from server.agent.tools.tool_manager import tool_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def stream_step_content(text: str) -> AsyncGenerator[str, None]:
    """Stream text character by character with a typing effect for step content."""
    for char in text:
        yield f"<step_content>{char}</step_content>"
        await asyncio.sleep(0.02)  # Small delay for typing effect

async def _call_and_parse_model(chat_messages: List[Dict[str, str]], tools: List[Dict[str, Any]], expected_key: Optional[str] = None) -> Any:
    """
    Handles the common logic of calling the model, parsing the response,
    and handling potential errors.
    """
    raw_response = ""
    async for token in call_model_server(chat_messages, tools):
        raw_response += token

    cleaned_response = raw_response
    if "</think>" in cleaned_response:
        cleaned_response = cleaned_response.split("</think>")[-1].strip()
        logger.info("Stripped <think> tags from model response.")

    try:
        parsed_json = json.loads(cleaned_response)
        if expected_key in parsed_json:
            logger.info(f"Router successfully parsed JSON: {parsed_json}")
            return parsed_json[expected_key]
        elif not expected_key and parsed_json:
            return parsed_json
        else:
            logger.warning(f"JSON is valid but missing '{expected_key}' key: {parsed_json}")
            return None # Or raise an error
    except json.JSONDecodeError:
        # "Bump" the agent with a reminder on the next attempt
        reminder = "Your previous response was not valid JSON. You MUST respond with only a JSON object."
        chat_messages.append({"role": "assistant", "content": raw_response})
        chat_messages.append({"role": "user", "content": reminder})
        # This function could return None or raise an exception to handle the error state.
        return None 

async def initial_assessment(messages: List[Dict[str, str]]) -> str:
    """Assesses a task based on the provided messages."""
    tools = tool_manager.get_tool_schemas()
    logger.info("Assessing Task")
    chat_prompt = ASSESSMENT_PROMPT.replace('{tools}', json.dumps(tools))
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages

    assessment = await _call_and_parse_model(chat_messages, [], "assessment")
    return assessment or ""

async def execute_researcher(messages: List[Dict[str, str]]) -> AsyncGenerator[Dict, None]:
    """Executes the relentless researcher - an LLM that analyzes scraped data comprehensively."""
    logger.info("Executing RELENTLESS researcher...")
    
    # Step 1: Collect initial data based on request type
    user_content = " ".join([msg.get("content", "") for msg in messages if msg.get("role") == "user"])
    
    # Get available research tools for the prompt
    meta_tools = tool_manager.get_tools_by_names([
        "search_codebase",
        "web_search",
        "read_file"
    ])
    
    # Create researcher prompt with collected data
    research_prompt = RESEARCHER_PROMPT.replace(
        "{current_date}", datetime.datetime.now().isoformat()
    )
    
    research_messages = [
        {"role": "system", "content": research_prompt},
        {"role": "user", "content": 'Perform research related to the following user query: ' + user_content}
    ]
    
    # Step 3: Call LLM researcher to analyze the data
    logger.info("Calling LLM researcher to analyze collected data...")
    knowledge_store = KnowledgeStore()
        
    budget = TokenBudget(total_budget=135000)

    react_agent = ReActAgent(
        api_base_url=MODEL_SERVER_URL,
        tools=meta_tools,
        knowledge_store=knowledge_store,
        max_iterations=75,
        user_id='',
        token_budget=budget
    )
    
    research_result = ""
    async for chunk in react_agent.execute_react_loop_streaming(research_messages, logger):
        if chunk.startswith("<step>") or chunk.startswith("<step_content>"):
            yield chunk
        else:
            research_result += chunk
    
    try:
        research_data = json.loads(research_result)
        yield research_data
    except Exception as e:
        logger.error('Failed decoding research: %s', e)
        # Fallback: try to get the raw response and extract findings
        logger.info("Attempting to extract research findings from raw LLM response...")
        
        import re
        # Clean control characters that break JSON parsing
        cleaned_result = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', research_result)
        
        # Try parsing the cleaned result first
        try:
            research_data = json.loads(cleaned_result)
            yield research_data
        except:
            # If that fails, try to extract just the research_findings section
            json_match = re.search(r'\{.*?"research_findings".*?\}', cleaned_result, re.DOTALL)
            if json_match:
                try:
                    research_data = json.loads(json_match.group())
                    yield research_data.get("research_findings", {"summary": research_result})
                except:
                    pass
            
            # Final fallback: return the raw result as summary
            yield {"summary": research_result}

async def execute_planner(prompt: str, messages: List[Dict[str, str]], research_findings: Dict = None) -> str:
    """Executes the planner to create a simple plan."""
    tools = tool_manager.get_tool_schemas()
    logger.info("Creating Plan")
    
    # Include research findings in the prompt
    research_context = json.dumps(research_findings, indent=2) if research_findings else "No prior research conducted."
    
    chat_prompt = prompt.replace(
        "{current_date}", datetime.datetime.now().isoformat()
    ).replace("{research_findings}", research_context).replace("{tools}", json.dumps(tools, indent=2))
    
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages
    
    plan = await _call_and_parse_model(chat_messages, [], "plan")
    logger.info(f"Generated Plan: {plan}")
    return plan or ""

async def execute_executor(overall_goal: str, full_plan: str, previous_steps: List[Dict[str, str]], current_step: Dict[str, str]) -> AsyncGenerator[str, None]:
    """Executes the executor to carry out the plan."""
    logger.info("Executing Plan")
    chat_prompt = EXECUTOR_PROMPT.replace("{overall_goal}", overall_goal).replace("{full_plan}", json.dumps(full_plan)).replace("{previous_steps}", json.dumps(previous_steps)).replace("{current_step}", json.dumps(current_step))
    chat_messages = [{"role": "system", "content": chat_prompt}]

    knowledge_store = KnowledgeStore()
    
    available_tools = tool_manager.get_tool_schemas()
    
    budget = TokenBudget(total_budget=135000)

    react_agent = ReActAgent(
        api_base_url=MODEL_SERVER_URL,
        tools=available_tools,
        knowledge_store=knowledge_store,
        max_iterations=50,
        user_id='',
        token_budget=budget
    )

    result = ""
    async for chunk in react_agent.execute_react_loop_streaming(chat_messages, logger):
        if chunk.startswith("<step>") or chunk.startswith("<step_content>"):
            yield chunk
        else:
            result += chunk
    
    yield result or ""

async def execute_adaptive_planner(goal: str, original_plan: List[Dict], completed_steps: List[Dict], latest_result: str, messages: List[Dict[str, str]], research_findings: Dict = None) -> Dict:
    """Executes adaptive planning to determine if plan adjustments are needed."""
    tools = tool_manager.get_tool_schemas()
    logger.info("Executing adaptive planning...")
    
    # Include research findings in the prompt
    research_context = json.dumps(research_findings, indent=2) if research_findings else "No prior research conducted."
    
    chat_prompt = ADAPTIVE_PLANNING_PROMPT.replace(
        "{current_date}", datetime.datetime.now().isoformat()
    ).replace(
        "{user_goal}", goal
    ).replace(
        "{original_plan}", json.dumps(original_plan, indent=2)
    ).replace(
        "{completed_steps}", json.dumps(completed_steps, indent=2)
    ).replace(
        "{latest_result}", latest_result
    ).replace(
        "{research_findings}", research_context
    ).replace(
        "{tools}", json.dumps(tools, indent=2)
    )
    
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages
    
    result = await _call_and_parse_model(chat_messages, [], None)
    logger.info(f"Adaptive planning result: {result}")
    return result or {"action": "continue", "reasoning": "Failed to parse planning response", "plan": original_plan}

async def generate_final_user_message(goal: str, step_history: List[Dict], plan_summary: str) -> AsyncGenerator[str, None]:
    """Generate a natural user-facing message based on the completed execution."""
    logger.info("Generating final user-facing message...")
    
    # First stream the step content to show progress
    progress_msg = "Crafting personalized response..."
    async for content in stream_step_content(progress_msg):
        yield content
    
    chat_prompt = FINAL_MESSAGE_GENERATOR_PROMPT.replace(
        "{user_goal}", goal
    ).replace(
        "{execution_history}", json.dumps(step_history, indent=2)
    ).replace(
        "{plan_summary}", plan_summary
    )
    
    chat_messages = [{"role": "system", "content": chat_prompt}]
    
    # Now stream the actual response tokens
    async for token in call_model_server(chat_messages, []):
        yield token

async def evaluate_plan_success(overall_goal: str, full_plan: str, previous_steps: List[Dict[str, str]]) -> Dict:
    logger.info("Evaluating plan success...")
    chat_prompt = PLAN_RESULT_EVALUATOR_PROMPT.replace("{user_goal}", overall_goal).replace("{original_plan}", json.dumps(full_plan)).replace("{execution_results}", json.dumps(previous_steps))
    result = await _call_and_parse_model([{'role' : 'system', 'content' : chat_prompt}], [])
    logger.info(f"Plan evaluation result: {result}")
    return result

async def handle_task_route(messages: List[Dict[str, str]], goal:str) -> AsyncGenerator[str, None]:
    """Complex task agent workflow with research phase."""
    logger.info("Executing task route...")

    # Step 1: Assessment
    yield "<step>Assessing Task Requirements</step>"
    assessment = await initial_assessment(messages)
    async for content in stream_step_content(f"Assessment result: {assessment}"):
        yield content
    
    research_findings = []
    if assessment != 'Sufficient':
        # Step 2: RELENTLESS RESEARCH PHASE
        yield "<step>Conducting Research</step>"
        logger.info("Starting RELENTLESS research phase...")
        
        async for chunk in execute_researcher(messages):
            if isinstance(chunk, dict):
                # This is the final research result
                research_findings = chunk
            elif chunk.startswith("<step>") or chunk.startswith("<step_content>"):
                yield chunk
        
        logger.info(f"Research completed with findings: {research_findings}")
        async for content in stream_step_content("Research completed with findings"):
            yield content
    else:
        research_findings = []

    # Step 3: Planning with Research Context
    yield "<step>Generating Plan</step>"
    prompt = SIMPLE_PLANNER_PROMPT
    plan = await execute_planner(prompt, messages, research_findings)
    plan_msg = f"Plan generated with {len(plan) if plan else 0} steps"
    async for content in stream_step_content(plan_msg):
        yield content
    
    # Step 4: Execute the plan
    async for result in execute_control_loop(plan, goal, messages, research_findings):
        yield result

async def execute_control_loop(plan, goal, messages, research_findings=None):
    step_history = []
    current_plan = plan
    step_index = 0
    
    def truncate_step_name(name: str, max_length: int = 30) -> str:
        """Truncate step name to max_length characters"""
        if len(name) <= max_length:
            return name
        return name[:max_length-3] + "..."
    
    while step_index < len(current_plan):
        step = current_plan[step_index]
        
        # Execute the current step
        step_name = f"Executing Step {step['step']}: {step.get('thought', 'Processing step')}"
        truncated_name = truncate_step_name(step_name)
        yield f"<step>{truncated_name}</step>"

        result = ""
        async for chunk in execute_executor(goal, current_plan, step_history, step):
            if chunk.startswith("<step>") or chunk.startswith("<step_content>"):
                yield chunk
            else:
                result = chunk  # The final result comes as the last chunk
        
        step_history.append({"step": step['step'], "result": result})
        
        completion_msg = f"Step {step['step']} completed"
        async for content in stream_step_content(completion_msg):
            yield content
        step_index += 1
        
        # After each step (except the last), check if we need to adjust the plan
        if step_index < len(current_plan):
            yield f"<step>Evaluating Plan Progress</step>"
            logger.info(f"Checking if plan adjustment needed after step {step['step']}")
            adaptive_result = await execute_adaptive_planner(
                goal, current_plan, step_history, result, messages, research_findings
            )
            
            if adaptive_result.get("action") == "replan":
                logger.info(f"Replanning: {adaptive_result.get('reasoning')}")
                replan_msg = f"Replanning required: {adaptive_result.get('reasoning')}"
                async for content in stream_step_content(replan_msg):
                    yield content
                current_plan = adaptive_result.get("plan", current_plan)
                step_index = 0  # Start from beginning with new plan
            elif adaptive_result.get("action") == "adjust":
                logger.info(f"Adjusting plan: {adaptive_result.get('reasoning')}")
                adjust_msg = f"Adjusting plan: {adaptive_result.get('reasoning')}"
                async for content in stream_step_content(adjust_msg):
                    yield content
                # Replace remaining steps with adjusted ones
                adjusted_plan = adaptive_result.get("plan", [])
                current_plan = adjusted_plan
                step_index = 0  # Start from beginning with adjusted plan
            else:
                logger.info(f"Continuing with original plan: {adaptive_result.get('reasoning')}")
                continue_msg = f"Continuing with plan: {adaptive_result.get('reasoning')}"
                async for content in stream_step_content(continue_msg):
                    yield content
                # Continue with current plan as-is
    
    yield f"<step>Evaluating Final Results</step>"
    result = await evaluate_plan_success(goal, current_plan, step_history)
    if result['result'] == 'success':
        success_msg = "Task completed successfully"
        async for content in stream_step_content(success_msg):
            yield content
        # Generate a natural user-facing message instead of yielding the raw summary
        yield f"<step>Generating Final Response</step>"
        async for token in generate_final_user_message(
            goal, 
            step_history, 
            step.get('args', {}).get('summary', 'Task completed successfully')
        ):
            yield token
        return
    else:
        logger.info("Replanning due to failure... " + result['reasoning'])
        failure_msg = f"Task failed, replanning: {result['reasoning']}"
        async for content in stream_step_content(failure_msg):
            yield content
        yield f"<step>Replanning Strategy</step>"
        prompt_string = SIMPLE_PLANNER_PROMPT + REPLANNING_PROMPT_ADDENDUM

        prompt_string = prompt_string.replace("{user_goal}", goal)
        prompt_string = prompt_string.replace("{original_plan}", json.dumps(current_plan))
        prompt_string = prompt_string.replace("{failure_reasoning}", result['reasoning'])
        prompt_string = prompt_string.replace("{steps_taken}", str(json.dumps(step_history)))

        new_plan = await execute_planner(prompt_string, messages, research_findings)
        new_plan_msg = "New plan created"
        async for content in stream_step_content(new_plan_msg):
            yield content
        async for token in execute_control_loop(new_plan, goal, messages, research_findings):
            yield token 