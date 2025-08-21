import datetime
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from server.agent.agentic_flow.helpers_and_prompts import ASSESSMENT_PROMPT, EXECUTOR_PROMPT, PLAN_RESULT_EVALUATOR_PROMPT, REPLANNING_PROMPT_ADDENDUM, RESEARCHER_PROMPT, SIMPLE_PLANNER_PROMPT, call_model_server
from server.agent.config import MODEL_SERVER_URL, USER_ROLE
from server.agent.knowledge_store import KnowledgeStore
from server.agent.react.react_agent import ReActAgent
from server.agent.react.token_budget import TokenBudget
from server.agent.tools.tool_manager import tool_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    chat_prompt = ASSESSMENT_PROMPT
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages

    assessment = await _call_and_parse_model(chat_messages, tools, "assessment")
    return assessment or "" # Return the assessment or an empty string if parsing failed

async def execute_researcher(messages: List[Dict[str, str]]) -> Dict:
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
    await react_agent.execute_react_loop(research_messages, logger)
    try:
        research_result = json.loads(research_messages[-1]['content'])
        return research_result
    except Exception as e:
        logger.error('Failed decoding research: %s', e)
        research_result = research_messages[-1]['content']
        # Fallback: try to get the raw response and extract findings
        logger.info("Attempting to extract research findings from raw LLM response...")
        
        import re
        # Clean control characters that break JSON parsing
        cleaned_result = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', research_result)
        
        # Try parsing the cleaned result first
        try:
            research_data = json.loads(cleaned_result)
            return research_data
        except:
            # If that fails, try to extract just the research_findings section
            json_match = re.search(r'\{.*?"research_findings".*?\}', cleaned_result, re.DOTALL)
            if json_match:
                try:
                    research_data = json.loads(json_match.group())
                    return research_data.get("research_findings", {"summary": research_result})
                except:
                    pass
            
            # Final fallback: return the raw result as summary
            return {"summary": research_result}

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

async def execute_executor(overall_goal: str, full_plan: str, previous_steps: List[Dict[str, str]], current_step: Dict[str, str]) -> str:
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
        max_iterations=1,
        user_id='',
        token_budget=budget
    )

    result = await react_agent.execute_react_loop(chat_messages, logger)
    return result or ""

async def evaluate_plan_success(overall_goal: str, full_plan: str, previous_steps: List[Dict[str, str]]) -> Dict:
    logger.info("Evaluating plan success...")
    chat_prompt = PLAN_RESULT_EVALUATOR_PROMPT.replace("{user_goal}", overall_goal).replace("{original_plan}", json.dumps(full_plan)).replace("{execution_results}", json.dumps(previous_steps))
    result = await _call_and_parse_model([{'role' : 'system', 'content' : chat_prompt}])
    logger.info(f"Plan evaluation result: {result}")
    return result

async def handle_task_route(messages: List[Dict[str, str]], goal:str) -> AsyncGenerator[str, None]:
    """Complex task agent workflow with research phase."""
    logger.info("Executing task route...")

    # Step 2: RELENTLESS RESEARCH PHASE
    logger.info("Starting RELENTLESS research phase...")
    research_findings = await execute_researcher(messages)
    logger.info(f"Research completed with findings: {research_findings}")

    # Step 3: Planning with Research Context
    prompt = SIMPLE_PLANNER_PROMPT
    plan = await execute_planner(prompt, messages, research_findings)
    
    # Step 4: Execute the plan
    async for result in execute_control_loop(plan, goal, messages, research_findings):
        yield result

async def execute_control_loop(plan, goal, messages, research_findings=None):
    step_history = []
    for step in plan:
        if step['tool'] == 'final_answer':
            logger.info("Final step reached")
            result = await evaluate_plan_success(goal, plan, step_history)
            if result['result'] == 'success':
                yield step['args']['summary']
                return
            else:
                logger.info("Replanning due to failure... " + result['reasoning'])
                prompt_string = SIMPLE_PLANNER_PROMPT + REPLANNING_PROMPT_ADDENDUM

                prompt_string = prompt_string.replace("{user_goal}", goal)
                prompt_string = prompt_string.replace("{original_plan}", json.dumps(plan))
                prompt_string = prompt_string.replace("{failure_reasoning}", result['reasoning'])
                prompt_string = prompt_string.replace("{steps_taken}", str(json.dumps(step_history)))

                plan = await execute_planner(prompt_string, messages, research_findings)
                yield execute_control_loop(plan, goal, messages, research_findings)
        result = await execute_executor(goal, plan, step_history, step)
        step_history.append({"step": step['step'], "result": result})
