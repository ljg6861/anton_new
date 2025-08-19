import datetime
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from server.agent.agentic_flow.helpers_and_prompts import ASSESSMENT_PROMPT, COMPLEX_PLANNER_PROMPT, EXECUTOR_PROMPT, SIMPLE_PLANNER_PROMPT, call_model_server
from server.agent.config import MODEL_SERVER_URL, USER_ROLE
from server.agent.knowledge_store import KnowledgeStore
from server.agent.react.react_agent import ReActAgent
from server.agent.react.token_budget import TokenBudget
from server.agent.tools.tool_manager import tool_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def _call_and_parse_model(chat_messages: List[Dict[str, str]], expected_key: str) -> Any:
    """
    Handles the common logic of calling the model, parsing the response,
    and handling potential errors.
    """
    raw_response = ""
    async for token in call_model_server(chat_messages):
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
    chat_prompt = ASSESSMENT_PROMPT.replace("{tools}", json.dumps(tools, indent=2))
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages
    
    assessment = await _call_and_parse_model(chat_messages, "assessment")
    return assessment or "" # Return the assessment or an empty string if parsing failed

async def execute_planner(prompt: str, messages: List[Dict[str, str]]) -> str:
    """Executes the planner to create a simple plan."""
    tools = tool_manager.get_tool_schemas()
    logger.info("Creating Simple Plan")
    chat_prompt = prompt.replace("{tools}", json.dumps(tools, indent=2)).replace(
        "{current_date}", datetime.datetime.now().isoformat()
    )
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages
    
    plan = await _call_and_parse_model(chat_messages, "plan")
    return plan or ""

async def execute_executor(overall_goal: str, full_plan: str, previous_steps: List[Dict[str, str]], current_step: Dict[str, str]) -> str:
    """Executes the executor to carry out the plan."""
    tools = tool_manager.get_tool_schemas()
    logger.info("Executing Plan")
    chat_prompt = EXECUTOR_PROMPT.replace("{tools}", json.dumps(tools, indent=2)).replace("{overall_goal}", overall_goal).replace("{full_plan}", json.dumps(full_plan)).replace("{previous_steps}", json.dumps(previous_steps)).replace("{current_step}", json.dumps(current_step))
    chat_messages = [{"role": "system", "content": chat_prompt}]

    knowledge_store = KnowledgeStore()
    
    available_tools = tool_manager.get_tool_schemas()
    
    budget = TokenBudget(total_budget=135000)

    react_agent = ReActAgent(
        api_base_url=MODEL_SERVER_URL,
        tools=available_tools,
        knowledge_store=knowledge_store,
        max_iterations=5,
        user_id='',
        token_budget=budget
    )

    result = await react_agent.execute_react_loop(chat_messages, logger)
    return result or ""

async def handle_task_route(messages: List[Dict[str, str]], goal:str) -> AsyncGenerator[str, None]:
    """Placeholder for the complex task agent workflow."""
    logger.info("Executing task route...")

    assessment = await initial_assessment(messages)
    logger.info(f"Initial assessment result: {assessment}")

    if assessment == "Sufficient":
        plan = await execute_planner(SIMPLE_PLANNER_PROMPT, messages)
    else:
        plan = await execute_planner(COMPLEX_PLANNER_PROMPT, messages)
    
    step_history = []
    for step in plan:
        if step['tool'] == 'final_answer':
            logger.info("Final step reached, returning result.")
            yield step['args']['summary']
            return
        result = await execute_executor(goal, plan, step_history, step)
        step_history.append({"step": step['step'], "result": result})
