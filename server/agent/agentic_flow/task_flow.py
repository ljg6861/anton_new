import datetime
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from server.agent.agentic_flow.helpers import SIMPLE_PLANNER_PROMPT, call_model_server
from server.agent.config import MODEL_SERVER_URL, USER_ROLE
from server.agent.knowledge_store import KnowledgeStore
from server.agent.react.react_agent import ReActAgent
from server.agent.react.token_budget import TokenBudget
from server.agent.tools.tool_manager import tool_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def initial_assessment(messages: List[Dict[str, str]]) -> str:
    tools = tool_manager.get_tool_schemas()
    logger.info("Assessing Task")
    chat_prompt = """
    You are a highly autonomous AI task assessment agent. Your only function is to analyze the user's request and determine if your system's core toolset is sufficient to handle it. You must output a single, valid JSON object with one of two possible values.

    # CORE TOOLS
    These are the primary tools for accomplishing tasks.
    {tools}

    # ASSESSMENT CRITERIA
    - If the user's request can be fully resolved using ONLY the CORE TOOLS listed above, the assessment is "Sufficient".
    - If the user's request CANNOT be fully resolved using the CORE TOOLS, the assessment is "Requires_Discovery".

    # OUTPUT FORMAT
    Output a single JSON object with a single key "assessment". Do not add any conversational text or explanations.

    --- EXAMPLES ---

    User Request: "What's the weather like in London and save it to a file called weather.txt?"
    {"assessment": "Sufficient"}

    User Request: "Write a python script to calculate the fibonacci sequence up to 10."
    {"assessment": "Sufficient"}

    User Request: "Book a flight for me to New York for next Tuesday."
    {"assessment": "Requires_Discovery"}

    User Request: "Check my Google Calendar for my next meeting."
    {"assessment": "Requires_Discovery"}

    User Request: "hello how are you"
    {"assessment": "Sufficient"}

    --- END EXAMPLES ---

    Note that the above examples are purely based on input and output. You MUST evaluate the tools available to determine your assessment.

    Now, assess the following messages.
    """.replace("{tools}", json.dumps(tools, indent=2))
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages
    raw_response = ""
    async for token in call_model_server(chat_messages):
            raw_response += token
    
    # Pre-process the response to handle <think> tags
    cleaned_response = raw_response
    if "</think>" in cleaned_response:
        cleaned_response = cleaned_response.split("</think>")[-1].strip()
        logger.info("Stripped <think> tags from model response.")

        try:
            parsed_json = json.loads(cleaned_response)
            if "assessment" in parsed_json:
                logger.info(f"Router successfully parsed JSON: {parsed_json}")
                return parsed_json["assessment"]
            else:
                logger.warning(f"JSON is valid but missing 'assessment' key: {parsed_json}")

        except json.JSONDecodeError:
            # "Bump" the agent with a reminder on the next attempt
            reminder = "Your previous response was not valid JSON. You MUST respond with only a JSON object."
            chat_messages.append({"role": "assistant", "content": raw_response})
            chat_messages.append({"role": "user", "content": reminder})


async def execute_planner(prompt: str, messages: List[Dict[str, str]]) -> str:
    tools = tool_manager.get_tool_schemas()
    logger.info("Creating Simple Plan")
    chat_prompt = prompt.replace("{tools}", json.dumps(tools, indent=2)).replace(
        "{current_date}", datetime.datetime.now().isoformat()
    )
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages
    raw_response = ""
    async for token in call_model_server(chat_messages):
            raw_response += token
    
    # Pre-process the response to handle <think> tags
    cleaned_response = raw_response
    if "</think>" in cleaned_response:
        cleaned_response = cleaned_response.split("</think>")[-1].strip()
        logger.info("Stripped <think> tags from model response.")

        try:
            parsed_json = json.loads(cleaned_response)
            if "plan" in parsed_json:
                logger.info(f"Router successfully parsed JSON: {parsed_json}")
                return parsed_json["plan"]
            else:
                logger.warning(f"JSON is valid but missing 'plan' key: {parsed_json}")

        except json.JSONDecodeError:
            # "Bump" the agent with a reminder on the next attempt
            reminder = "Your previous response was not valid JSON. You MUST respond with only a JSON object."
            chat_messages.append({"role": "assistant", "content": raw_response})
            chat_messages.append({"role": "user", "content": reminder})



async def handle_task_route(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Placeholder for the complex task agent workflow."""
    logger.info("Executing task route...")
    
    knowledge_store = KnowledgeStore()
    
    available_tools = tool_manager.get_tool_schemas()
    
    budget = TokenBudget(total_budget=135000)

    initial_assessment = await initial_assessment(messages)
    logger.info(f"Initial assessment result: {initial_assessment}")

    if initial_assessment == "Sufficient":
        plan = await execute_planner(SIMPLE_PLANNER_PROMPT, messages)

    # react_agent = ReActAgent(
    #     api_base_url=MODEL_SERVER_URL,
    #     tools=available_tools,
    #     knowledge_store=knowledge_store,
    #     max_iterations=30,
    #     user_id='',
    #     token_budget=budget
    # )
    