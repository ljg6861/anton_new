import json
import logging
from typing import AsyncGenerator, Dict, List, Optional
from server.agent.agentic_flow.helpers_and_prompts import call_model_server
from server.agent.config import USER_ROLE
from server.agent.agentic_flow.task_flow import handle_task_route

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit" 


FEW_SHOT_ROUTER_PROMPT = """You are a precise and logical AI router. Your only function is to classify the user's intent based on their most recent message and output a single, valid JSON object. Do not add any conversational text or explanations.

# ROUTES
- `chat`: For general conversation, greetings, and simple questions that do not require action.
- `task`: For any request that requires an action, tool use, or a multi-step process. This includes simple commands like setting reminders and complex requests like writing code.

Based on the route descriptions, classify the user's message by following the pattern in the examples below.

--- EXAMPLES ---

User Message: "hello there how are you"
{"route": "chat"}

User Message: "What is the largest planet in our solar system?"
{"route": "chat"}

User Message: "Set a reminder to call Mom at 5pm."
{"route": "task"}

User Message: "Schedule a meeting with the design team for tomorrow morning."
{"route": "task"}

User Message: "Create a to-do list for my grocery shopping trip."
{"route": "task"}

User Message: "Write a python script to parse a CSV file."
{"route": "task"}

--- END EXAMPLES ---

Now, classify the following user message.
"""

def extract_user_prompt(messages: List[Dict[str, str]]) -> str:
    """Extract the latest user prompt from messages"""
    for msg in reversed(messages):
        if msg.get("role") == USER_ROLE:
            return msg.get("content", "")
    return ""

async def determine_route(conversation_history: List[Dict]) -> Optional[str]:
    MAX_ROUTER_RETRIES = 2
    router_messages = [
        {"role": "system", "content": FEW_SHOT_ROUTER_PROMPT},
    ] + conversation_history

    for attempt in range(MAX_ROUTER_RETRIES):
        logger.info(f"Router attempt {attempt + 1}/{MAX_ROUTER_RETRIES}...")
        
        raw_response = ""
        async for token in call_model_server(router_messages):
            raw_response += token
        
        # Pre-process the response to handle <think> tags
        cleaned_response = raw_response
        if "</think>" in cleaned_response:
            cleaned_response = cleaned_response.split("</think>")[-1].strip()
            logger.info("Stripped <think> tags from model response.")

        try:
            parsed_json = json.loads(cleaned_response)
            if "route" in parsed_json:
                logger.info(f"Router successfully parsed JSON: {parsed_json}")
                return parsed_json["route"]
            else:
                logger.warning(f"JSON is valid but missing 'route' key: {parsed_json}")

        except json.JSONDecodeError:
            logger.warning(f"Attempt {attempt + 1} failed: Could not decode JSON from response: '{raw_response}'")
            # "Bump" the agent with a reminder on the next attempt
            reminder = "Your previous response was not valid JSON. You MUST respond with only a JSON object."
            router_messages.append({"role": "assistant", "content": raw_response})
            router_messages.append({"role": "user", "content": reminder})
    
    logger.error("Router failed to produce a valid route after multiple attempts.")
    return None


async def execute_agentic_flow(initial_messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Execute the main agentic flow"""
    logger.info("--- Starting Agentic Flow ---")

    user_prompt = extract_user_prompt(initial_messages)
    
    if user_prompt:
        # learning_loop.start_task(user_prompt)
        pass

    # --- Route to the appropriate handler ---
    route = await determine_route(initial_messages)
    logger.info(f"Routing to: {route}")

    if route == "chat":
        async for token in _handle_chat_route(initial_messages):
            yield token
    elif route == "task":
        async for token in handle_task_route(initial_messages, user_prompt):
            yield token
    else:
        yield f"Error: Unknown route '{route}' provided by the router."

async def _handle_chat_route(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    chat_prompt = "You are Anton, a friendly and helpful assistant. Provide a concise answer to the user."
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages
    
    buffer = ''
    async for token in call_model_server(chat_messages):
        if not '</think>' in buffer:
            yield f'<thought>{token}</thought>'
        elif '</think>' in buffer:
            yield token
        #Add buffer at end to avoid </think> annotation
        buffer += token