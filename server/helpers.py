# FILE: server/helpers.py
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)


async def evaluate_turn_for_memorability(
        user_prompt: str,
        assistant_response: str,
        model,
        tokenizer
) -> Dict:
    """
    Uses an LLM to evaluate if a conversation turn is worth remembering
    and extracts the core learning if it is.
    """
    system_prompt = """You are a meta-analysis model. Your task is to analyze a conversation turn between a user and an AI assistant. Your goal is to decide if the interaction contains a valuable, long-term lesson for the AI.

Do not focus on trivial facts or conversational fluff. Look for:
- Novel strategies or techniques the AI used successfully.
- Explicit user corrections or negative feedback that reveals a flaw in the AI's approach.
- Key insights or context that could significantly improve performance on similar tasks in the future.

You must respond in a specific JSON format. Your entire output must be a single JSON object with two keys:
1. "is_memorable": A boolean value (true or false).
2. "memory_content": If "is_memorable" is true, this should be a string containing a concise, generalized principle or recommendation for the AI's future self. If "is_memorable" is false, this should be null.
"""

    turn_text = f"Analyze the following interaction:\n\n[User Prompt]:\n{user_prompt}\n\n[Assistant Response]:\n{assistant_response}"

    # NOTE: Replace this with your actual LLM call function.
    # It needs to be able to handle a system prompt and user content.
    # from your_llm_library import generate_llm_response
    # json_response_str = await generate_llm_response(system_prompt, turn_text, model, tokenizer)

    # Placeholder for the LLM response string
    json_response_str = '{"is_memorable": false, "memory_content": null}'  # Replace with actual call

    try:
        # Robustly parse the JSON from the LLM's response
        json_part = json_response_str[json_response_str.find('{'):json_response_str.rfind('}') + 1]
        data = json.loads(json_part)
        if "is_memorable" in data and "memory_content" in data:
            return data
        logger.error("LLM response for evaluation is missing required keys.")
        return {"is_memorable": False, "memory_content": None}
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Failed to parse JSON from LLM evaluation response: {e}")
        return {"is_memorable": False, "memory_content": None}