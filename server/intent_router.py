# server/intent_router.py

import logging
from typing import List, Dict, Literal

from transformers import PreTrainedModel, PreTrainedTokenizer

# Set up a dedicated logger for this module
logger = logging.getLogger(__name__)

# The prompt for the classification model. It's a simple, zero-shot prompt.
ROUTER_PROMPT = """
You are an expert intent classifier. Your job is to determine the user's intent based on their message.
Classify the user's intent into one of the following categories:
- 'simple_chat': For greetings, simple questions, chitchat, or any request that does not require external tools or knowledge of the file system.
- 'task_execution': For complex requests that require executing code, accessing files, using git, searching the web, or any other tool-based action.

User prompt: "{user_prompt}"

Based on the prompt, the user's intent is:
"""

Intent = Literal['simple_chat', 'task_execution']

async def classify_intent(
    user_prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer
) -> Intent:
    """
    Uses a lightweight LLM call to classify the user's intent.
    """
    logger.info(f"Classifying intent for prompt: '{user_prompt[:100]}...'")

    prompt = ROUTER_PROMPT.format(user_prompt=user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Use a very restrictive generation to get just the classification
    outputs = await model.generate(
        **inputs,
        max_new_tokens=5,  # We only need a few tokens
        temperature=0.01,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the classification from the full text
    cleaned_response = response_text.split("intent is:")[-1].strip().lower()

    if 'task_execution' in cleaned_response:
        logger.info("Intent classified as: task_execution")
        return 'task_execution'

    logger.info("Intent classified as: simple_chat")
    return 'simple_chat'
