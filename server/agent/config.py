# agent/config.py

"""
Central configuration for the agent, including model parameters and constants.
"""
import re
from typing import Any

# --- Agent Behavior ---
MAX_TURNS = 10
LOOP_DETECTION_THRESHOLD = 2
# --- Roles ---
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
SYSTEM_ROLE = "system"

# --- Regex Patterns ---
TOOL_CALL_REGEX = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
THOUGHT_SUMMARY_REGEX = re.compile(r"<thought_summary>(.*?)</thought_summary>", re.DOTALL)
SENTENCE_END_REGEX = re.compile(r"[.?!](?:\s|$)")

def get_generation_kwargs(tokenizer: Any, temperature: float = 0.6, max_new_tokens: int = 1024) -> dict[str, Any]:
    """Constructs the generation arguments for the model."""
    return {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
    }