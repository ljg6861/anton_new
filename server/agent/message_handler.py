# agent/message_handler.py

"""
Handles conversation history management and reasoning loop detection.
"""
from typing import Any, List, Dict


def prepare_initial_messages(request_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepares the initial list of messages for the agent from the request."""
    # This is a good place to inject a system prompt if needed.
    return [msg.model_dump() for msg in request_messages]


def handle_loop_detection(
        recent_thoughts: list[str],
        current_thought: str,
        messages: list[dict],
        logger: Any,
        threshold: int
) -> bool:
    """
    Detects and handles reasoning loops by modifying the message history.

    Returns:
        True if a loop was detected and an intervention message was added.
    """
    normalized_thought = " ".join(current_thought.lower().split())
    recent_thoughts.append(normalized_thought)

    if len(recent_thoughts) > threshold:
        recent_thoughts.pop(0)

    if len(recent_thoughts) == threshold and len(set(recent_thoughts)) == 1:
        logger.warning("Agent is stuck in a reasoning loop. Intervening.")
        messages.append({
            "role": "user",
            "content": "You are repeating the same thought process. You must change your plan. Re-evaluate the problem, try a different tool, or ask for help."
        })
        recent_thoughts.clear()
        return True  # Loop detected

    return False  # No loop