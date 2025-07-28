"""
Handles parsing and processing of <learn> tags from the model's output.
"""
import json
import re
from typing import Any

from server.agent.rag_manager import rag_manager

LEARN_TAG_REGEX = re.compile(r"<learn>(.*?)</learn>", re.DOTALL)


async def process_learning_request(
    response_buffer: str,
    logger: Any
) -> bool:
    learn_match = LEARN_TAG_REGEX.search(response_buffer)
    if not learn_match:
        return False

    learn_content = learn_match.group(1).strip()
    logger.info("Detected <learn> tag. Attempting to process.")

    try:
        learn_data = json.loads(learn_content)
        new_knowledge = learn_data.get("new_knowledge")
        source = learn_data.get("source")

        if not new_knowledge or not source:
            raise KeyError("JSON must contain 'new_knowledge' and 'source' keys.")

        # Add the extracted information to the knowledge base
        rag_manager.add_knowledge(text=new_knowledge, source=source)
        logger.info("Successfully processed and stored new knowledge.")

    except (json.JSONDecodeError, KeyError) as e:
        # Log the error but don't interrupt the flow. The agent doesn't need
        # to know about a failure to learn.
        error_msg = f"Error: Invalid <learn> tag content. Reason: {e}"
        logger.error(f"{error_msg}\nContent: {learn_content}")

    return True