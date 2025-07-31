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


async def save_reflection(original_task: str, reflection_data: dict, logger: Any) -> None:
    """
    ### UPDATED ###
    Saves the structured reflection data to the RAG knowledge base.

    This now uses the RAGManager to embed and store the new knowledge.
    """
    logger.info(f"Saving reflection to knowledge base: \"{reflection_data.get('summary', 'N/A')}\"")
    try:
        # 1. 🧠 Format the learned insight into a text document for storage.
        # This text will be converted into a vector for semantic search.
        knowledge_text = (
            f"Regarding the task '{original_task}', a key learning was identified.\n"
            f"Summary of outcome: {reflection_data['summary']}\n"
            f"Key Takeaway: {reflection_data['key_takeaway']}\n"
            f"Strategy Used: {reflection_data['strategy']}"
        )

        # 2. Add the new knowledge to the in-memory RAG index.
        # The source helps identify where this knowledge came from.
        rag_manager.add_knowledge(
            text=knowledge_text,
            source=f"reflection_on_{original_task[:30]}"  # A descriptive source
        )

        # 3. 💾 Persist the updated index and document store to disk.
        # This saves all knowledge added since the last save.
        rag_manager.save()

        logger.info("Successfully saved reflection to the RAG knowledge base.")

    except Exception as e:
        logger.error(f"Failed to save reflection to RAG knowledge base. Error: {e}", exc_info=True)

