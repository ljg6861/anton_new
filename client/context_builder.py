# context_builder.py

"""
Responsible for gathering and assembling all contextual information required
for the agent's prompt.
"""

import logging
from typing import List, Dict

from server.agent.prompts import get_thinking_prompt
from server.tools.tool_manager import tool_manager
from utils.context_gatherer import get_git_diff

# Set up a dedicated logger for this module
logger = logging.getLogger(__name__)


class ContextBuilder:
    """Gathers and formats context for the agent."""

    def __init__(self):
        """
        Initializes the ContextBuilder.

        Args:
            api_client: An instance of ApiClient to fetch remote context (e.g., memories).
        """

    def get_tool_context(self) -> List[Dict]:
        """Gets the schemas of the available tools."""
        tool_schemas = tool_manager.get_tool_schemas()
        logger.info(f"Making {len(tool_schemas)} tools available for this request.")
        return tool_schemas

    async def build_system_prompt(self, user_prompt: str) -> str:
        logger.info("Building system prompt with git and memory context.")
        git_context = get_git_diff()

        system_prompt = (
            "--- AUTOMATIC CONTEXT ---\n"
            f"## Git Status:\n{git_context}\n"
            "\n--- END CONTEXT ---\n\n"
            + get_thinking_prompt() +
            "Based on the context above, please begin the following:\n"
        )
        return system_prompt
