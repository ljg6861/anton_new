# context_builder.py

"""
Responsible for gathering and assembling all contextual information required
for the agent's prompt.
"""

import logging
import subprocess
from typing import List, Dict

from server.agent.prompts import get_planner_prompt, get_doer_prompt
from server.agent.rag_manager import rag_manager
from server.agent.tools.tool_manager import tool_manager

# Set up a dedicated logger for this module
logger = logging.getLogger(__name__)


class ContextBuilder:
    """Gathers and formats context for the agent."""

    def get_tool_context(self) -> List[Dict]:
        """Gets the schemas of the available tools."""
        tool_schemas = tool_manager.get_tool_schemas()
        logger.info(f"Making {len(tool_schemas)} tools available for this request.")
        return tool_schemas

    def get_project_structure(self) -> str:
        try:
            # 1) find the absolute path to the Git repo root
            toplevel = subprocess.run(
                ['git', 'rev-parse', '--show-toplevel'],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()

            # 2) list all tracked files from that root
            result = subprocess.run(
                ['git', 'ls-files'],
                cwd=toplevel,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Successfully retrieved project structure via 'git ls-files'.")
            return result.stdout.strip()

        except FileNotFoundError:
            msg = "Error: Git command not found. Is Git installed and in your PATH?"
            logger.error(msg)
            return msg

        except subprocess.CalledProcessError as e:
            # could fail either on rev-parse (not a repo) or ls-files
            stderr = e.stderr.strip() or e.stdout.strip()
            msg = f"Error retrieving project structure. Are you in a Git repo?\nDetails: {stderr}"
            logger.error(msg)
            return msg

    async def build_system_prompt_planner(self, query: str) -> str:
        system_prompt = (
            get_planner_prompt().replace('{tools}', str(self.get_tool_context())))
        return system_prompt

    async def build_system_prompt_doer(self, query: str) -> str:
        system_prompt = (
            "--- AUTOMATIC CONTEXT ---\n"
            f'## Your source code files:\n {self.get_project_structure()}\n'
            f'## Things that might help:\n {self.find_relevant_context(query)}'
            "\n--- END CONTEXT ---\n\n"
            + get_doer_prompt().replace('{tools}', str(self.get_tool_context())))
        return system_prompt

    def find_relevant_context(self, query: str):
        print(f"\n🔍 Searching for context related to: '{query}'")
        relevant_docs = rag_manager.retrieve_knowledge(query=query, top_k=2)
        if relevant_docs:
            for doc in relevant_docs:
                print(f"  - Source: {doc['source']}\n    Knowledge: {doc['text']}")
        else:
            print("  - No relevant knowledge found.")
