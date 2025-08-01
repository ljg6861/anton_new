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

    async def build_system_prompt_planner(self, task_description: str = "") -> str:
        """Build system prompt for planner with memory context injection."""
        # Get relevant memories for the current task
        memory_context = self._get_memory_context(task_description)
        
        system_prompt = (
            get_planner_prompt()
            .replace('{tools}', str(self.get_tool_context()))
            .replace('{memory_context}', memory_context)
        )
        return system_prompt

    async def build_system_prompt_doer(self) -> str:
        system_prompt = (
            get_doer_prompt().replace('{tools}', str(self.get_tool_context())))
        return system_prompt

    def _get_memory_context(self, task_description: str) -> str:
        """Retrieve relevant memories for the current task."""
        if not task_description.strip():
            return "No relevant memories found for this task."
        
        logger.info(f"Retrieving memories for task: {task_description[:100]}...")
        relevant_docs = rag_manager.retrieve_knowledge(query=task_description, top_k=3)
        
        if not relevant_docs:
            return "No relevant memories found for this task."
        
        memory_sections = []
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.get('source', 'Unknown source')
            text = doc.get('text', 'No content available')
            
            # Truncate very long memories to keep prompt manageable
            if len(text) > 300:
                text = text[:300] + "..."
            
            memory_sections.append(f"{i}. From {source}:\n   {text}")
        
        memory_context = "\n\n".join(memory_sections)
        logger.info(f"Retrieved {len(relevant_docs)} relevant memories for task context")
        
        return memory_context

    def find_relevant_context(self, query: str):
        print(f"\n🔍 Searching for context related to: '{query}'")
        relevant_docs = rag_manager.retrieve_knowledge(query=query, top_k=2)
        if relevant_docs:
            for doc in relevant_docs:
                print(f"  - Source: {doc['source']}\n    Knowledge: {doc['text']}")
        else:
            print("  - No relevant knowledge found.")
