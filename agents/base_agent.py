# /multi_agent_project/agents/base_agent.py

import logging
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

class Agent:
    """
    A reusable agent class that can be configured with a specific role and LLM.
    """

    def __init__(self, llm: OllamaLLM, role_prompt: str, name: str):
        self.llm = llm
        self.role_prompt = role_prompt
        self.name = name
        logger.info(f"Initialized Agent: {self.name}")

    def invoke(self, task_description: str) -> str:
        """
        Invokes the agent to perform a task.
        """
        prompt = self.role_prompt.format(**task_description)

        return self.llm.invoke(prompt)