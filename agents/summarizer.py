# /multi_agent_project/agents/planner.py

from langchain_ollama import OllamaLLM
from .base_agent import Agent
from prompts import EXECUTOR_PROMPT, SUMMARIZER_PROMPT


class SummarizerAgent(Agent):
    """An agent that breaks down a user request into steps."""
    def __init__(self, llm: OllamaLLM):
        super().__init__(llm, SUMMARIZER_PROMPT, name="Summarizer")
