# /multi_agent_project/agents/planner.py
from typing import List

from langchain_core.tools import BaseTool
from langchain_ollama import OllamaLLM
from .base_agent import Agent
from prompts import PLANNER_PROMPT


class PlannerAgent(Agent):
    """An agent that breaks down a user request into steps based on available tools."""

    def __init__(self, llm: OllamaLLM, tools: List[BaseTool]):
        # 1. Create a simple, readable summary of the tools for the prompt
        tool_summary = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)

        # 2. Format the main prompt template with the dynamic tool summary
        prompt_template = PLANNER_PROMPT.format(
            original_prompt="{original_prompt}",
            chat_history="{chat_history}",
            completed_steps_summary="{completed_steps_summary}",
            tools=tool_summary
        )

        super().__init__(llm, prompt_template, name="Planner")
