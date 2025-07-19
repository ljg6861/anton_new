import logging
from typing import List, Dict, Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, render_text_description
from langchain_ollama import OllamaLLM
import re

from prompts import EXECUTOR_PROMPT
from utils.output_parser import CustomReActOutputParser

logger = logging.getLogger(__name__)


class ExecutorAgent:
    """
    An agent that uses a set of tools to execute a given task by reasoning about the steps.
    """

    def __init__(self, llm: OllamaLLM, tools: List[BaseTool]):
        self.logger = logger
        self.agent_executor = None
        self.logger.info("Initializing ExecutorAgent with tools...")

        tool_description_string = render_text_description(tools)

        # 2. Create the prompt template and partial-fill the 'tools' variable.
        prompt = PromptTemplate.from_template(EXECUTOR_PROMPT).partial(
            tools=tool_description_string
        )

        output_parser = CustomReActOutputParser()

        # 3. Create the ReAct agent.
        react_agent = create_react_agent(llm=llm, tools=tools,
                                         output_parser=output_parser,
                                         prompt=prompt)

        # Create the AgentExecutor, which is the runtime that executes the agent's decisions.
        self.agent_executor = AgentExecutor(
            agent=react_agent,
            tools=tools,
            verbose=True,  # Logs the agent's thoughts and actions. Highly recommended.
            handle_parsing_errors=True,  # Makes the agent more robust.
        )
        self.logger.info("ExecutorAgent initialized successfully.")


    def invoke(self, inputs: Dict[str, Any]) -> str:
        """
        Invokes the agent executor with the given inputs.
        The executor will run the ReAct loop until the task is complete.
        """
        self.logger.info(f"Executing task: {inputs.get('task_to_execute')}")
        print(inputs)

        response = self.agent_executor.invoke(inputs)
        raw_output = response.get("output", "No output from executor.")

        # 🛠️ Sanitize any stray <think>...</think> blocks from hallucinated responses
        cleaned_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

        self.logger.info(f"Executor: {cleaned_output}")


        return cleaned_output