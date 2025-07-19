import logging
import time
from typing import List, Dict, Generator, Optional

from langchain_ollama import OllamaLLM

from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.summarizer import SummarizerAgent
from prompts import ORCHESTRATOR_PROMPT, CODE_REVIEWER, MAIN_LLM
from tools.tool_defs import ALL_TOOLS
logger = logging.getLogger(__name__)
reviewing = False


def start_code_review(history):
    global reviewing
    if reviewing:
        return
    reviewing = True
    orchestrator = MultiStepAgentOrchestrator()

    orchestrator.stream(CODE_REVIEWER + history)
    reviewing = False


class MultiStepAgentOrchestrator:
    """
    Orchestrates a team of specialized agents to solve a user's request.

    This orchestrator is responsible for validating if the task is complete,
    rather than relying on the planner agent.
    """

    def __init__(self, max_steps: int = 40):
        logger.info("Initializing MultiStepAgentOrchestrator...")
        self.llm = OllamaLLM(model=MAIN_LLM)
        self.max_steps = max_steps

        # A dedicated prompt for the orchestrator to check for task completion.
        self.completion_checker_prompt = ORCHESTRATOR_PROMPT

        self.tools = ALL_TOOLS
        self.planner = PlannerAgent(self.llm, self.tools)
        self.executor = ExecutorAgent(self.llm, self.tools)
        self.summarizer = SummarizerAgent(self.llm)
        logger.info("Orchestrator initialized successfully with all agents.")

    def _format_completed_steps(self, completed_steps: List[Dict]) -> str:
        """Helper to format the history of completed steps."""
        if not completed_steps:
            return "No work has been done yet."
        return "\n\n".join(
            f"Step {i + 1}: {step['plan']}\nResult: {step['result']}"
            for i, step in enumerate(completed_steps)
        )

    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Helper to format the chat history for the planner."""
        if not chat_history:
            return "No previous conversation."
        return "\n".join(
            f"{entry['role']}: {entry['content']}" for entry in chat_history
        )

    def _is_task_complete(self, original_prompt: str, completed_steps: List[Dict]) -> bool:
        """
        Uses an LLM to determine if the task is complete based on the work done.
        """
        if not completed_steps:
            return False

        completed_steps_summary = self._format_completed_steps(completed_steps)
        prompt = self.completion_checker_prompt.format(
            original_prompt=original_prompt,
            completed_steps_summary=completed_steps_summary
        )
        logger.info("--- Checking for task completion ---")
        response = self.llm.invoke(prompt)
        logger.info(f"Completion check response: '{response.strip()}'")
        if "ERROR" in response.strip():
            start_code_review(completed_steps_summary)
            while reviewing:
                time.sleep(1)

        return "YES" in response.strip()

    def stream(self, user_input: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Generator[
        str, None, None]:
        """
        Runs the agent loop and streams the final response.
        The orchestrator, not the planner, now validates if the task is complete.
        """
        chat_history = chat_history or []
        completed_steps = []

        yield "Orchestrator started. Beginning planning process...\n\n"

        for i in range(self.max_steps):
            current_context = self._format_completed_steps(completed_steps)

            # 1. Validation: Check if the task is complete before planning the next step.
            # This check occurs after at least one step has been attempted.
            if i > 0 and self._is_task_complete(user_input, completed_steps):
                logger.info("Orchestrator determined the task is complete.")
                yield "Task fulfillment confirmed. Proceeding to summarization.\n\n"
                break

            # 2. Planning
            logger.info(f"--- Cycle {i + 1}: Planning ---")
            planner_task = {
                "original_prompt": user_input,
                "chat_history": self._format_chat_history(chat_history),
                "completed_steps_summary": current_context,
            }
            next_step_plan = self.planner.invoke(planner_task)

            yield f"**Step {i + 1}:** {next_step_plan}\n"
            logger.info(f"Plan for step {i + 1}: {next_step_plan}")

            # Note: This currently reviews the *plan* from the planner. If you intend to
            # review code *output* from the executor, this call should be moved
            # after the execution step and passed the `execution_result`.

            # 3. Execution
            logger.info(f"--- Cycle {i + 1}: Executing ---")
            executor_task = {
                "original_prompt": user_input,
                "context": current_context,
                "task_to_execute": next_step_plan,
            }
            execution_result = self.executor.invoke(executor_task)

            yield f"*Result:* {execution_result}\n\n"
            logger.info(f"Result of step {i + 1}: {execution_result}")

            completed_steps.append({"plan": next_step_plan, "result": execution_result})
        else:
            # This block executes if the for loop completes without a 'break'.
            logger.warning(f"Max steps ({self.max_steps}) reached without task completion.")
            yield f"Max steps ({self.max_steps}) reached. Proceeding to summarization.\n\n"

        # 4. Summarization
        logger.info("--- Final Step: Summarizing ---")
        yield "Synthesizing final answer...\n\n"
        final_context = self._format_completed_steps(completed_steps)
        summarizer_task = {
            "original_prompt": user_input,
            "full_context": final_context,
        }
        final_answer = self.summarizer.invoke(summarizer_task)

        logger.info("Orchestration complete.")
        yield "--- Final Answer ---\n"
        yield final_answer