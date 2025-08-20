import asyncio
import logging
import time
import traceback
from typing import Dict, List
from server.agent.agentic_flow.full_agentic_flow import determine_route, execute_agentic_flow
from server.agent.agentic_flow.helpers_and_prompts import COMPLEX_PLANNER_PROMPT, SIMPLE_PLANNER_PROMPT
from server.agent.agentic_flow.task_flow import execute_executor, handle_task_route, initial_assessment, execute_planner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_determine_route(messages, expected):
    """Asserts that the determine_route function returns the expected route."""
    route = await determine_route(messages)
    assert route == expected, f"Expected '{expected}', but got '{route}'"

async def test_chat_route(messages):
    start_time = time.time()
    execute_agentic_flow(messages)
    end_time = time.time()
    assert start_time - end_time < 10, "Chat route took too long."

async def test_task_assessment(messages, expected):
    assessment = await initial_assessment(messages)
    assert assessment == expected, f"Expected '{expected}', but got '{assessment}'"

async def test_planner(prompt, messages):
    plan = await execute_planner(prompt, messages)
    final_plan = False
    assert len(plan) > 0, "Plan should contain at least one step."
    for step in plan:
        assert "step" in step, "Each step should have a 'step' key."
        assert "tool" in step, "Each step should have a 'tool' key."
        assert "args" in step, "Each step should have an 'args' key."
        assert "thought" in step, "Each step should have a 'thought' key."
        if step["tool"] == "final_answer":
            final_plan = True
    assert final_plan, "Plan must include a final step with the 'final_answer' tool."

async def test_task_executor(prompt, messages, overall_goal):
    plan = {'plan': [{'step': 1, 'thought': 'I need to retrieve the official text of the Pledge of Allegiance. Using web_search will provide the most accurate and current wording.', 'tool': 'web_search', 'args': {'query': 'official Pledge of Allegiance text'}}, {'step': 2, 'thought': "Now that I have the Pledge text from the web search, I'll write it to the specified file using the write_file tool.", 'tool': 'write_file', 'args': {'file_path': 'pledge.txt', 'content': '{{step_1_output}}'}}, {'step': 3, 'thought': "The Pledge of Allegiance has been successfully written to pledge.txt. I'll confirm completion to the user.", 'tool': 'final_answer', 'args': {'summary': "The Pledge of Allegiance has been written to 'pledge.txt' with the official text retrieved via web search."}}]}
    plan = plan['plan']
    results = []
    result = await execute_executor(overall_goal, plan, results, plan[0])
    assert result is not None, "Executor should return a result."

async def test_task_route(messages):
    async for token in handle_task_route(messages, "Test task"):
        assert token is not None, "Task route should return a result."
        print(f"Task route output: {token}")

class TestRunner:
    """Manages test execution, concurrency, and result counting."""
    def __init__(self, concurrency_limit=5):
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.passes = 0
        self.failures = 0
        self.total = 0

    async def run(self, test_coro, name):
        """Runs a single test coroutine under the semaphore."""
        self.total += 1
        async with self.semaphore:
            try:
                await test_coro
                self.passes += 1
                print(f"✅ PASS: {name}")
            except AssertionError as e:
                self.failures += 1
                print(f"❌ FAIL: {name} ({e})")
            except Exception:
                self.failures += 1
                # Prints the full error for unexpected issues
                print(f"💥 ERROR: {name}")
                traceback.print_exc()

    def print_summary(self):
        """Prints the final results of the test run."""
        print("\n" + "---" * 10)
        print("📊 Test Run Summary")
        print(f"Total Tests: {self.total}")
        print(f"✅ Passes: {self.passes}")
        print(f"❌ Failures: {self.failures}")
        print("---" * 10)


async def main():
    """Defines the test suite and orchestrates the test runner."""
    runner = TestRunner(concurrency_limit=5)

    # A dictionary of test cases for clarity {test_name: test_coroutine}
    test_cases = {
        # "Chat: Weather": test_determine_route(
        #     [{"role": "user", "content": "What's the weather like today?"}], "chat"
        # ),
        # "Chat: Joke": test_determine_route(
        #     [{"role": "user", "content": "Tell me a joke."}], "chat"
        # ),
        # "Chat: Capital": test_determine_route(
        #     [{"role": "user", "content": "What's the capital of France?"}], "chat"
        # ),
        # "Task: To-do list": test_determine_route(
        #     [{"role": "user", "content": "Create a to-do list based off my schedule"}], "task"
        # ),
        # "Task: Schedule meeting": test_determine_route(
        #     [{"role": "user", "content": "Schedule a meeting."}], "task"
        # ),
        # "Task: Set reminder": test_determine_route(
        #     [{"role": "user", "content": "Set a reminder."}], "task"
        # ),
        # "Chat: Weather": test_chat_route(
        #     [{"role": "user", "content": "What's the weather like today?"}]
        # ),
        # "Assessment: Code": test_task_assessment(
        #     [{"role": "user", "content": "Write some python code to count from 1 to 100, but dont output it to a file"}], "Sufficient"
        # ),
        # "Assessment: File Writing": test_task_assessment(
        #     [{"role": "user", "content": "Write some python code to count from 1 to 100, and output it to a file"}], "Sufficient"
        # ),
        # "Assessment: Incapable": test_task_assessment(
        #     [{"role": "user", "content": "Browse the web and purchase me the cheapest doll you can find"}], "Requires_Discovery"
        # ),
        # "Assessment: Incapable": test_task_assessment(
        #     [{"role": "user", "content": "Create a new tool for yourself where that allows you to multiply numbers"}], "Requires_Discovery"
        # ),
        # "Simple Plan: Output code to a file": test_planner(SIMPLE_PLANNER_PROMPT,
        #     [{"role": "user", "content": "Write python code to output 'Hello, World!' to a file."}]
        # ),
        # "Complex Plan: Create a new tool": test_planner(COMPLEX_PLANNER_PROMPT,
        #     [{"role": "user", "content": "Determine if you have the ability to fetch web pages. If you dont, implement a new tool to do so."}]
        # ),
        # "Task Executor: Write the pledge": test_task_executor(SIMPLE_PLANNER_PROMPT,
        #     [{"role": "user", "content": "Write the pledge of allegiance to pledge.txt"}], "Write the pledge of allegiance to pledge.txt"
        # ),
        # "Task Route: Write the pledge": test_task_route(
        #     [{"role": "user", "content": "Write the pledge of allegiance to pledge.txt"}]
        # ),
        "Task Route: Create a tool": test_task_route(
            [{"role": "user", "content": "Create a new tool for yourself that allows you to multiply numbers. You MUST test it by calling the tool after it has been created. If you create the tool correctly, it will automatically be available to you."}]
        ),
    }

    # Create a list of tasks to run concurrently
    tasks = [runner.run(coro, name) for name, coro in test_cases.items()]
    await asyncio.gather(*tasks)

    runner.print_summary()


if __name__ == "__main__":
    asyncio.run(main())