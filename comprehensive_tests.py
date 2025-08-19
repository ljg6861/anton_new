import asyncio
import time
import traceback

from server.agent.full_agentic_flow import determine_route, execute_agentic_flow

async def test_determine_route(messages, expected):
    """Asserts that the determine_route function returns the expected route."""
    route = await determine_route(messages)
    assert route == expected, f"Expected '{expected}', but got '{route}'"

async def test_chat_route(messages):
    start_time = time.time()
    execute_agentic_flow(messages)
    end_time = time.time()
    assert start_time - end_time < 10, "Chat route took too long."


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
        "Chat: Weather": test_determine_route(
            [{"role": "user", "content": "What's the weather like today?"}], "chat"
        ),
        "Chat: Joke": test_determine_route(
            [{"role": "user", "content": "Tell me a joke."}], "chat"
        ),
        "Chat: Capital": test_determine_route(
            [{"role": "user", "content": "What's the capital of France?"}], "chat"
        ),
        "Task: To-do list": test_determine_route(
            [{"role": "user", "content": "Create a to-do list based off my schedule"}], "task"
        ),
        "Task: Schedule meeting": test_determine_route(
            [{"role": "user", "content": "Schedule a meeting."}], "task"
        ),
        "Task: Set reminder": test_determine_route(
            [{"role": "user", "content": "Set a reminder."}], "task"
        ),
        "Chat: Weather": test_chat_route(
            [{"role": "user", "content": "What's the weather like today?"}]
        ),
    }

    # Create a list of tasks to run concurrently
    tasks = [runner.run(coro, name) for name, coro in test_cases.items()]
    await asyncio.gather(*tasks)

    runner.print_summary()


if __name__ == "__main__":
    asyncio.run(main())