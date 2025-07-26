import json

import requests

from tools.tool_creation.tool_creator import ToolCreator


# --- Placeholder LLM Simulation Functions ---

def get_owner_llm_response(history: list) -> str:
    """Simulates the Owner LLM deciding the next high-level step."""
    print("🤖 OWNER LLM (Placeholder): Analyzing history...")
    history_str = str(history).lower()

    if "'file_writer' created and loaded" not in history_str:
        return "Create a new tool named 'file_writer'. It should accept 'filename' and 'content' and write the content to the file."
    elif "wrote content to hello.txt" not in history_str:
        return "Use the 'file_writer' tool to create a file named 'hello.txt' with the content 'Hello from my new AI agent!'."
    else:
        return "Conclude: Goal Achieved."


def get_doer_llm_response(context: list, tool_schemas: list) -> dict:
    """Simulates the Doer LLM deciding which tool to call."""
    print(f"🤖 DOER LLM (Placeholder): Analyzing task...")
    task = context[-1]['content'].lower()

    if "create a new tool" in task and "file_writer" in task:
        # Simulate the LLM generating the code for the file writer tool
        tool_code = """
import os

class FileWriter:
    function = {
        "type": "function",
        "function": {
            "name": "write_to_file",
            "description": "Writes given content to a specified file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "The name of the file to write to."},
                    "content": {"type": "string", "description": "The content to write into the file."}
                }, "required": ["filename", "content"]
            }
        }
    }
    def run(self, arguments: dict):
        try:
            with open(arguments.get('filename'), 'w') as f:
                f.write(arguments.get('content', ''))
            return f"✅ Success: Wrote content to {arguments.get('filename')}"
        except Exception as e:
            return f"❌ Error: {e}"
"""
        return {
            "tool_calls": [{
                "function": {
                    "name": "create_new_tool",
                    "arguments": json.dumps({"tool_name": "file_writer", "tool_code": tool_code.strip()})
                }
            }]
        }
    elif "use the 'file_writer' tool" in task:
        return {
            "tool_calls": [{
                "function": {
                    "name": "write_to_file",
                    "arguments": json.dumps({"filename": "hello.txt", "content": "Hello from my new AI agent!"})
                }
            }]
        }
    return {"tool_calls": None, "content": "Task complete or no tool needed."}


# --- Agent Classes ---

class OwnerAgent:
    """Orchestrates the overall goal."""

    def __init__(self, goal: str):
        self.goal = goal
        self.history = [
            {"role": "system", "content": "You are the Owner..."},
            {"role": "user", "content": f"The overall goal is: {self.goal}"}
        ]

    def determine_next_step(self, last_doer_result=None):
        if last_doer_result:
            self.history.append({"role": "user", "content": f"The last task resulted in: {last_doer_result}"})

        # Call the placeholder LLM function
        next_task = get_owner_llm_response(self.history)

        if "conclude" in next_task.lower():
            return None

        self.history.append({"role": "assistant", "content": next_task})
        print(f"👑 OWNER: New task for Doer -> '{next_task}'")
        return next_task


API_BASE_URL = "http://localhost:8000/v1"


class DoerAgent:
    def __init__(self, task_prompt: str, tool_registry: ToolRegistry):
        self.task = task_prompt
        self.tool_registry = tool_registry
        self.context = [
            {"role": "system", "content": "You are the Doer..."},
            {"role": "user", "content": self.task}
        ]

    def execute_task(self):
        print("🛠️  DOER: Preparing to execute task via API...")

        # Get the schemas for all currently registered tools
        tool_schemas = self.tool_registry.get_tool_schemas()

        # Prepare the request payload for the API
        payload = {
            "messages": self.context,
            "tools": tool_schemas,
            "temperature": 0.3
        }

        try:
            # Make the call to your local server
            response = requests.post(f"{API_BASE_URL}/agent/chat", json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes

            # The response from your execute_agent function will be processed here
            # For now, we'll just print the raw text response
            result = response.text
            print(f"🛠️  DOER: Got result from server -> '{result}'")
            return result
        except requests.exceptions.RequestException as e:
            error_message = f"❌ API call failed: {e}"
            print(error_message)
            return error_message

# --- Main Orchestration Loop ---

def main(initial_goal: str):
    """The main orchestration loop."""
    tool_registry = ToolRegistry()
    tool_registry.register(ToolCreator())

    owner = OwnerAgent(initial_goal)
    last_result = None

    for i in range(10):  # Safety break
        print(f"\n--- Cycle {i + 1} {'-' * 40}\n")
        task = owner.determine_next_step(last_result)
        if task is None:
            print("\n👑 OWNER: Goal achieved! Loop is complete.")
            break

        doer = DoerAgent(task, tool_registry)
        result = doer.execute_task()
        last_result = result
    else:
        print("\n⚠️ Loop limit reached.")


if __name__ == "__main__":
    goal = """
    First, create a tool named 'file_writer' that can write content to a file.
    Then, use that new tool to create 'hello.txt' with the content 'Hello from my new AI agent!'.
    """
    main(goal)