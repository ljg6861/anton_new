import os
import importlib.util
import re

# The ToolRegistry class is correct as-is.
class ToolRegistry:
    """Manages the registration and retrieval of available tools."""

    def __init__(self):
        self.tools = {}

    def register(self, tool_instance):
        function_name = tool_instance.function["function"]["name"]
        self.tools[function_name] = tool_instance
        print(f"🔧 Tool '{function_name}' registered.")

    def get_tool_schemas(self):
        return [tool.function for tool in self.tools.values()]


class ToolCreator:
    """
    A tool for creating new, usable tools for the LLM.
    It writes Python code to a file and dynamically loads it.
    """
    function = {
        "type": "function",
        "function": {
            "name": "create_new_tool",
            "description": (
                "Writes the Python code for a new tool to a file in the 'tools' directory "
                "and makes it available for use in subsequent steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The snake_case name for the new tool (e.g., 'file_writer')."
                    },
                    "tool_code": {
                        "type": "string",
                        "description": "A string containing the entire Python code for the new tool's class."
                    }
                },
                "required": ["tool_name", "tool_code"]
            }
        }
    }

    def run(self, arguments: dict) -> str:
        """
        Executes tool logic: saves code, dynamically loads the new tool,
        and registers it to the client's live tool lists.
        """
        tool_name = arguments.get('tool_name')
        tool_code = arguments.get('tool_code')

        if not tool_name or not tool_code:
            return "❌ Error: 'tool_name' and 'tool_code' are required."

        try:
            sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '', tool_name)
            tools_dir = "tools"
            os.makedirs(tools_dir, exist_ok=True)
            file_path = os.path.join(tools_dir, f"{sanitized_name}.py")

            with open(file_path, "w", encoding='utf-8') as f:
                f.write(tool_code)

            spec = importlib.util.spec_from_file_location(sanitized_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            return f"✅ Success: Tool '{sanitized_name}' created and loaded. It is now available for the next turn."
        except Exception as e:
            return f"❌ An unexpected error occurred: {str(e)}"