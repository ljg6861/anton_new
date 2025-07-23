# tools/tool_creation/tool_creator.py

import os
import re
import importlib.util
import inspect
from tools.tool_manager import tool_manager  # <-- IMPORT THE MANAGER


class ToolCreator:
    """
    A tool for creating new, usable tools. It writes the code to a file
    and registers the new tool with the central ToolManager.
    """
    function = {
        "type": "function",
        "function": {
            "name": "create_new_tool",
            "description": (
                "Writes the Python code for a new tool to a file and makes it "
                "immediately available for use in subsequent steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The snake_case name for the new tool's file (e.g., 'file_writer')."
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
        """Saves code, dynamically loads the new tool, and registers it."""
        tool_name = arguments.get('tool_name')
        tool_code = arguments.get('tool_code')

        if not tool_name or not tool_code:
            return "❌ Error: 'tool_name' and 'tool_code' are required."

        try:
            sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '', tool_name)
            # We'll create a dedicated directory for custom tools
            tools_dir = "custom_tools"
            os.makedirs(tools_dir, exist_ok=True)

            # Add an __init__.py to make it a package
            init_path = os.path.join(tools_dir, "__init__.py")
            if not os.path.exists(init_path):
                open(init_path, 'a').close()

            file_path = os.path.join(tools_dir, f"{sanitized_name}.py")

            with open(file_path, "w", encoding='utf-8') as f:
                f.write(tool_code)

            # Dynamically load the new module
            module_name = f"{tools_dir}.{sanitized_name}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the class inside the loaded module and register it
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module_name:
                    new_tool_instance = obj()
                    tool_manager.register(new_tool_instance)  # <-- THE FIX
                    return f"✅ Success: Tool '{sanitized_name}' created and loaded. It is now available."

            return "❌ Error: Could not find a class to load in the provided tool code."
        except Exception as e:
            return f"❌ An unexpected error occurred: {str(e)}"