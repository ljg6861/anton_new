# tools/tool_manager.py

from typing import Dict, Any, List


class ToolManager:
    """Manages the registration, retrieval, and execution of all tools."""

    def __init__(self):
        self.tools: Dict[str, Any] = {}
        print("✅ ToolManager initialized.")

    def register(self, tool_instance: Any):
        """Registers a single tool instance."""
        function_name = tool_instance.function["function"]["name"]
        self.tools[function_name] = tool_instance
        print(f"🔧 Tool '{function_name}' registered.")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns the JSON schemas of all registered tools."""
        return [tool.function for tool in self.tools.values()]

    def run_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Finds a tool in the registry and executes it with the given arguments."""
        if tool_name not in self.tools:
            return f"❌ Error: Tool '{tool_name}' not found."

        tool_instance = self.tools[tool_name]
        return tool_instance.run(tool_args)


# Create a single, global instance that the entire application will share.
tool_manager = ToolManager()