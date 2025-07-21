# In your tools/tool_defs.py file
from server.tools.web_search import WebSearchTool
from tools.coding import ExecutePythonCode
from tools.file_management import WriteFileTool, ReadFileTool, ListDirectoryTool
from tools.git import GitManagementSkill
from tools.tool_creation.tool_creator import ToolCreator

# This list is what your server will use to generate the system prompt.
ALL_TOOLS = [
                ExecutePythonCode(),
                WriteFileTool(),
                ReadFileTool(),
                ListDirectoryTool(),
                ToolCreator(), WebSearchTool()
            ] + GitManagementSkill().get_tools()

# You can create a registry for your client-side code to easily find and run the tools.
TOOL_REGISTRY = {
    tool.function['function']['name']: tool for tool in ALL_TOOLS
}