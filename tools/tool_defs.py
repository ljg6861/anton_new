# In your tools/tool_defs.py file
from tools.coding import ExecutePythonCode
from tools.file_management import WriteFileTool, ReadFileTool, ListDirectoryTool
from tools.git import GitManagementSkill

# This list is what your server will use to generate the system prompt.
ALL_TOOLS = [
                ExecutePythonCode(),
                WriteFileTool(),
                ReadFileTool(),
                ListDirectoryTool(),
            ] + GitManagementSkill().get_tools()

# You can create a registry for your client-side code to easily find and run the tools.
TOOL_REGISTRY = {
    tool.function['function']['name']: tool.run for tool in ALL_TOOLS
}
