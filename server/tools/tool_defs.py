
from server.tools.web_search import WebSearchTool
from server.tools.coding import ExecutePythonCode
from server.tools.file_management import WriteFileTool, ReadFileTool, ListDirectoryTool
from server.tools.git import GitManagementSkill
from tools.tool_creation.tool_creator import ToolCreator # The updated one

STATIC_TOOLS = [
    ExecutePythonCode(),
    WriteFileTool(),
    ReadFileTool(),
    ListDirectoryTool(),
    ToolCreator(),
    WebSearchTool()
] + GitManagementSkill().get_tools()
