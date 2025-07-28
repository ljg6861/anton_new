from server.agent.tools.tool_creation.tool_creator import ToolCreator
from server.agent.tools.web_search import WebSearchTool
from server.agent.tools.coding import ExecutePythonCode
from server.agent.tools.file_management import WriteFileTool, ReadFileTool, ListDirectoryTool
from server.agent.tools.git import GitManagementSkill

STATIC_TOOLS = [
    ExecutePythonCode(),
    WriteFileTool(),
    ReadFileTool(),
    ListDirectoryTool(),
    ToolCreator(),
    WebSearchTool()
] + GitManagementSkill().get_tools()
