# utils/tool_formatter.py
from typing import List

def format_tools_for_prompt(tools: List[object]) -> str:
    """Formats the tool list for inclusion in a prompt."""
    tool_strings = [f"- {tool.name}: {tool.description}" for tool in tools]
    print('\n'.join(tool_strings))
    return "\n".join(tool_strings)