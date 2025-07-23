import os
from typing import List

class ToolLoader:
    @staticmethod
    def list_tools() -> List[str]:
        """Returns a list of available tool names from the 'tools' directory."""
        tool_files = [f for f in os.listdir('tools') if f.endswith('.py') and f != 'tool_loader.py']
        return [f[:-3] for f in tool_files]