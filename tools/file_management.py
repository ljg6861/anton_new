import os
import json


class WriteFileTool:
    """
    A tool for writing content to a file.
    """
    function = {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes content to a specified file, creating parent directories if they don't exist and overwriting the file if it already exists.",
            "parameters": [
                {
                    "name": "file_path",
                    "type": "string",
                    "description": "The relative or absolute path to the file.",
                    "required": True
                },
                {
                    "name": "content",
                    "type": "string",
                    "description": "The full content to write to the file.",
                    "required": True
                }
            ]
        }
    }

    def run(self, arguments: dict) -> str:
        """Executes the tool's logic."""
        try:
            print(f"Executing tool 'write_file' with arguments: {arguments}")
            file_path = arguments.get('file_path')
            content = arguments.get('content')

            if not file_path or content is None:
                return "❌ Error: Both 'file_path' and 'content' are required."

            # Create parent directories if they don't exist
            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            # Write the content to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return f"✅ Successfully wrote {len(content)} characters to '{file_path}'."

        except Exception as e:
            return f"❌ An unexpected error occurred: {type(e).__name__}: {str(e)}"


class ReadFileTool:
    """
    A tool for reading the content of a file.
    """
    function = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the entire content of a specified file.",
            "parameters": [{
                "name": "file_path",
                "type": "string",
                "description": "The relative or absolute path to the file to be read.",
                "required": True
            }]
        }
    }

    def run(self, arguments: dict) -> str:
        """Executes the tool's logic."""
        try:
            print(f"Executing tool 'read_file' with arguments: {arguments}")
            file_path = arguments.get('file_path')

            if not file_path:
                return "❌ Error: 'file_path' is required."

            if not os.path.exists(file_path):
                return f"❌ Error: The file '{file_path}' was not found."

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return content

        except Exception as e:
            return f"❌ An unexpected error occurred: {type(e).__name__}: {str(e)}"


class ListDirectoryTool:
    """
    A tool for listing the contents of a directory.
    """
    function = {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists all files and subdirectories within a given path.",
            "parameters": [
                {
                    "name": "path",
                    "type": "string",
                    "description": "The path to the directory to inspect. Defaults to the current directory '.'",
                    "required": False
                },
                {
                    "name": "recursive",
                    "type": "boolean",
                    "description": "Whether to list directories and files recursively. Defaults to True.",
                    "required": False
                }
            ]
        }
    }

    def run(self, arguments: dict) -> str:
        """Executes the tool's logic."""
        try:
            print(f"Executing tool 'list_directory' with arguments: {arguments}")
            # Set defaults if parameters are not provided
            path = arguments.get('path', '.')
            recursive = arguments.get('recursive', True)

            if not os.path.isdir(path):
                return f"❌ Error: The path '{path}' is not a valid directory."

            if not recursive:
                entries = os.listdir(path)
                return f"✅ Contents of '{path}':\n" + "\n".join(entries)

            output = f"✅ Recursive listing for directory '{path}':\n"
            for root, _, files in os.walk(path):
                level = root.replace(path, '').count(os.sep)
                indent = ' ' * 4 * level
                output += f"{indent}{os.path.basename(root) or '.'}/\n"
                sub_indent = ' ' * 4 * (level + 1)
                for f in files:
                    output += f"{sub_indent}📄 {f}\n"
            return output.strip()

        except Exception as e:
            return f"❌ An unexpected error occurred: {type(e).__name__}: {str(e)}"