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
            "description": "Writes content to a file, creating parent directories and overwriting the file if it exists.",
            # CORRECTED: 'parameters' is now a dictionary
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative or absolute path to the file."
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write to the file."
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    }

    def run(self, arguments: dict) -> str:
        """Executes the tool's logic."""
        try:
            file_path = arguments.get('file_path')
            content = arguments.get('content')

            if not file_path or content is None:
                return "❌ Error: Both 'file_path' and 'content' are required."

            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

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
            # CORRECTED: 'parameters' is now a dictionary
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative or absolute path to the file to be read."
                    }
                },
                "required": ["file_path"]
            }
        }
    }

    def run(self, arguments: dict) -> str:
        """Executes the tool's logic."""
        try:
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
            # CORRECTED: 'parameters' is now a dictionary
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the directory to inspect. Defaults to '.' (the current directory)."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list directories and files recursively. Defaults to True."
                    }
                }
            }
        }
    }

    def run(self, arguments: dict) -> str:
        """Executes the tool's logic."""
        try:
            path = arguments.get('path', '../../tools')
            recursive = arguments.get('recursive', True)

            if not os.path.isdir(path):
                return f"❌ Error: The path '{path}' is not a valid directory."

            if not recursive:
                entries = os.listdir(path)
                return f"✅ Contents of '{path}':\n" + "\n".join(entries)

            output = f"✅ Recursive listing for '{path}':\n"
            for root, dirs, files in os.walk(path):
                # Sort directories and files to ensure consistent output
                dirs.sort()
                files.sort()
                level = root.replace(path, '').count(os.sep)
                indent = ' ' * 4 * level
                output += f"{indent}{os.path.basename(root) or '.'}/\n"
                sub_indent = ' ' * 4 * (level + 1)
                for d in dirs:
                    output += f"{sub_indent}📁 {d}/\n"
                for f in files:
                    output += f"{sub_indent}📄 {f}\n"
            return output.strip()
        except Exception as e:
            return f"❌ An unexpected error occurred: {type(e).__name__}: {str(e)}"