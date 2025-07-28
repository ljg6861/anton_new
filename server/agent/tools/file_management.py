import os
import json

# --- START: Added for Project Root ---
# Get the absolute path of the directory containing this script (server/tools)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the project root as two directories up from the script's location
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, '..', '..'))


# --- END: Added for Project Root ---


def _resolve_path(user_path: str) -> str:
    """
    Safely resolves a user-provided path against the project root.
    Prevents path traversal attacks.
    """
    # Treat all paths as relative to the project root.
    # If a user provides an absolute path (e.g., "/etc/passwd"),
    # os.path.join will correctly handle it if the base is absolute.
    # However, for security, we explicitly join from our trusted root.

    # Create the full path by joining the project root with the user-provided path.
    # This automatically handles both relative ('data/file.txt') and "absolute-like" ('/data/file.txt') inputs safely.
    combined_path = os.path.join(PROJECT_ROOT, user_path)

    # Normalize the path to resolve '..' etc., and get the real, absolute path.
    safe_path = os.path.realpath(combined_path)

    # Security Check: Ensure the final, resolved path is still inside the PROJECT_ROOT.
    if not safe_path.startswith(PROJECT_ROOT):
        raise ValueError(f"Path traversal attempt detected. Access denied for path: {user_path}")

    return safe_path


class WriteFileTool:
    """
    A tool for writing content to a file within the project directory.
    """
    function = {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes content to a file relative to the project root. Creates parent directories and overwrites the file if it exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file, relative to the project root."
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
            file_path_arg = arguments.get('file_path')
            content = arguments.get('content')

            if not file_path_arg or content is None:
                return "❌ Error: Both 'file_path' and 'content' are required."

            # Use the helper to get a safe, absolute path
            safe_file_path = _resolve_path(file_path_arg)

            dir_name = os.path.dirname(safe_file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            with open(safe_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Show the relative path in the success message for clarity
            return f"✅ Successfully wrote {len(content)} characters to '{file_path_arg}'."
        except ValueError as e:
            return f"❌ Security Error: {str(e)}"
        except Exception as e:
            return f"❌ An unexpected error occurred: {type(e).__name__}: {str(e)}"


class ReadFileTool:
    """
    A tool for reading the content of a file from the project directory.
    """
    function = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the entire content of a specified file relative to the project root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to be read, relative to the project root."
                    }
                },
                "required": ["file_path"]
            }
        }
    }

    def run(self, arguments: dict) -> str:
        """Executes the tool's logic."""
        try:
            file_path_arg = arguments.get('file_path')
            if not file_path_arg:
                return "❌ Error: 'file_path' is required."

            # Use the helper to get a safe, absolute path
            safe_file_path = _resolve_path(file_path_arg)

            if not os.path.exists(safe_file_path):
                return f"❌ Error: The file '{file_path_arg}' was not found."

            with open(safe_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except ValueError as e:
            return f"❌ Security Error: {str(e)}"
        except Exception as e:
            return f"❌ An unexpected error occurred: {type(e).__name__}: {str(e)}"


class ListDirectoryTool:
    """
    A tool for listing the contents of a directory within the project.
    """
    function = {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists all files and subdirectories within a given path, relative to the project root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the directory to inspect, relative to the project root. Defaults to '.' (the project root)."
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
            # Default path is now '.', representing the project root itself
            path_arg = arguments.get('path', '')
            recursive = arguments.get('recursive', True)

            # Use the helper to get a safe, absolute path
            safe_path = _resolve_path(path_arg)

            if not os.path.isdir(safe_path):
                return f"❌ Error: The path '{path_arg}' is not a valid directory."

            display_path = path_arg if path_arg != '.' else 'project root'

            if not recursive:
                entries = os.listdir(safe_path)
                return f"✅ Contents of '{display_path}':\n" + "\n".join(entries)

            output = f"✅ Recursive listing for '{display_path}':\n"
            for root, dirs, files in os.walk(safe_path):
                # Sort for consistent output
                dirs.sort()
                files.sort()

                # Calculate indentation level relative to the starting safe_path
                level = root.replace(safe_path, '').count(os.sep)
                indent = ' ' * 4 * level

                if level == 0:
                    output += f"{indent}{os.path.basename(root) or '.'}/\n"
                else:
                    output += f"{indent}📁 {os.path.basename(root)}/\n"

                sub_indent = ' ' * 4 * (level + 1)
                for f in files:
                    output += f"{sub_indent}📄 {f}\n"
            return output.strip()
        except ValueError as e:
            return f"❌ Security Error: {str(e)}"
        except Exception as e:
            return f"❌ An unexpected error occurred: {type(e).__name__}: {str(e)}"