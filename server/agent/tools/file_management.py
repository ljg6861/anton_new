import os
import json

import pathspec

# --- START: Added for Project Root ---
# Get the absolute path of the directory containing this script (server/tools)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the project root as two directories up from the script's location
def find_project_root(start_path):
    current_path = os.path.abspath(start_path)
    while True:
        if os.path.exists(os.path.join(current_path, '.git')):
            return current_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path: # Reached the filesystem root
            return None
        current_path = parent_path

PROJECT_ROOT = find_project_root(os.path.dirname(os.path.abspath(__file__)))
if not PROJECT_ROOT:
    raise FileNotFoundError("Could not find project root containing a .git directory.")

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
        file_path_arg = arguments.get('file_path')
        content = arguments.get('content')

        if not file_path_arg or content is None:
            raise ValueError("Both 'file_path' and 'content' are required.")

        # Use the helper to get a safe, absolute path
        safe_file_path = _resolve_path(file_path_arg)

        dir_name = os.path.dirname(safe_file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(safe_file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Show the relative path in the success message for clarity
        return f"✅ Successfully wrote {len(content)} characters to '{file_path_arg}'."


class ReadFileTool:
    """
    A tool for reading the content of a file from the project directory.
    """
    function = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the entire content of a specified file relative to the project root. Accepts either 'file_path' or 'path' parameter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to be read, relative to the project root."
                    },
                    "path": {
                        "type": "string",
                        "description": "Alternative parameter name for file_path."
                    }
                },
                "anyOf": [
                    {"required": ["file_path"]},
                    {"required": ["path"]}
                ]
            }
        }
    }

    def run(self, arguments: dict) -> str:
        """Executes the tool's logic."""
        # Support both 'file_path' and 'path' parameter names for compatibility
        file_path_arg = arguments.get('file_path') or arguments.get('path')
        
        if not file_path_arg:
            raise ValueError("Missing required parameter: either 'file_path' or 'path' must be provided")

        # Use the helper to get a safe, absolute path
        safe_file_path = _resolve_path(file_path_arg)

        if not os.path.exists(safe_file_path):
            raise FileNotFoundError(f"The file '{file_path_arg}' was not found.")

        with open(safe_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Sanitize potentially problematic patterns that could be misinterpreted as tool calls
        # Replace <tool_code> patterns with escaped versions to prevent false tool call detection
        content = content.replace("<tool_code>", "&lt;tool_code&gt;")
        content = content.replace("</tool_code>", "&lt;/tool_code&gt;")
        content = content.replace("<tool_call>", "&lt;tool_call&gt;")
        content = content.replace("</tool_call>", "&lt;/tool_call&gt;")
        
        return content


class ListDirectoryTool:
    """
    A tool for listing the contents of a directory, ignoring files and folders
    specified in the project's .gitignore file.
    """
    function = {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists files and subdirectories within a given path, ignoring anything in .gitignore.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the directory, relative to the project root. Defaults to '.' (the project root)."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list contents recursively. Defaults to True."
                    }
                }
            }
        }
    }

def run(self, arguments: dict) -> str:
    """
    Lists files and directories not excluded by .gitignore, starting from the
    project root. The output is a simple newline-separated list of relative
    paths, suitable for agent consumption.
    """
    path_arg = arguments.get('path', '.')
    recursive = arguments.get('recursive', True)

    # All paths are resolved relative to the project root.
    project_root = PROJECT_ROOT
    scan_path = _resolve_path(path_arg)

    if not os.path.isdir(scan_path):
        return "" # Return empty string for invalid or non-existent paths.

    # Load .gitignore rules from the project root.
    spec = None
    gitignore_path = os.path.join(project_root, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            spec = pathspec.PathSpec.from_lines('gitwildmatch', f)

    output_paths = []

    for root, dirs, files in os.walk(scan_path, topdown=True):
        # Determine the current directory's path relative to the project root.
        relative_root = os.path.relpath(root, project_root)
        if relative_root == '.':
            relative_root = ''

        # Filter directories and files using .gitignore spec if it exists.
        if spec:
            # Filter dirs in-place to prevent `os.walk` from traversing them.
            dirs[:] = [d for d in dirs if not spec.match_file(os.path.join(relative_root, d))]
            # Filter files.
            files = [f for f in files if not spec.match_file(os.path.join(relative_root, f))]

        # Add resulting directories to the output list.
        for d in dirs:
            # Construct the full relative path and normalize to forward slashes.
            full_path = os.path.join(relative_root, d).replace(os.sep, '/')
            output_paths.append(f"{full_path}/")

        # Add resulting files to the output list.
        for f in files:
            # Construct the full relative path and normalize to forward slashes.
            full_path = os.path.join(relative_root, f).replace(os.sep, '/')
            output_paths.append(full_path)
        
        # If not recursive, stop after processing the top-level directory.
        if not recursive:
            break

    # Return a single, sorted string with each path on a new line.
    return "\n".join(sorted(output_paths))