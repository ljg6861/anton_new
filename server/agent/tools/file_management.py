import os
import json

import pathspec

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
            return content.replace("tool_call", "tool call placeholder")
        except ValueError as e:
            return f"❌ Security Error: {str(e)}"
        except Exception as e:
            return f"❌ An unexpected error occurred: {type(e).__name__}: {str(e)}"


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
        """Executes the tool's logic, filtering results based on .gitignore."""
        try:
            path_arg = arguments.get('path', '.')
            recursive = arguments.get('recursive', True)

            # These are assumed to be defined elsewhere in your project
            project_root = PROJECT_ROOT
            safe_path = _resolve_path(path_arg)

            if not os.path.isdir(safe_path):
                return f"❌ Error: The path '{path_arg}' is not a valid directory."

            # Load .gitignore rules from the project root
            spec = None
            gitignore_path = os.path.join(project_root, '.gitignore')
            if os.path.exists(gitignore_path):
                with open(gitignore_path, 'r') as f:
                    spec = pathspec.PathSpec.from_lines('gitwildmatch', f)

            display_path = path_arg if path_arg != '.' else 'project root'
            output_message = f"✅ Contents of '{display_path}' (respecting .gitignore):\n"

            # --- Non-Recursive Listing ---
            if not recursive:
                entries = os.listdir(safe_path)
                if spec:
                    relative_path = os.path.relpath(safe_path, project_root)
                    if relative_path == '.': relative_path = ''
                    entries = [
                        e for e in entries
                        if not spec.match_file(os.path.join(relative_path, e))
                    ]
                return output_message + "\n".join(sorted(entries))

            # --- Recursive Listing ---
            output_lines = []
            has_content = False
            for root, dirs, files in os.walk(safe_path, topdown=True):
                # Filter directories and files using .gitignore spec
                if spec:
                    relative_root = os.path.relpath(root, project_root)
                    if relative_root == '.': relative_root = ''

                    # Filter dirs IN-PLACE so os.walk doesn't traverse them
                    dirs[:] = [d for d in dirs if not spec.match_file(os.path.join(relative_root, d))]
                    files = [f for f in files if not spec.match_file(os.path.join(relative_root, f))]

                # Don't bother printing empty directories
                if not files and not dirs:
                    continue

                has_content = True
                dirs.sort()
                files.sort()

                level = root.replace(safe_path, '', 1).count(os.sep)
                indent = ' ' * 4 * level

                dir_name = os.path.basename(root)
                if root == safe_path:
                    dir_name = display_path

                output_lines.append(f"{indent}📁 {dir_name}/")

                sub_indent = ' ' * 4 * (level + 1)
                for f in files:
                    output_lines.append(f"{sub_indent}📄 {f}")

            if not has_content and os.listdir(safe_path):
                return f"✅ All contents of '{display_path}' are ignored by .gitignore."

            if not has_content:
                return f"✅ The directory '{display_path}' is empty."

            return output_message + "\n".join(output_lines)

        except ValueError as e:
            return f"❌ Security Error: {str(e)}"
        except Exception as e:
            return f"❌ An unexpected error occurred: {type(e).__name__}: {str(e)}"