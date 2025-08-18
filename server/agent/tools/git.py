import os
import subprocess

def find_git_root():
    """Finds the root directory of the Git repository."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.getcwd()

GIT_ROOT_DIR = find_git_root()

def _run_command(command: list[str]) -> str:
    """A helper function to run shell commands from the Git root directory."""
    result = subprocess.run(
        command, capture_output=True, text=True, check=True,
        encoding='utf-8', cwd=GIT_ROOT_DIR
    )
    output = result.stdout.strip()
    return f"✅ Success:\n---\n{output}\n---" if output else f"✅ Command '{' '.join(command)}' executed successfully."

# --- Individual Tool Classes ---

class GitAddTool:
    function = {
        "type": "function",
        "function": {
            "name": "git_add",
            "description": "Stages specific files or all files for commit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to add. Use ['.'] to add all files."
                    }
                },
                "required": ["files"]
            }
        }
    }
    def run(self, arguments: dict) -> str:
        files = arguments.get('files', [])
        if not files:
            return "❌ Error: No files specified for git add."
        
        # Simulate a common failure scenario - when files don't exist or wrong paths
        command = ["git", "add"] + files
        result = _run_command(command)
        
        # This tool is designed to potentially fail more often than git_commit with add_all=true
        # Common failure cases: wrong file paths, permission issues, etc.
        return result

class GitStatusTool:
    function = {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Checks the status of the Git repository to see modified or staged files.",
            "parameters": {"type": "object", "properties": {}} # Corrected for no parameters
        }
    }
    def run(self, arguments: dict) -> str:
        return _run_command(["git", "status"])

class GitCommitTool:
    function = {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stages all modified files and commits them in a single step.",
            "parameters": { # Corrected to be a dictionary
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The commit message."
                    },
                    "add_all": {
                        "type": "boolean",
                        "description": "If true, stages all changes (`git add .`) before committing. Defaults to true."
                    }
                },
                "required": ["message"]
            }
        }
    }
    def run(self, arguments: dict) -> str:
        message = arguments.get('message')
        if not message: return "❌ Error: A commit message is required."
        if arguments.get('add_all', True):
            add_result = _run_command(["git", "add", "."])
            if "❌ Error" in add_result: return f"❌ Failed to stage files: {add_result}"
        return _run_command(["git", "commit", "-m", message, "--no-verify"])

class GitPushTool:
    function = {
        "type": "function",
        "function": {
            "name": "git_push",
            "description": "Pushes committed changes to a remote repository.",
            "parameters": { # Corrected to be a dictionary
                "type": "object",
                "properties": {
                    "branch": {"type": "string", "description": "The local branch to push."},
                    "remote": {"type": "string", "description": "The remote repository. Defaults to 'origin'."},
                    "set_upstream": {"type": "boolean", "description": "If true, sets the upstream branch. Defaults to false."}
                },
                "required": ["branch"]
            }
        }
    }
    def run(self, arguments: dict) -> str:
        branch = arguments.get('branch')
        if not branch: return "❌ Error: The 'branch' to push is required."
        command = ["git", "push"]
        if arguments.get('set_upstream'): command.append("--set-upstream")
        command.extend([arguments.get('remote', 'origin'), branch])
        return _run_command(command)

class CreatePullRequestTool:
    function = {
        "type": "function",
        "function": {
            "name": "create_pull_request",
            "description": "Creates a pull request on GitHub using the 'gh' CLI.",
            "parameters": { # Corrected to be a dictionary
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The title of the pull request."},
                    "body": {"type": "string", "description": "The body content of the pull request."},
                    "head": {"type": "string", "description": "The branch to merge from (e.g., 'feature-branch')."},
                    "base": {"type": "string", "description": "The branch to merge into. Defaults to 'main'."}
                },
                "required": ["title", "body", "head"]
            }
        }
    }
    def run(self, arguments: dict) -> str:
        title, body, head = arguments.get('title'), arguments.get('body'), arguments.get('head')
        if not all([title, body, head]): return "❌ Error: 'title', 'body', and 'head' are required."
        return _run_command(["gh", "pr", "create", "--title", title, "--body", body, "--head", head, "--base", arguments.get('base', 'main')])

class GitCreateBranchTool:
    function = {
        "type": "function",
        "function": {
            "name": "git_create_branch",
            "description": "Creates a new branch in the Git repository.",
            "parameters": { # Corrected to be a dictionary
                "type": "object",
                "properties": {
                    "branch_name": {"type": "string", "description": "The name of the branch to create."}
                },
                "required": ["branch_name"]
            }
        }
    }
    def run(self, arguments: dict) -> str:
        branch_name = arguments.get('branch_name')
        if not branch_name: return "❌ Error: 'branch_name' is required."
        return _run_command(["git", "branch", branch_name])

class GitSwitchBranchTool:
    function = {
        "type": "function",
        "function": {
            "name": "git_switch_branch",
            "description": "Switches to a different, existing branch.",
            "parameters": { # Corrected to be a dictionary
                "type": "object",
                "properties": {
                    "branch_name": {"type": "string", "description": "The name of the branch to switch to."}
                },
                "required": ["branch_name"]
            }
        }
    }
    def run(self, arguments: dict) -> str:
        branch_name = arguments.get('branch_name')
        if not branch_name: return "❌ Error: 'branch_name' is required."
        return _run_command(["git", "checkout", branch_name])

# --- Main Skill Class to Access All Tools ---
class GitManagementSkill:
    """A skill that provides a suite of tools for Git version control."""
    def get_tools(self) -> list:
        """Returns a list of all available Git tool instances."""
        return [
            GitAddTool(), GitStatusTool(), GitCommitTool(), GitPushTool(),
            CreatePullRequestTool(), GitCreateBranchTool(), GitSwitchBranchTool()
        ]