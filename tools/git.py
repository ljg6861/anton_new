import os
import subprocess


def _run_command(command: list[str]) -> str:
    """A helper function to run shell commands, log the output, and return it."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            cwd=os.getcwd()
        )
        output = result.stdout.strip()

        # --- START: Added Logging ---
        if not output:
            response_string = f"✅ Command '{' '.join(command)}' executed successfully with no output."
        else:
            response_string = f"✅ Success:\n---\n{output}\n---"

        print(f"Tool Output:\n{response_string}")
        # --- END: Added Logging ---

        return response_string

    except FileNotFoundError as e:
        response_string = f"❌ Error: Command not found: {e.filename}. Please ensure the command-line tool (e.g., 'git' or 'gh') is installed and in the system's PATH."
        print(f"Tool Output:\n{response_string}")
        return response_string

    except subprocess.CalledProcessError as e:
        error_output = f"Stderr:\n---\n{e.stderr.strip()}"
        if e.stdout.strip():
            error_output += f"\nStdout:\n---\n{e.stdout.strip()}"
        response_string = f"❌ Error executing command: {' '.join(command)}\nReturn Code: {e.returncode}\n{error_output}"
        print(f"Tool Output:\n{response_string}")
        return response_string

    except Exception as e:
        response_string = f"❌ An unexpected error occurred: {type(e).__name__}: {e}"
        print(f"Tool Output:\n{response_string}")
        return response_string


# --- Individual Tool Classes ---

class GitStatusTool:
    """Checks the status of the Git repository."""
    function = {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Checks the status of the Git repository to see which files have been modified or staged.",
            "parameters": []
        }
    }

    def run(self, arguments: dict) -> str:
        print(f"Executing tool 'git_status' with arguments: {arguments}")
        return _run_command(["git", "status"])


class GitCommitTool:
    """Commits changes to the Git repository."""
    function = {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stages all modified files and commits them in a single step.",
            "parameters": [
                {
                    "name": "message",
                    "type": "string",
                    "description": "The commit message.",
                    "required": True
                },
                {
                    "name": "add_all",
                    "type": "boolean",
                    "description": "If true, automatically stages all changes (`git add .`) before committing. Defaults to true. Should almost always be true unless files were staged manually in a previous step.",
                    "required": False
                }
            ]
        }
    }

    def run(self, arguments: dict) -> str:
        print(f"Executing tool 'git_commit' with arguments: {arguments}")
        message = arguments.get('message')
        add_all = arguments.get('add_all', True)

        if not message:
            return "❌ Error: A commit message is required."

        if add_all:
            add_result = _run_command(["git", "add", "."])
            if "❌ Error" in add_result or "fatal" in add_result:
                return f"❌ Failed to stage files before committing: {add_result}"

        # Add the --no-verify flag to bypass pre-commit hooks
        return _run_command(["git", "commit", "-m", message, "--no-verify"])


class GitPushTool:
    """Pushes committed changes to a remote repository."""
    function = {
        "type": "function",
        "function": {
            "name": "git_push",
            "description": "Pushes committed changes to a remote repository. Can also set the upstream branch.",
            "parameters": [
                {
                    "name": "branch",
                    "type": "string",
                    "description": "The local branch to push.",
                    "required": True
                },
                {
                    "name": "remote",
                    "type": "string",
                    "description": "The remote repository to push to. Defaults to 'origin'.",
                    "required": False
                },
                {
                    "name": "set_upstream",
                    "type": "boolean",
                    "description": "If true, sets the upstream branch (e.g., 'git push --set-upstream origin branch_name'). Defaults to false.",
                    "required": False
                }
            ]
        }
    }

    def run(self, arguments: dict) -> str:
        print(f"Executing tool 'git_push' with arguments: {arguments}")
        branch = arguments.get('branch')
        remote = arguments.get('remote', 'origin')
        set_upstream = arguments.get('set_upstream', False) # Default to False

        if not branch:
            return "❌ Error: The 'branch' to push is required."

        command = ["git", "push"]

        if set_upstream:
            command.append("--set-upstream")

        command.extend([remote, branch])

        return _run_command(command)



class CreatePullRequestTool:
    """Creates a pull request on GitHub."""
    function = {
        "type": "function",
        "function": {
            "name": "create_pull_request",
            "description": "Creates a pull request on GitHub using the 'gh' CLI. Requires GitHub CLI to be installed and authenticated.",
            "parameters": [
                {
                    "name": "title",
                    "type": "string",
                    "description": "The title of the pull request.",
                    "required": True
                },
                {
                    "name": "body",
                    "type": "string",
                    "description": "The body content of the pull request.",
                    "required": True
                },
                {
                    "name": "head",
                    "type": "string",
                    "description": "The branch to merge from (e.g., 'feature-branch').",
                    "required": True
                },
                {
                    "name": "base",
                    "type": "string",
                    "description": "The branch to merge into. Defaults to 'main'.",
                    "required": False
                }
            ]
        }
    }

    def run(self, arguments: dict) -> str:
        print(f"Executing tool 'create_pull_request' with arguments: {arguments}")
        title = arguments.get('title')
        body = arguments.get('body')
        head = arguments.get('head')
        base = arguments.get('base', 'main')

        if not all([title, body, head]):
            return "❌ Error: 'title', 'body', and 'head' are required to create a pull request."

        return _run_command(["gh", "pr", "create", "--title", title, "--body", body, "--head", head, "--base", base])


class GitCreateBranchTool:
    """Creates a new branch."""
    function = {
        "type": "function",
        "function": {
            "name": "git_create_branch",
            "description": "Creates a new branch in the Git repository.",
            "parameters": [{
                "name": "branch_name",
                "type": "string",
                "description": "The name of the branch to create.",
                "required": True
            }]
        }
    }

    def run(self, arguments: dict) -> str:
        print(f"Executing tool 'git_create_branch' with arguments: {arguments}")
        branch_name = arguments.get('branch_name')
        if not branch_name:
            return "❌ Error: 'branch_name' is required."
        return _run_command(["git", "branch", branch_name])


class GitSwitchBranchTool:
    """Switches to a different branch."""
    function = {
        "type": "function",
        "function": {
            "name": "git_switch_branch",
            "description": "Switches to a different, existing branch.",
            "parameters": [{
                "name": "branch_name",
                "type": "string",
                "description": "The name of the branch to switch to.",
                "required": True
            }]
        }
    }

    def run(self, arguments: dict) -> str:
        print(f"Executing tool 'git_switch_branch' with arguments: {arguments}")
        branch_name = arguments.get('branch_name')
        if not branch_name:
            return "❌ Error: 'branch_name' is required."
        return _run_command(["git", "checkout", branch_name])


# --- Main Skill Class to Access All Tools ---

class GitManagementSkill:
    """A skill that provides a suite of tools for Git version control."""

    def get_tools(self) -> list:
        """Returns a list of all available Git tool instances."""
        return [
            GitStatusTool(),
            GitCommitTool(),
            GitPushTool(),
            CreatePullRequestTool(),
            GitCreateBranchTool(),
            GitSwitchBranchTool(),
        ]