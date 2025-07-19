import docker
import shlex
import json

class ExecutePythonCode:
    """
    A tool for executing Python code in a secure Docker container.
    """
    # 1. The schema is now a simple dictionary attribute.
    #    Our server reads this to tell the LLM how the tool works.
    function = {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": (
                "Executes a string of Python code in a secure Docker container and returns the output. "
                "IMPORTANT: This tool cannot save files or interact with the local file system. "
                "DO NOT try to read or write files with this tool. "
                "It is solely for running code and getting the results back."
            ),
            "parameters": [{
                "name": "code",
                "type": "string",
                "description": "The raw Python code to be executed.",
                "required": True
            }]
        }
    }

    # 2. The core logic is now in a 'run' method.
    #    It now accepts a dictionary of arguments directly.
    def run(self, arguments: dict) -> str:
        """
        Executes the tool's logic.
        `arguments` is a dictionary containing the code to run.
        """
        try:
            print(f"Executing tool 'execute_python_code' with arguments: {arguments}")
            code = arguments.get('code')
            if not code:
                return "❌ Error: No code provided to execute."

            # The Docker execution logic remains the same - it's solid.
            client = docker.from_env()
            escaped_code = shlex.quote(code.strip())
            command = f'python -c {escaped_code}'

            container_output = client.containers.run(
                image="python:3.11-slim",
                command=command,
                remove=True,
                stderr=True,
                stdout=True,
                detach=False
            )

            result = container_output.decode('utf-8').strip()

            return result

        except docker.errors.ContainerError as e:
            error_output = e.stderr.decode('utf-8') if e.stderr else str(e)
            return f"❌ An error occurred during code execution:\n```\n{error_output.strip()}\n```"
        except docker.errors.APIError as e:
            return f"❌ An error occurred with the Docker API: {e}"
        except Exception as e:
            return f"❌ An unexpected error occurred: {type(e).__name__}: {str(e)}"