import docker
import shlex
import json

class ExecutePythonCode:
    """
    A tool for executing Python code in a secure Docker container.
    """
    function = {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": (
                "Executes a string of Python code in a secure Docker container and returns the output. "
                "IMPORTANT: This tool cannot save files or interact with the local file system. "
                "It is solely for running code and getting the results back."
            ),
            # CORRECTED: 'parameters' is now a dictionary
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The raw Python code to be executed."
                    }
                },
                "required": ["code"]
            }
        }
    }

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

            client = docker.from_env()
            # Use shlex.quote for robust shell escaping
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
            return f"✅ Code executed successfully:\n---\n{result}\n---" if result else "✅ Code executed successfully with no output."

        except docker.errors.ContainerError as e:
            error_output = e.stderr.decode('utf-8') if e.stderr else str(e)
            return f"❌ An error occurred during code execution:\n```\n{error_output.strip()}\n```"
        except docker.errors.APIError as e:
            # This can happen if the Docker daemon isn't running
            return f"❌ An error occurred with the Docker API: {e}. Is the Docker daemon running?"
        except Exception as e:
            return f"❌ An unexpected error occurred: {type(e).__name__}: {str(e)}"