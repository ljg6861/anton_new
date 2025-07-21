import asyncio
import logging
import json
from typing import AsyncIterator, Dict, List, Optional
import httpx

# The client defines the list of tools available for a given request.
from tools.tool_defs import ALL_TOOLS
from utils.context_gatherer import get_git_diff

# Set up a dedicated logger for this module
logger = logging.getLogger(__name__)


class Anton:
    """
    Client for interacting with the Anton agent server.
    This class prepares the context and streams the response from the server's
    self-contained agent loop.
    """

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.http_client = httpx.AsyncClient(timeout=None)
        logger.info(f"Anton client initialized, pointing to server at {api_base_url}")

    async def _get_relevant_memories(self, user_prompt: str) -> str:
        """Retrieves memories relevant to the user's prompt from the server."""
        logger.info("Attempting to retrieve relevant memories...")
        try:
            response = await self.http_client.post(
                f"{self.api_base_url}/v1/memory/retrieve",
                json={"query": user_prompt}
            )
            response.raise_for_status()
            memories = response.json().get("memories", [])
            if not memories:
                logger.info("No relevant memories were found.")
                return "No relevant memories found."
            logger.info(f"Retrieved {len(memories)} relevant memories.")
            formatted_memories = "\n".join(f"- {mem}" for mem in memories)
            return f"## Relevant Memories:\n{formatted_memories}"
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return "Could not retrieve memories."

    async def _store_conversation_turn(self, user_prompt: str, assistant_response: str):
        """Sends the completed conversation turn to the server for memory storage."""
        if not assistant_response.strip():
            logger.info("Assistant response was empty, skipping memory storage.")
            return
        logger.info("Sending conversation to memory for evaluation and storage.")
        try:
            await self.http_client.post(
                f"{self.api_base_url}/v1/memory/evaluate_and_store",
                json={"user_prompt": user_prompt, "assistant_response": assistant_response}
            )
            logger.info("Memory evaluation and storage request sent successfully.")
        except Exception as e:
            logger.error(f"Failed to send turn for evaluation: {e}")

    async def stream_response(
            self, user_prompt: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[Dict[str, str]]:
        """
        Gathers context, sends a single request to the agent, and streams its
        processed response.
        """
        # --- 1. Context Gathering ---
        yield {"type": "info", "content": "Gathering context..."}
        logger.info("Phase 1: Gathering Context")
        git_context = get_git_diff()
        memory_context = await self._get_relevant_memories(user_prompt)

        tool_schemas = [tool.function for tool in ALL_TOOLS]
        logger.info(f"Client is making {len(tool_schemas)} tools available for this request.")

        system_context = (
            "--- AUTOMATIC CONTEXT ---\n"
            f"## Git Status:\n{git_context}\n"
            f"{memory_context}"
            "\n--- END CONTEXT ---"
        )
        messages = chat_history or []
        messages.append({"role": "system", "content": system_context})
        messages.append({"role": "user", "content": user_prompt})

        # --- 2. Single Request to Agent ---
        logger.info("Phase 2: Streaming request to agent server")
        yield {"type": "info", "content": "Sending request to agent..."}

        request_data = {
            "messages": messages,
            "tools": tool_schemas,
            "temperature": 0.3,  # You can adjust this as needed
        }

        assistant_response_for_memory = ""
        try:
            async with self.http_client.stream(
                    "POST", f"{self.api_base_url}/v1/agent/chat", json=request_data, timeout=300
            ) as response:
                response.raise_for_status()
                # Stream the response chunk by chunk as the server thinks and acts
                async for chunk in response.aiter_text():
                    yield {"type": "token", "content": chunk}
                    assistant_response_for_memory += chunk

        except Exception as e:
            error_message = f"API call failed: {e}"
            logger.error(f"API call failed during stream: {e}", exc_info=True)
            yield {"type": "error", "content": error_message}
            return

        # --- 3. Finalization ---
        logger.info("Agent stream finished. Storing conversation history.")
        await self._store_conversation_turn(user_prompt, assistant_response_for_memory)
        yield {"type": "info", "content": "Done."}

    async def close(self):
        """Closes the underlying HTTP client session."""
        logger.info("Closing Anton client http session.")
        await self.http_client.aclose()


async def main():
    """A simple test function to run the client from the command line."""
    print("🚀 Initializing Anton Test Suite...")
    anton_client = Anton()

    # Example prompt
    user_input = "Can you write a python script to list all the files in the current directory and save it as 'list_files.py'?"

    print(f"\n--- Streaming response for: '{user_input}' ---\n")
    try:
        async for chunk in anton_client.stream_response(user_prompt=user_input):
            # The server's pre-formatted output is in the 'content' field
            content = chunk.get("content", "")

            # Print agent's summarized thoughts and actions directly
            if chunk["type"] == "token":
                print(content, end="", flush=True)
            # Print info/error messages on a new line
            elif chunk["type"] in ["info", "error"]:
                print(f"\n[{chunk['type'].upper()}]: {content}")

    finally:
        await anton_client.close()
        print("\n\n--- Test complete ---")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Silence the overly verbose httpx logger
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main())