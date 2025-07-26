# client/anton_client.py

import logging
from typing import AsyncIterator, Dict, List, Optional

from client import config
from client.api_client import ApiClient
# We no longer need the ContextBuilder on the client side.

# Set up a dedicated logger for this module
logger = logging.getLogger(__name__)


class AntonClient:
    """
    A simplified client for interacting with the Anton agent.
    It sends the user's request and streams the server's processed response.
    It is no longer responsible for gathering context.
    """

    def __init__(self, api_base_url: str = config.API_BASE_URL, timeout: float = config.DEFAULT_TIMEOUT):
        self.api_client = ApiClient(api_base_url, timeout)
        logger.info("Anton client initialized.")

    async def stream_response(
            self, user_prompt: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[Dict[str, str]]:
        """
        Sends the user prompt and chat history to the server and streams the response.
        The server now handles all context gathering and routing.
        """
        assistant_response_for_memory = ""
        try:
            yield {"type": "info", "content": "Sending request to agent..."}
            logger.info("Phase 1: Sending raw prompt to server for routing.")

            messages = (chat_history or []) + [{"role": "user", "content": user_prompt}]

            # The request_data is now much simpler. The server will add tools as needed.
            request_data = {
                "messages": messages,
                "temperature": config.DEFAULT_TEMPERATURE,
            }

            # --- 2. Streaming from Server ---
            logger.info("Phase 2: Streaming processed response from server.")
            stream = self.api_client.stream_agent_chat(request_data)
            async for chunk in stream:
                yield {"type": "token", "content": chunk}
                assistant_response_for_memory += chunk

        except Exception as e:
            error_message = f"API call failed: {e}"
            logger.error(f"API call failed during stream: {e}", exc_info=True)
            yield {"type": "error", "content": error_message}
            return
        finally:
            # --- 3. Finalization (remains the same) ---
            logger.info("Agent stream finished. Storing conversation history.")
            # Note: The server could also handle this, but leaving it here is fine.
            await self.api_client.store_conversation(user_prompt, assistant_response_for_memory)
            yield {"type": "info", "content": "Done."}

    async def close(self):
        """Closes the underlying API client session."""
        await self.api_client.close()
