# api_client.py

"""
Handles all direct HTTP communications with the Anton agent server.
This class encapsulates the logic for making API calls, handling responses,
and managing the HTTP client session.
"""

import logging
from typing import AsyncIterator, Dict, List, Any

import httpx

# Set up a dedicated logger for this module
logger = logging.getLogger(__name__)


class ApiClient:
    """A client for making requests to the Anton agent API."""

    def __init__(self, api_base_url: str, timeout: float):
        """
        Initializes the ApiClient.

        Args:
            api_base_url: The base URL of the API server.
            timeout: The timeout for HTTP requests.
        """
        self.api_base_url = api_base_url
        self.http_client = httpx.AsyncClient(timeout=timeout)
        logger.info(f"API client initialized for server at {api_base_url}")

    async def retrieve_memories(self, query: str) -> List[str]:
        logger.info("Requesting relevant memories from server...")
        try:
            response = await self.http_client.post(
                f"{self.api_base_url}/v1/memory/retrieve",
                json={"query": query}
            )
            response.raise_for_status()
            memories = response.json().get("memories", [])
            logger.info(f"Successfully retrieved {len(memories)} memories.")
            return memories
        except httpx.RequestError as e:
            logger.error(f"HTTP error while retrieving memories: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while retrieving memories: {e}")
        return []

    async def store_conversation(self, user_prompt: str, assistant_response: str):
        if not assistant_response.strip():
            logger.warning("Assistant response is empty. Skipping memory storage.")
            return

        logger.info("Sending conversation turn for memory evaluation and storage.")
        try:
            await self.http_client.post(
                f"{self.api_base_url}/v1/memory/evaluate_and_store",
                json={"user_prompt": user_prompt, "assistant_response": assistant_response}
            )
            logger.info("Memory evaluation and storage request sent successfully.")
        except httpx.RequestError as e:
            logger.error(f"HTTP error while storing conversation: {e}")
        except Exception as e:
            logger.error(f"Failed to send turn for evaluation: {e}")

    async def stream_agent_chat(self, request_data: Dict[str, Any]) -> AsyncIterator[str]:
        logger.info("Streaming request to agent chat endpoint.")
        async with self.http_client.stream(
                "POST", f"{self.api_base_url}/v1/agent/chat", json=request_data
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_text():
                yield chunk

    async def close(self):
        """Closes the underlying HTTP client session."""
        if not self.http_client.is_closed:
            logger.info("Closing API client http session.")
            await self.http_client.aclose()
