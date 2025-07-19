import asyncio
import logging
import json
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple

import httpx

from tools.tool_defs import ALL_TOOLS
# The ConversationalMemorySystem remains on the client side
from utils.rag_system import EMBEDDING_MODEL, ConversationalMemorySystem

logger = logging.getLogger(__name__)


# --- Main Agent Class (Client) ---
class Anton:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.http_client = httpx.AsyncClient(timeout=None)  # Use a single client instance
        logger.info(f"Anton client initialized, pointing to server at {api_base_url}")

    async def stream_response(
            self, user_prompt: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[str]:
        print(f"Retrieving memories for query: '{user_prompt}'")
        retrieved_context = ''#self.memory_system.retrieve(user_prompt)
        print(f"Retrieved memories:\n---\n{retrieved_context}\n---")

        tool_check_prompt = (
            f"You have access to the following tools:\n{ALL_TOOLS}\n\n"
            f"Based on the user's question, do you need to use one of these tools? "
            f"If yes, respond in the affirmative, telling the user you plan on working on this, then finish with a new line and the single word: TOOLS. Otherwise, answer the user's question directly."
            f"\n\nUser's Question: {user_prompt}"
        )
        initial_messages = chat_history + [{"role": "user", "content": tool_check_prompt}]

        request_data = {
            "messages": initial_messages,
            "stream": True
        }

        final_reply = ''
        agent = False
        try:
            async with self.http_client.stream(
                    "POST", f"{self.api_base_url}/v1/chat/stream", json=request_data, timeout=300
            ) as response:
                async for chunk in response.aiter_text():
                    final_reply += chunk
                    if chunk == 'TOOLS':
                        agent = True
                        break
                    else:
                        yield chunk
        except Exception as e:
            print('Failed to post to server ' + str(e))

        if agent or 'TOOLS' in final_reply:
            request_data = {
                "input": user_prompt,
                "messages": chat_history + [{"role": "user", "content": user_prompt}],
                "stream": True
            }
            try:
                async with self.http_client.stream(
                        "POST", f"{self.api_base_url}/v1/agent/chat", json=request_data, timeout=300
                ) as agent_response:
                    async for agent_chunk in agent_response.aiter_text():
                        yield agent_chunk
            except Exception as e:
                print(e)

    async def close(self):
        """Closes the HTTP client."""
        await self.http_client.aclose()


async def main():
    """Main function to run a series of tests on the Anton client."""
    print("🚀 Initializing Anton Test Suite...")
    anton_client = Anton()
    async for chunk in anton_client.stream_response(user_prompt="Who are you?"):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    # To run the tests, execute this file: python client/anton.py
    # Make sure the model_server.py is running in another terminal first!
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())