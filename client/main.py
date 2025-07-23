# main.py

"""
Main entry point for running the Anton command-line client.
"""

import asyncio
import logging

from anton_client import AntonClient


async def main():
    """A simple test function to run the client from the command line."""
    print("🚀 Initializing Anton Test Suite...")
    client = AntonClient()

    # Example prompt
    user_input = "Can you write a python script to list all the files in the current directory and save it as 'list_files.py'?"

    print(f"\n--- Streaming response for: '{user_input}' ---\n")
    try:
        # The stream_response method now yields dictionaries for structured processing
        async for chunk in client.stream_response(user_prompt=user_input):
            content = chunk.get("content", "")
            chunk_type = chunk.get("type", "token")

            if chunk_type == "token":
                # For tokens, print them continuously on the same line
                print(content, end="", flush=True)
            elif chunk_type in ["info", "error"]:
                # For status updates or errors, print them on a new line for clarity
                print(f"\n[{chunk_type.upper()}]: {content}")

    except Exception as e:
        logging.error(f"An error occurred in the main loop: {e}", exc_info=True)
    finally:
        await client.close()
        print("\n\n--- Test complete ---")


if __name__ == "__main__":
    # Configure logging for the entire application
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Silence the overly verbose httpx logger to keep the output clean
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Run the asynchronous main function
    asyncio.run(main())
