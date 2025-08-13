import sys, asyncio
from client.anton_client import AntonClient

async def main():
    anton = AntonClient()
    user_prompt = "output some python print statement using markdown"

    async for tok in anton.stream_response(
        user_prompt=user_prompt,
        chat_history=[{"role": "user", "content": user_prompt}],
    ):
        t = tok.get("content")
        if t is None:
            continue
        # CRITICAL: do not add your own newline
        sys.stdout.write(t)
        sys.stdout.flush()

    # final newline so shell prompt isn't glued
    print()

if __name__ == "__main__":
    asyncio.run(main())
