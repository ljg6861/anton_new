import html
import re
from typing import List, Dict

import chainlit as cl
import os
import httpx
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
MD_DEBUG = os.getenv("ANTON_MD_DEBUG", "0") == "1"

from client.anton_client import AntonClient

@cl.on_chat_start
async def on_chat_start():
    """Initializes the assistant and its memory when a new chat session starts."""
    cl.user_session.set("anton", AntonClient())
    cl.user_session.set("chat_history", [])


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming messages, streams thoughts and the final response,
    and learns from the exchange.
    """
    anton: AntonClient = cl.user_session.get("anton")
    chat_history: List[Dict] = cl.user_session.get("chat_history")

    final_answer = ""
    answer_msg = None

    # We keep the async with block to maintain the single, shimmering step
    async with cl.Step(name="Working...", parent_id=message.id, type="run") as step:
        async for chunk in anton.stream_response(
            user_prompt=message.content, chat_history=chat_history
        ):
            if chunk["type"] == "step":
                # A new phase is starting. Update the name of our single step.
                new_name = chunk.get("content")
                if new_name:
                    step.name = new_name
                    # INSTEAD of clearing output, we stream a separator for clarity
                    await step.stream_token("\n\n---\n")
                    await step.update() # Push the name change to the UI

            elif chunk["type"] == "step_content":
                # Stream content to the step. It will now append.
                await step.stream_token(chunk["content"])

            elif chunk["type"] == "tool_result":
                # Stream tool results.
                result_str = (
                    f"\n*Tool Result:*\n```json\n{chunk['content']}\n```\n"
                )
                await step.stream_token(result_str)

            elif chunk["type"] == "token":
                # Handle the final answer stream as usual.
                final_answer += chunk["content"]
                token = chunk["content"]
                if answer_msg is None:
                    answer_msg = cl.Message(content="", author="Anton", parent_id=message.id)
                await answer_msg.stream_token(token)

            elif chunk["type"] == "info":
                continue

    if answer_msg:
        clean_content = re.sub(r"<tool_call>.*?</tool_call>", "", html.unescape(answer_msg.content), flags=re.DOTALL)
        answer_msg.content = clean_content.strip()
        await answer_msg.update()

    # Update the session's chat history.
    chat_history.append({"role": "user", "content": message.content})
    chat_history.append({"role": "assistant", "content": final_answer})
    cl.user_session.set("chat_history", chat_history)



@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(
                f"http://localhost:8000/v1/auth",
                json={"username": username, "password": password},
            )
        if r.status_code == 200:
            data = r.json()
            return cl.User(identifier=data["identifier"], metadata=data.get("metadata", {}))
        return None
    except Exception:
        return None
    
    
