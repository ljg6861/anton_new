import asyncio
import subprocess
from typing import List, Dict

import chainlit as cl
from anton import Anton

anton = Anton()


@cl.on_chat_start
async def on_chat_start():
    """Initializes the assistant and its memory when a new chat session starts."""
    cl.user_session.set("anton", anton)
    cl.user_session.set("chat_history", [])
    await cl.Message(
        content="Anton is ready. I will remember the key points of our conversation to provide better answers. Let's begin!"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handles incoming messages, streams the response, and learns from the exchange."""
    chat_history: List[Dict] = cl.user_session.get("chat_history")

    thought_msg = cl.Message(content="", author="Thinking", parent_id=message.id)
    answer_msg = cl.Message(content="", author="Anton", parent_id=message.id)

    async for content in anton.stream_response(user_prompt=message.content, chat_history=chat_history):
            await answer_msg.stream_token(content)

    # Update standard chat history
    chat_history.append({"role": "user", "content": message.content})
    chat_history.append({"role": "assistant", "content": answer_msg.content})
    cl.user_session.set("chat_history", chat_history)
