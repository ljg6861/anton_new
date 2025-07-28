from typing import List, Dict

import chainlit as cl

from client.anton_client import AntonClient

@cl.on_chat_start
async def on_chat_start():
    """Initializes the assistant and its memory when a new chat session starts."""
    cl.user_session.set("anton", AntonClient())  # <-- Initialize our new class
    cl.user_session.set("chat_history", [])


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming messages, streams thoughts and the final response,
    and learns from the exchange.
    """
    anton: AntonClient = cl.user_session.get("anton")
    chat_history: List[Dict] = cl.user_session.get("chat_history")

    # Initialize the UI elements for thoughts and the final answer
    thought_msg = cl.Message(content="", author="Thinking", parent_id=message.id)
    answer_msg = cl.Message(content="", author="Anton", parent_id=message.id)
    await thought_msg.send()  # Send the empty message to get an ID

    final_answer = ""
    try:
        # The new stream yields structured dictionaries
        async for chunk in anton.stream_response(
                user_prompt=message.content, chat_history=chat_history
        ):
            if chunk["type"] == "thought":
                await thought_msg.stream_token(f"• {chunk['content']}\n")

            elif chunk["type"] == "tool_result":
                # Display tool results clearly in the thought process
                result_str = f"\n*Tool Result:*\n```json\n{chunk['content']}\n```\n"
                await thought_msg.stream_token(result_str)

            elif chunk["type"] == "token":
                # Stream final answer tokens to the main answer message
                await answer_msg.stream_token(chunk["content"])
                final_answer += chunk["content"]

    finally:
        await thought_msg.update()

        await answer_msg.update()

        # Update standard chat history
        chat_history.append({"role": "user", "content": message.content})
        chat_history.append({"role": "assistant", "content": final_answer})
        cl.user_session.set("chat_history", chat_history)