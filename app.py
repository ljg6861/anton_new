from typing import List, Dict

import chainlit as cl

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
    # Initialize answer_msg to None. We will create it on the fly.
    answer_msg = None

    # By creating the Step first, it will appear at the top.
    async with cl.Step(name="Thinking", parent_id=message.id) as step:
        try:
            async for chunk in anton.stream_response(
                user_prompt=message.content, chat_history=chat_history
            ):
                if chunk["type"] == "thought":
                    await step.stream_token(f"• {chunk['content']}\n")

                elif chunk["type"] == "tool_result":
                    result_str = (
                        f"\n*Tool Result:*\n```json\n{chunk['content']}\n```\n"
                    )
                    await step.stream_token(result_str)

                elif chunk["type"] == "token":
                    # If this is the first token, create the message object.
                    # This ensures it appears *after* the "Thinking" step.
                    if answer_msg is None:
                        answer_msg = cl.Message(
                            content="", author="Anton", parent_id=message.id
                        )
                    
                    await answer_msg.stream_token(chunk["content"])
                    final_answer += chunk["content"]

        finally:
            # Update the message only if it was created.
            if answer_msg and answer_msg.content:
                await answer_msg.update()

    # Update the session's chat history.
    chat_history.append({"role": "user", "content": message.content})
    chat_history.append({"role": "assistant", "content": final_answer})
    cl.user_session.set("chat_history", chat_history)