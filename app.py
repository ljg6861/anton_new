import html
from typing import List, Dict

import chainlit as cl
import httpx

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
    answer_msg = None  # create on first token so the Thinking step appears first
    # Buffer to accumulate the content of a single thought.
    thought_buffer = ""

    # By creating the Step first, it will appear at the top.
    async with cl.Step(name="Thinking", parent_id=message.id) as step:
        try:
            async for chunk in anton.stream_response(
                user_prompt=message.content, chat_history=chat_history
            ):
                # If the chunk is part of a thought, add it to the buffer.
                if chunk["type"] == "thought":
                    thought_buffer += chunk["content"]
                # If a new chunk type arrives, the previous thought is complete.
                # Flush the buffered thought before processing the new chunk.
                else:
                    if thought_buffer:
                        await step.stream_token(f"• {thought_buffer}\n")
                        thought_buffer = ""  # Reset buffer for the next thought

                    # Process the non-thought chunk.
                    if chunk["type"] == "tool_result":
                        result_str = (
                            f"\n*Tool Result:*\n```json\n{chunk['content']}\n```\n"
                        )
                        await step.stream_token(result_str)

                    elif chunk["type"] == "token":
                        token = html.unescape(chunk["content"])  # keep markdown; html entities only
                        if answer_msg is None:
                            answer_msg = cl.Message(content="", author="Anton", parent_id=message.id)
                            await answer_msg.send()
                        await answer_msg.stream_token(token)
                        final_answer += token
            
            # After the loop, flush any final thought that might be in the buffer.
            if thought_buffer:
                await step.stream_token(f"• {thought_buffer}\n")

        finally:
            # Ensure the final message content is committed
            if answer_msg:
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
    
    
