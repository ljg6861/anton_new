# agent/streaming.py

"""
Contains the custom TextStreamer and the asynchronous generator for streaming model responses.
"""

import asyncio
from queue import Queue
from threading import Thread
from typing import AsyncGenerator

import torch
from transformers import TextStreamer, PreTrainedModel, PreTrainedTokenizer

class QueueTextStreamer(TextStreamer):
    """A custom TextStreamer that puts generated tokens into a thread-safe queue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_queue = Queue()

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Puts the text onto the queue and a sentinel value at the end."""
        self.token_queue.put(text)
        if stream_end:
            self.token_queue.put(None)


async def stream_model_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict],
        gen_kwargs: dict,
    think: bool = True,
) -> AsyncGenerator[str, None]:
    """
    An async generator that streams model outputs token-by-token.

    It runs the synchronous `model.generate` in a separate thread to avoid
    blocking the asyncio event loop and yields tokens as they become available.
    """
    streamer = QueueTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=think
    ).to(model.device)

    generation_args = {
        "input_ids": input_ids,
        "streamer": streamer,
        **gen_kwargs,
    }

    # Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_args)
    thread.start()

    loop = asyncio.get_running_loop()
    while True:
        # Asynchronously get tokens from the synchronous queue
        token = await loop.run_in_executor(None, streamer.token_queue.get)
        if token is None:  # Sentinel for end of stream
            break
        yield token

    # Clean up the thread
    await loop.run_in_executor(None, thread.join)