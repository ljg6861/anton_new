# agent/streaming.py

"""
Contains the custom TextStreamer and the asynchronous generator for streaming model responses.
"""

import asyncio
from queue import Queue
from threading import Thread
from typing import AsyncGenerator, Union

import torch
from transformers import (
    TextStreamer,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM, # Use this for summarization models
)

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


async def stream_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Union[str, list[dict]],
    gen_kwargs: dict,
        think: bool = True
) -> AsyncGenerator[str, None]:
    """
    An async generator that streams model outputs token-by-token.
    It can handle both plain string prompts (for summarization) and chat message lists.
    """
    streamer = QueueTextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Prepare inputs for either a plain string or a chat template
    if isinstance(prompt, str):
        # Handle summarization prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    else:
        # Handle chat prompt
        input_ids = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_tensors="pt", enable_thinking = think
        ).to(model.device)


    generation_args = {
        "input_ids": input_ids,
        "streamer": streamer,
        **gen_kwargs,
    }

    # Run generation in a separate thread to avoid blocking the event loop
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