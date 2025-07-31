import asyncio
from threading import Thread

import torch
from transformers import TextIteratorStreamer

def threaded_generation(model, kwargs):
    """
    Wraps model.generate in torch.no_grad() to prevent VRAM leaks during inference.
    """
    with torch.no_grad():
        model.generate(**kwargs)

async def stream_base_model(tokenizer, request, model):
    """
    Handles streaming responses from the base model.
    This function now correctly returns an async generator.
    """
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    messages_dict = [msg.model_dump() for msg in request.messages]

    input_ids = tokenizer.apply_chat_template(
        messages_dict,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=(request.temperature or 0.6) > 0,
    )

    # Use the helper function as the thread target
    thread = Thread(target=threaded_generation, kwargs={'model': model, 'kwargs': generation_kwargs})
    thread.start()

    # This is the async generator that the metrics wrapper can iterate over.
    async def stream_generator():
        for text in streamer:
            yield text
            await asyncio.sleep(0) # Allows other async tasks to run

    return stream_generator()
