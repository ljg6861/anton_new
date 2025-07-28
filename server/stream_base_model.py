import asyncio
from threading import Thread

import torch
from starlette.responses import StreamingResponse
from transformers import TextIteratorStreamer

def threaded_generation(model, kwargs):
    """
    Wraps model.generate in torch.no_grad() to prevent VRAM leaks during inference.
    """
    with torch.no_grad():
        model.generate(**kwargs)

def stream_base_model(tokenizer, request, model):
    """
    Handles streaming responses from the base model, now with VRAM leak protection.
    """
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    messages_dict = [msg.model_dump() for msg in request.messages]

    prompt = tokenizer.apply_chat_template(
        messages_dict, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=2048,
        temperature=request.temperature or 0.6,
        do_sample=(request.temperature or 0.6) > 0,
    )

    # Use the new helper function as the thread target
    thread = Thread(target=threaded_generation, kwargs={'model': model, 'kwargs': generation_kwargs})
    thread.start()

    async def stream_generator():
        for text in streamer:
            yield text
            await asyncio.sleep(0)

    return StreamingResponse(stream_generator(), media_type="text/plain")