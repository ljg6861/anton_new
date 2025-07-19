import asyncio
from threading import Thread

from starlette.responses import StreamingResponse
from transformers import TextIteratorStreamer


def stream_base_model(tokenizer, request, model):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    messages_dict = [msg.model_dump() for msg in request.messages]
    prompt = tokenizer.apply_chat_template(
        messages_dict, tokenize=False, add_generation_prompt=True, enable_thinking=request.think,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_kwargs = dict(
        **inputs, streamer=streamer, max_new_tokens=request.max_tokens or 1024,
        temperature=request.temperature or 0.5, do_sample=(request.temperature or 0.5) > 0,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    async def stream_generator():
        for text in streamer:
            yield text
            await asyncio.sleep(0)

    return StreamingResponse(stream_generator(), media_type="text/plain")