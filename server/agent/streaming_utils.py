# agent/streaming_utils.py

"""
Utilities for processing and transforming the model's output stream in real-time.
"""
from typing import AsyncGenerator, List, Any
from server.agent.streaming import stream_response
from .config import SENTENCE_END_REGEX, get_generation_kwargs


def is_paragraph_end(buf: List[str]) -> bool:
    """Heuristic to detect a paragraph break within a <think> block."""
    joined = "".join(buf)
    sentence_count = len(SENTENCE_END_REGEX.findall(joined))
    has_blank_line = "\n\n" in joined
    return sentence_count >= 2 and has_blank_line


async def summarize_thought_chunk(text: str, model: Any, tokenizer: Any) -> AsyncGenerator[str, None]:
    """Summarizes a chunk of text from the agent's thought process."""
    if not text.strip():
        return

    summary_kwargs = get_generation_kwargs(tokenizer, temperature=0.7, max_new_tokens=128)
    prompt = f"Concisely summarize the following thought process:\n\n{text}"

    async for token in stream_response(model, tokenizer, [{"role": "user", "content": prompt}], summary_kwargs):
        yield token