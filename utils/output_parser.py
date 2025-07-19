import re
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# in _initialize_hf_pipeline (after you build your text‑generation pipeline):
summary_pipe = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn",
    max_length=60,
    min_length=10,
)
summarizer = HuggingFacePipeline(pipeline=summary_pipe)

class CleanOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        # 1) pull out the think block
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        reasoning = m.group(1).strip() if m else ""

        # 2) OPTIONAL: summarize it with a lightweight summarizer
        #    (you could hook another HF pipeline here)
        summary = summarizer.invoke(reasoning)

        # 3) remove the raw think tags and append your summary
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if summary:
            cleaned += f"\n\n**Reasoning (brief):** {summary}"
        return cleaned


