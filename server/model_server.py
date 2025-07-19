import logging
import os
import torch
import uvicorn
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from server.run_agent import execute_agent
from server.stream_base_model import stream_base_model
# --- Configuration ---
os.environ["HF_HOME"] = "./hf_cache"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAIN_LLM_MODEL_ID = 'Qwen/Qwen3-14B'
SUMMARIZER_MODEL_ID = 'sshleifer/distilbart-cnn-12-6'
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# --- Global State ---
models = {}


# --- Pydantic Models ---
class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class AgentChatRequest(BaseModel):
    messages: List[OpenAIChatMessage]
    temperature: Optional[float] = 0.5


class OpenAIChatCompletionRequest(BaseModel):
    messages: List[OpenAIChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.3
    stream: Optional[bool] = False
    think: Optional[bool] = False


# --- Model Loading ---
def load_main_llm():
    logger.info(f"Loading main LLM: {MAIN_LLM_MODEL_ID} with 4-bit quantization...")
    if not torch.cuda.is_available():
        logger.error("❌ No CUDA devices found. This model requires a GPU.")
        raise RuntimeError("CUDA is not available, but is required for this model.")
    logger.info(f"✅ Found {torch.cuda.device_count()} CUDA device(s).")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(MAIN_LLM_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MAIN_LLM_MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        # summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_ID)
        # summarizer_model = AutoModelForCausalLM.from_pretrained(SUMMARIZER_MODEL_ID)
        # models['summarizer_tokenizer'] = summarizer_tokenizer
        # models['summarizer_model'] = summarizer_model
        models['main_llm_tokenizer'] = tokenizer
        models['main_llm_model'] = model
        logger.info("✅ Main LLM loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load main LLM: {e}")
        raise


# --- FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting up...")
    load_main_llm()
    logger.info("✅ Server is ready and listening.")
    yield
    logger.info("Server shutting down.")
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- FastAPI App Initialization ---
app = FastAPI(title="Simple Agent Server", lifespan=lifespan)


# --- API Endpoints ---
@app.post("/v1/chat/stream")
async def chat_completions_stream(request: OpenAIChatCompletionRequest):
    model = models.get('main_llm_model')
    tokenizer = models.get('main_llm_tokenizer')
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Main LLM not initialized")

    return stream_base_model(tokenizer, request, model, )


@app.post("/v1/agent/chat")
async def agent_chat(request: AgentChatRequest):
    """Agent chat endpoint supporting chained tool calls."""
    model = models.get('main_llm_model')
    tokenizer = models.get('main_llm_tokenizer')
    return StreamingResponse(
        execute_agent(request, logger, model, tokenizer),
        media_type="text/plain"  # Or "text/event-stream" for server-sent events
    )


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)