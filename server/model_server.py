import logging
import os


os.environ["HF_HOME"] = "D:/huggingface"
from typing import List, Optional, Dict, Any
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM, )
from server.stream_base_model import stream_base_model
from utils.memory_manager import MemoryManager

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAIN_LLM_MODEL_ID = 'unsloth/Qwen3-14B-unsloth-bnb-4bit'
SUMMARY_MODEL_ID = 'Qwen/Qwen3-0.6B'
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# --- Global State ---
models = {}
memory_manager = None


# --- Pydantic Models ---
class OpenAIChatMessage(BaseModel):
    role: str
    content: str

class StoreTurnRequest(BaseModel):
    user_prompt: str
    assistant_response: str

class MemoryQueryRequest(BaseModel):
    query: str
    k: int = 3

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition

class AgentChatRequest(BaseModel):
    messages: List[OpenAIChatMessage]
    tools: Optional[List[ToolDefinition]] = None
    temperature: Optional[float] = 0.6
    think: Optional[bool] = True

class OpenAIChatCompletionRequest(BaseModel):
    messages: List[OpenAIChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.3
    stream: Optional[bool] = False
    think: Optional[bool] = False


# --- Model Loading ---
def load_main_llm():
    global memory_manager
    logger.info(f"Attempting to load main LLM: {MAIN_LLM_MODEL_ID}")
    if not torch.cuda.is_available():
        logger.error("❌ No CUDA devices found. This model requires a GPU.")
        raise RuntimeError("CUDA is not available, but is required for this model.")
    logger.info(f"✅ Found {torch.cuda.device_count()} CUDA device(s).")
    logger.info("Using 4-bit quantization with BitsAndBytes.")

    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MAIN_LLM_MODEL_ID)
        models['main_llm_tokenizer'] = tokenizer
        logger.info("✅ Tokenizer loaded.")
        logger.info("Loading model... (This may take a moment)")
        model = AutoModelForCausalLM.from_pretrained(
            MAIN_LLM_MODEL_ID,
            max_memory={
                0 : "13GB",
                1 : "11GB",
                "cpu" : "28GB"
            },
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
        models['main_llm_model'] = model
        logger.info(f"✅ Model loaded and mapped to device(s): {model.device}")

        logger.info("Initializing MemoryManager...")
        memory_manager = MemoryManager()
        logger.info("✅ MemoryManager initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load main LLM: {e}", exc_info=True)
        raise


# --- FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Server starting up...")
    load_main_llm()
    logger.info("✅ Server is fully initialized and ready to accept requests.")
    yield
    logger.info("🌙 Server shutting down.")
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Server shutdown complete.")


# --- FastAPI App Initialization ---
app = FastAPI(title="Anton Agent Server", version="1.0.0", lifespan=lifespan)

@app.post("/v1/chat/stream")
async def chat_completions_stream(request: AgentChatRequest):
    logger.info("Received request on /v1/chat/stream")
    logger.debug(f"Request details: {request.model_dump_json(indent=2)}")
    model = models.get('main_llm_model')
    tokenizer = models.get('main_llm_tokenizer')
    if not model or not tokenizer:
        logger.error("Main LLM not available for streaming chat request.")
        raise HTTPException(status_code=503, detail="Main LLM not initialized")

    return stream_base_model(tokenizer, request, model)


if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on {SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)