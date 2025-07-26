import logging
import os
from typing import List, Optional, Dict, Any
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, )

from client.context_builder import ContextBuilder
from server.agent.agent_loop import run_agent_loop
from server.helpers import evaluate_turn_for_memorability
from server.intent_router import classify_intent
from server.stream_base_model import stream_base_model
from server.tools.tool_defs import STATIC_TOOLS
from server.tools.tool_manager import tool_manager
from utils.memory_manager import MemoryManager

# --- Configuration ---
os.environ["HF_HOME"] = "./hf_cache"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAIN_LLM_MODEL_ID = 'unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit'
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
    temperature: Optional[float] = 0.5
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

    print("--- Registering Static Tools ---")
    for tool in STATIC_TOOLS:
        tool_manager.register(tool)
    print("--- Static Tool Registration Complete ---")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info("Using 4-bit quantization with BitsAndBytes.")
    logger.debug(f"Quantization config: {quantization_config}")

    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MAIN_LLM_MODEL_ID)
        models['main_llm_tokenizer'] = tokenizer
        logger.info("✅ Tokenizer loaded.")

        logger.info("Loading model... (This may take a moment)")
        model = AutoModelForCausalLM.from_pretrained(
            MAIN_LLM_MODEL_ID,
            quantization_config=quantization_config,
            max_memory={
                0 : "12GB",
                1 : "11GB",
                "cpu" : "28GB"
            },
            device_map="auto",
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


# --- API Endpoints ---
@app.post("/v1/memory/retrieve")
async def retrieve_memory(request: MemoryQueryRequest):
    logger.info(f"Received request on /v1/memory/retrieve for query: '{request.query}'")
    if not memory_manager:
        logger.error("MemoryManager not initialized, cannot retrieve memory.")
        raise HTTPException(status_code=503, detail="MemoryManager not initialized")
    try:
        results = memory_manager.query_memories(request.query, request.k)
        logger.info(f"Retrieved {len(results)} memories for query.")
        logger.debug(f"Retrieved memories: {results}")
        return {"memories": results}
    except Exception as e:
        logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve memories.")

@app.post("/v1/chat/stream")
async def chat_completions_stream(request: OpenAIChatCompletionRequest):
    logger.info("Received request on /v1/chat/stream")
    logger.debug(f"Request details: {request.model_dump_json(indent=2)}")
    model = models.get('main_llm_model')
    tokenizer = models.get('main_llm_tokenizer')
    if not model or not tokenizer:
        logger.error("Main LLM not available for streaming chat request.")
        raise HTTPException(status_code=503, detail="Main LLM not initialized")

    return stream_base_model(tokenizer, request, model)


@app.post("/v1/agent/chat")
async def agent_chat(request: AgentChatRequest):
    model = models.get('main_llm_model')
    tokenizer = models.get('main_llm_tokenizer')
    last_user_message = request.messages[-1].content

    logger.info("Handling as task_execution. Initiating agent loop.")

    system_prompt = await ContextBuilder().build_system_prompt(last_user_message)
    tool_schemas = tool_manager.get_tool_schemas()

    full_user_prompt = system_prompt + last_user_message
    request.messages[-1].content = full_user_prompt
    request.tools = tool_schemas

    # Run the original agent loop
    return StreamingResponse(
        run_agent_loop(request, logger, model, tokenizer),
        media_type="text/plain"
    )


@app.post("/v1/memory/evaluate_and_store")
async def evaluate_and_store_turn(request: StoreTurnRequest):
    logger.info("Received request on /v1/memory/evaluate_and_store")
    model = models.get('main_llm_model')
    tokenizer = models.get('main_llm_tokenizer')
    if not model or not tokenizer:
        logger.error("Main LLM not available for memory evaluation.")
        raise HTTPException(status_code=503, detail="Main LLM not initialized")

    logger.info("Evaluating conversation turn for memorability...")
    evaluation = await evaluate_turn_for_memorability(
        request.user_prompt,
        request.assistant_response,
        model,
        tokenizer
    )
    logger.info(f"Memorability evaluation result: {evaluation}")

    if evaluation and evaluation.get("is_memorable"):
        memory_content = evaluation.get("memory_content")
        if memory_content:
            logger.info("Turn deemed memorable. Storing content in vector memory.")
            logger.debug(f"Content to be stored: '{memory_content}'")
            memory_manager.store_memory(memory_text=memory_content)
            return {"status": "stored", "detail": memory_content}
        else:
            logger.warning("Turn was marked memorable but no content was extracted.")
            return {"status": "not_stored", "detail": "Marked memorable, but content was empty."}
    else:
        logger.info("Turn not deemed significant enough for long-term memory.")
        return {"status": "not_stored", "detail": "Turn was not deemed significant enough for long-term memory."}


if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on {SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)