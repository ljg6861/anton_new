import asyncio
import json
import logging
import os
import time
import psutil
from pydantic import BaseModel
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from openai import AsyncOpenAI

from server.auth_db import create_user, init_db, verify_user
from server.helpers import AgentChatRequest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
MD_DEBUG = os.getenv("ANTON_MD_DEBUG", "0") == "1"

# vLLM Configuration
VLLM_PORT = os.getenv("VLLM_PORT", "8003")
VLLM_HOST = os.getenv("VLLM_HOST", f"http://localhost:{VLLM_PORT}")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "anton-vllm-key")
DEFAULT_MODEL = os.getenv("VLLM_MODEL", "qwen-coder-32b")

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8002
MAX_SEQ_LENGTH = 32768

def get_all_resource_usage(logger_instance) -> dict:
    """
    Captures a snapshot of current CPU, RAM usage.
    GPU monitoring via pynvml is removed as Ollama abstracts this.
    If you need GPU stats, consider `nvidia-smi` via subprocess.
    """
    usage = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "gpus": [] # Placeholder, as we don't have direct pynvml access here
    }
    # You could add subprocess calls to `nvidia-smi` here if detailed GPU stats are critical,
    # but be aware of the performance overhead for frequent calls.
    return usage

# --- FastAPI Lifespan (Startup/Shutdown Events) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Server starting up with vLLM backend...")
    
    startup_start_time = time.monotonic()
    pre_load_usage = get_all_resource_usage(logger)
    
    # Initialize vLLM client
    app.state.client = AsyncOpenAI(
        base_url=f"{VLLM_HOST}/v1",
        api_key=VLLM_API_KEY
    )
    logger.info(f"vLLM client initialized: {VLLM_HOST}")
    
    startup_duration = time.monotonic() - startup_start_time
    post_load_usage = get_all_resource_usage(logger)
    
    logger.info(f"[Resources] Pre-Load  - CPU: {pre_load_usage['cpu_percent']:.1f}%, RAM: {pre_load_usage['ram_percent']:.1f}%")
    logger.info(f"[Resources] Post-Load - CPU: {post_load_usage['cpu_percent']:.1f}%, RAM: {post_load_usage['ram_percent']:.1f}%")
    logger.info(f"[Latency] Backend initialization complete in {startup_duration:.2f} seconds.")
    logger.info("✅ Server is fully initialized and ready to accept requests.")
    
    yield
    
    logger.info("🌙 Server shutting down.")


app = FastAPI(title="vLLM API Server", version="1.0.0", lifespan=lifespan)


@app.post("/v1/chat/stream")
async def chat_completions_stream(request: AgentChatRequest):
    logger.info("Received request on /v1/chat/stream")
    
    client = app.state.client
    model_to_use = request.model or DEFAULT_MODEL
    
    # Prepare tools if available
    tools = getattr(request, 'tools', None)
    tool_choice = getattr(request, 'tool_choice', 'auto') if tools else None
    
    logger.info(f"Query: \n{request.messages} for model: {model_to_use}")
    
    try:
        stream = await client.chat.completions.create(
            model=model_to_use,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=16384,
            stream=True,
            tools=tools,
            tool_choice=tool_choice
        )
        
        async def vllm_json_streamer():
            async for chunk in stream:
                # Convert OpenAI chunk to Anton's expected format
                chunk_dict = {
                    "message": {
                        "content": "",
                        "role": "assistant"
                    },
                    "done": False
                }
                
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    
                    if delta.content:
                        chunk_dict["message"]["content"] = delta.content
                    
                    if delta.tool_calls:
                        chunk_dict["message"]["tool_calls"] = []
                        for tool_call in delta.tool_calls:
                            tool_call_dict = {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                            chunk_dict["message"]["tool_calls"].append(tool_call_dict)
                    
                    if chunk.choices[0].finish_reason:
                        chunk_dict["done"] = True
                
                json_string = json.dumps(chunk_dict)
                yield f"{json_string}\n"
                await asyncio.sleep(0)
        
        return StreamingResponse(vllm_json_streamer(), media_type="application/x-ndjson")
        
    except Exception as e:
        logger.error(f"vLLM streaming error: {e}")
        raise HTTPException(status_code=500, detail=f"vLLM error: {str(e)}")

router = APIRouter(prefix="/v1", tags=["auth"])
init_db()  # ensure schema exists

class AuthRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    identifier: str
    metadata: dict

@app.post("/v1/auth", response_model=AuthResponse)
async def auth(req: AuthRequest):
    ok = verify_user(req.username, req.password)
    if not ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    # You can enrich metadata here (role, plan, etc.)
    return AuthResponse(
        identifier=req.username,
        metadata={"role": "user", "provider": "credentials"}
    )

class SignupRequest(BaseModel):
    username: str
    password: str

@app.post("/v1/users", status_code=status.HTTP_201_CREATED)
async def signup(req: SignupRequest):
    try:
        create_user(req.username, req.password)
        return {"ok": True}
    except ValueError:
        raise HTTPException(status_code=409, detail="Username already exists")


if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on {SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)