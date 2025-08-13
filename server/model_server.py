import asyncio
import logging
import os
import time
import psutil
from pydantic import BaseModel
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import ollama
from server.auth_db import create_user, init_db, verify_user
from server.helpers import AgentChatRequest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
MD_DEBUG = os.getenv("ANTON_MD_DEBUG", "0") == "1"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# ----------------------------

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
MAX_SEQ_LENGTH = 8192 # This is less critical here as Ollama manages context internally

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
    logger.info("🚀 Server starting up...")

    logger.info("--- OLLAMA MODEL CHECK METRICS ---")

    pre_load_usage = get_all_resource_usage(logger)

    startup_start_time = time.monotonic()
    startup_duration = time.monotonic() - startup_start_time

    post_load_usage = get_all_resource_usage(logger)

    # Simplified resource logging as GPU details are not directly accessible via pynvml
    logger.info(
        f"[Resources] Pre-Load  - CPU: {pre_load_usage['cpu_percent']:.1f}%, RAM: {pre_load_usage['ram_percent']:.1f}%"
    )
    logger.info(
        f"[Resources] Post-Load - CPU: {post_load_usage['cpu_percent']:.1f}%, RAM: {post_load_usage['ram_percent']:.1f}%"
    )

    cpu_diff = post_load_usage['cpu_percent'] - pre_load_usage['cpu_percent']
    ram_diff = post_load_usage['ram_percent'] - pre_load_usage['ram_percent']

    logger.info(
        f"[Resources] Difference- CPU: {cpu_diff:+.1f}%, RAM: {ram_diff:+.1f}%"
    )

    logger.info(f"[Latency] Ollama model check complete in {startup_duration:.2f} seconds.")
    logger.info("-----------------------------")

    logger.info("✅ Server is fully initialized and ready to accept requests.")
    yield
    logger.info("🌙 Server shutting down.")
    logger.info("Server shutdown complete.")


app = FastAPI(title="Ollama API Server", version="1.0.0", lifespan=lifespan)


@app.post("/v1/chat/stream")
async def chat_completions_stream(request: AgentChatRequest):
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    # Instantiate the asynchronous client once
    client = ollama.AsyncClient(host=OLLAMA_HOST)
    logger.info("Received request on /v1/chat/stream")

    ollama_options = {
        "temperature": request.temperature,
        "num_predict": 16384,
        "repeat_penalty": 1.1,  # Penalize repetition (1.0 = no penalty, >1.0 = penalty)
        "repeat_last_n": 64,    # Look at last 64 tokens for repetition
        "top_k": 40,           # Limit to top 40 tokens for sampling
        "top_p": 0.9,          # Nucleus sampling with 90% probability mass
        "frequency_penalty": 0.1,  # Additional frequency-based penalty
        "presence_penalty": 0.1,   # Penalize tokens that have appeared
    }

    model_to_use = request.model
    logger.info(f"Query: \n{request.messages}")

    # Step 2: Use the determined model for the actual chat
    actual_ollama_stream_generator = await client.chat(
        model=model_to_use,
        messages=request.messages,
        stream=True,
        options=ollama_options,
    )

    async def ollama_json_streamer(generator):
        async for chunk in generator:
            json_string = chunk.model_dump_json()
            yield f"{json_string}\n" 
            await asyncio.sleep(0)


    # Pass the raw generator into our new streaming function.
    streaming_generator = ollama_json_streamer(actual_ollama_stream_generator)

    # Use the correct media type for a stream of JSON objects.
    return StreamingResponse(streaming_generator, media_type="application/x-ndjson")

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