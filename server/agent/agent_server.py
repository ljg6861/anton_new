import logging
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from starlette.responses import StreamingResponse

from server.agent.agent_loop import run_agent_loop # Import your loop
from server.agent.tools.tool_defs import STATIC_TOOLS
from server.agent.tools.tool_manager import tool_manager

# --- Configuration ---
AGENT_SERVER_HOST = "0.0.0.0"
AGENT_SERVER_PORT = 8001 # Run on a different port
MODEL_SERVER_URL = "http://localhost:8000" # URL of your main model server

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("--- Registering Static Tools ---")
for tool in STATIC_TOOLS:
    tool_manager.register(tool)
print("--- Static Tool Registration Complete ---")

# --- Pydantic Models (Copy from your main server file) ---
class OpenAIChatMessage(BaseModel):
    role: str
    content: str

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

# --- FastAPI App ---
app = FastAPI(title="Agent Logic Server")

@app.post("/v1/agent/chat")
async def agent_chat(request: AgentChatRequest):
    logger.info("Agent Server received request.")

    return StreamingResponse(
        run_agent_loop(request, logger, MODEL_SERVER_URL),
        media_type="text/plain"
    )

if __name__ == "__main__":
    logger.info(f"Starting Agent Server on {AGENT_SERVER_HOST}:{AGENT_SERVER_PORT}")
    uvicorn.run(app, host=AGENT_SERVER_HOST, port=AGENT_SERVER_PORT)