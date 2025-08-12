# FILE: server/helpers.py
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from server.config import QWEN_30B_THINKING

logger = logging.getLogger(__name__)


# Represents an individual tool call made by the model.
class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: dict[str, str] # e.g., {"name": "get_weather", "arguments": '{"location": "Boston"}'}


# A more accurate and type-safe representation of a chat message.
class OpenAIChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None # For assistant messages that use tools.
    tool_call_id: str | None = None          # For 'tool' role messages with results.


# Defines the name, description, and JSON schema for a function's parameters.
class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


# Defines a tool, which is currently always a function.
class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


# The main request body for your agent.
class AgentChatRequest(BaseModel):
    messages: list[OpenAIChatMessage]
    tools: list[ToolDefinition] | None = None
    temperature: float = 0.6
    complex: bool = False
    model: str = QWEN_30B_THINKING