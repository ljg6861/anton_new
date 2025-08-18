# Anton AI Assistant - Development Instructions

## Architecture Overview

Anton is a **ReAct (Reason-Act) AI assistant** with a sophisticated multi-service architecture:

### Core Services (started via `./run.sh`)
- **Agent Server** (`server/agent/agent_server.py:8001`) - ReAct logic, tool execution, learning
- **Model Server** (`server/model_server.py:8002`) - Ollama LLM interface 
- **Chainlit UI** (`app.py:7860`) - Chat interface

### Key Architectural Components

**ReAct Agent** (`server/agent/react/`) - Modular three-memory system:
- Working Memory: Recent context within token budget
- Session Memory: Current conversation decisions/TODOs  
- Long-term Memory: RAG retrieval from knowledge packs
- Token budget management prevents context overflow

**Tool Learning System** (`server/agent/tool_learning_store.py`) - **Unique Feature**:
- Records ALL tool executions with outcomes
- Learns failure→success patterns via LLM analysis
- Provides **immediate corrective action** when tools fail with confidence-scored alternatives
- Example: `git clone` fails → suggests `git pull origin main` (95% confidence)

**Knowledge Architecture**:
- `KnowledgeStore` - Session-scoped file access tracking
- `RAGManager` - Long-term retrieval from domain packs in `learning/packs/`
- Domain packs created via `bin/make_pack.sh` from PDFs/code

## Development Workflows

### Local Development
```bash
# Start all services
./run.sh

# Services log to logs/ directory
tail -f logs/agent.log    # ReAct agent
tail -f logs/model.log    # LLM server  
tail -f logs/ui.log       # Chainlit UI
```

### Tool Development
Tools in `server/agent/tools/` auto-discovered by `ToolManager`:
```python
# Tool structure (see git.py for examples)
class ExampleTool:
    function = {
        "type": "function", 
        "function": {
            "name": "tool_name",
            "description": "What it does",
            "parameters": {"type": "object", "properties": {...}}
        }
    }
    def run(self, arguments: dict) -> str:
        # Tool logic - exceptions = failures, no exceptions = success
        return "✅ Success: result" or raise Exception("failure")
```

### Git Integration Pattern
Git tools execute from repo root via `GIT_ROOT_DIR`:
- `git_commit` with `add_all=True` (default) stages everything first
- `create_pull_request` requires GitHub CLI (`gh`)
- All git commands return formatted success/error messages

### Critical: Tool Failure Semantics  
**Only exceptions indicate tool failures** - content analysis was removed:
- Tools returning error information (like log analysis) are **successful**
- Real failures bubble up as exceptions and trigger corrective action
- Tool results prefixed with `TOOL_RESULT from {tool_name}:` to prevent agent confusion

## Project-Specific Patterns

### Client-Server Communication
- `AntonClient` (`client/`) - HTTP streaming to agent server
- Structured streaming: `<thought>`, `<tool_result>` tags parsed by client
- `AgentChatRequest` - Pydantic models in `server/helpers.py`

### Learning Pack Creation
```bash
# From code directory
bin/make_pack.sh --domain myproject --input /path/to/code --pack myproject.v1

# From PDF  
bin/make_pack.sh --domain calculus --input textbook.pdf --pack calc.v1
```

### Database Integration
- Tool learning: SQLite at `tool_learning.db` 
- User auth: PostgreSQL (see `pgdata/`)
- Both auto-initialized on startup

### Configuration
- Environment: `.env` file (see `run.sh` sourcing)
- Client config: `client/config.py` (API_BASE_URL, timeouts)
- Model config: `server/config.py` (model names, URLs)

## Testing Patterns

**CRITICAL: Never create new test, demo, or documentation files.**

All testing must be consolidated into `comprehensive_tests.py`:
- Tests interact with running server directly (assume services are up)
- Use HTTP calls to agent server at `http://localhost:8001`
- Test tool learning system, ReAct flows, and corrective actions
- Verify both success/failure paths and learning integration

Example server interaction pattern:
```python
import httpx
async def test_agent_endpoint():
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8001/v1/agent/chat", 
            json={"messages": [{"role": "user", "content": "test"}]})
```

## Key Files for AI Agents

- `server/agent/react/react_agent.py` - Main coordination logic
- `server/agent/tool_executor.py` - Tool execution with learning
- `server/agent/tool_learning_store.py` - Failure learning system
- `server/agent/tools/` - Tool implementations
- `learning/` - Knowledge pack processing pipeline
- `client/anton_client.py` - HTTP streaming client

The tool learning system is Anton's **differentiating feature** - it learns from failures and provides immediate corrective suggestions, making the assistant self-improving.
