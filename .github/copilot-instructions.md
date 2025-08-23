# Anton AI Assistant - Development Instructions

## Architecture Overview

Anton is a **ReAct (Reason-Act) AI assistant** with a sophisticated multi-service architecture and intelligent agentic routing:

### Core Services (started via `./run.sh`)
- **Agent Server** (`server/agent/agent_server.py:8001`) - ReAct logic, agentic flow, tool execution
- **vLLM Server** (`start_vllm.sh:8003`) - High-performance model inference via vLLM  
- **Chainlit UI** (`app.py:7860`) - Streaming chat interface with real-time thought display

### Agentic Flow Architecture (`server/agent/agentic_flow/`)

**Two-Route System** (`full_agentic_flow.py`):
- **Chat Route**: Simple conversations handled by ReAct agent directly
- **Task Route**: Complex requests processed through specialized agent pipeline

**Task Flow Pipeline** (`task_flow.py`) - **Core Innovation**:
1. **Assessment**: Analyzes complexity and determines research needs
2. **Research**: Multi-researcher system (4 specialized agents: high_precision, breadth_explorer, skeptic_validator, codebase_specialist)
3. **Planning**: Creates execution plan with research context and adaptive replanning
4. **Execution**: Step-by-step tool execution with failure recovery and learning
5. **Control Loop**: Monitors progress and triggers adaptive research when needed

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
- `KnowledgeStore` - Session-scoped file access tracking with episodic memory
- `RAGManager` - Long-term retrieval from domain packs in `learning/packs/`
- Domain packs created via `bin/make_pack.sh` from PDFs/code

## Development Workflows

### Service Management (`./run.sh`)
```bash
# Start all services with hot-reload controls
./run.sh

# Interactive controls (while running):
# [r] restart Agent+UI   [a] Agent only   [u] UI only   [v] vLLM   [q] quit

# Service endpoints:
# Web UI:  http://localhost:7860
# Agent:   http://localhost:8001  
# vLLM:    http://localhost:8003
```

### vLLM Configuration
- Environment: `.env.vllm` (vLLM-specific config) or `.env` (fallback)
- Model: `start_vllm.sh` handles CUDA setup and model loading
- Performance tuning via `TENSOR_PARALLEL_SIZE`, `GPU_MEMORY_UTILIZATION`
- Health checks at `/health` endpoint before agent startup

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

### Agentic Flow Development
- Router logic in `full_agentic_flow.py` - extend `FEW_SHOT_ROUTER_PROMPT` for new routes
- Researcher specialization in `task_flow.py` - modify `researcher_configs` for new research types
- Planning prompts in `helpers_and_prompts.py` - customize reasoning for domain-specific tasks

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

## Critical Development Rules

**ALWAYS use the virtual environment for development.**
- call source .venv/bin/activate to enable the virtual environment

**NEVER create new files of these types:**
- Test files (use `comprehensive_tests.py` only)
- Demo files or example scripts
- Documentation files (README.md, .md files)
- Any file that clutters the repository

**NEVER start/restart servers via commands:**
- Always ASK the user to restart services before proceeding
- Wait for user confirmation that services are ready
- Never execute `./run.sh`, `uvicorn`, `chainlit run`, etc. directly

**NEVER use keyword-based implementations:**
- Always query the LLM for classification, parsing, analysis tasks
- Avoid hardcoded keyword matching or rule-based logic
- Example: Use LLM to classify user intent, not `if "help" in message`
- Leverage the model's intelligence for all advanced processing

## Testing Patterns

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
- `server/agent/agentic_flow/full_agentic_flow.py` - Route determination and chat handling
- `server/agent/agentic_flow/task_flow.py` - Multi-phase task execution pipeline
- `learning/` - Knowledge pack processing pipeline
- `client/anton_client.py` - HTTP streaming client

The tool learning system is Anton's **differentiating feature** - it learns from failures and provides immediate corrective suggestions, making the assistant self-improving.
