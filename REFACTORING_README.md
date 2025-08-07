# Anton Agent Refactoring

This document describes the major refactoring changes made to simplify the Anton agent system and improve performance.

## Key Changes

### 1. Simplified Agentic Loop: ReAct Model

**Before**: Rigid Planner-Doer-Evaluator loop with multiple LLM calls per turn
**After**: Single-agent ReAct (Reason-Act) model

- Replaced complex `organizer.py` orchestration with `react_agent.py`
- Single LLM call per iteration instead of separate planner/doer/evaluator phases
- Significantly reduced latency and complexity
- Agent decides next step (reason, act, or respond) in one call

### 2. Consolidated Code Structure

**Entry Points**:
- **Primary CLI**: `client/main.py` - Command-line interface
- **Server**: `server/agent/agent_server.py` - Sole server entry point

**Removed Redundant Files**:
- `main.py` (broken orchestrator reference)
- `backend.py` (broken orchestrator reference)
- `utils/code_review.py` (broken orchestrator reference)

### 3. State Management Class

**New**: `ConversationState` class in `server/agent/conversation_state.py`

- Centralized management of all request-specific data
- Tracks messages, tool outputs, context, and explored files
- Replaces scattered state management across components
- Provides context building and conversation flow tracking

### 4. Fixed Router Implementation

**QUERY_KNOWLEDGE Path**: Now performs proper RAG queries

- Retrieves relevant documents from indexed knowledge base using `rag_manager`
- Builds context from retrieved documents
- Provides contextually relevant answers based on documents
- Fallback handling for empty knowledge base

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client CLI    │    │  Agent Server    │    │   ReAct Agent   │
│ client/main.py  │───▶│agent_server.py   │───▶│ react_agent.py  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   RAG Manager    │    │ConversationState│
                       │  (Knowledge)     │    │  (State Mgmt)   │
                       └──────────────────┘    └─────────────────┘
```

## Request Flow

1. **Router**: Classifies intent (COMPLEX_CHAT, QUERY_KNOWLEDGE, GENERAL_CHAT)
2. **ReAct Agent**: For complex tasks, uses reason-act pattern
3. **RAG**: For knowledge queries, retrieves and contextualizes documents
4. **Simple Chat**: For general conversation, direct LLM response

## Benefits

- **Reduced Latency**: Single LLM call vs multiple calls per turn
- **Simplified Logic**: One agent vs complex orchestration
- **Better State Management**: Centralized vs scattered state
- **Proper RAG**: Working knowledge retrieval vs broken implementation
- **Cleaner Structure**: Clear entry points vs multiple redundant files

## Testing

Run tests to verify functionality:

```bash
python test_react_agent.py          # Test new components
python test_integration_refactor.py # Integration tests
python test_anton.py                # Existing tests
```

## Migration Notes

- Old `organizer.py` is no longer used but preserved for reference
- `KnowledgeStore` class remains but may be deprecated in favor of `ConversationState`
- Tool execution integrated with new state management via adapter pattern
- Existing tool definitions and management remain unchanged