# Anton Agent Learning System

The Anton agent system includes an advanced learning identification and memory integration system that enables the agent to learn from its interactions and apply that knowledge to future tasks.

## Overview

The learning system consists of four main components:

1. **Learning Identification Component** - Analyzes agent interactions to identify novel insights
2. **Enhanced RAG System** - Stores and retrieves learned knowledge using vector similarity or fallback text matching
3. **Memory Integration** - Injects relevant memories into agent prompts
4. **Automatic Learning Trigger** - Determines when to store new knowledge

## Components

### 1. Learning Identification Component (`learning_identifier.py`)

Analyzes agent interactions using pattern matching to identify valuable insights:

- **Pattern-based Detection**: Uses regex patterns to find discovery statements, successful actions, and system understanding
- **Confidence Scoring**: Assigns confidence scores based on pattern weights and keyword presence
- **Duplicate Filtering**: Removes similar insights to prevent noise
- **Learning Triggers**: Determines when insights are valuable enough to store

#### Key Patterns:
- Discovery statements: "discovered that...", "found that...", "identified that..."
- Content descriptions: "contains...", "implements...", "uses..."
- Successful actions: "successfully read...", "effectively processed..."
- System structure: "the system uses...", "application handles..."

### 2. Enhanced RAG System (`rag_manager_enhanced.py`)

Provides robust knowledge storage and retrieval:

- **Fallback Implementation**: Works with or without ML dependencies (FAISS, sentence-transformers)
- **Simple Text Matching**: Uses keyword and text overlap scoring when ML libraries unavailable
- **JSON Storage**: Portable, debuggable storage format
- **Memory Management**: Automatically removes old entries to prevent excessive growth
- **Error Handling**: Graceful degradation on failures

#### Storage Format:
```json
{
  "entries": {
    "entry_id": {
      "text": "Knowledge content...",
      "source": "learning_source",
      "keywords": ["keyword1", "keyword2"],
      "timestamp": 1234567890.123,
      "entry_id": "unique_hash"
    }
  }
}
```

### 3. Memory Integration (`context_builder.py`)

Enhances agent prompts with relevant memories:

- **Task-based Retrieval**: Searches for relevant memories based on current task description
- **Prompt Injection**: Injects memories into the `{memory_context}` placeholder in prompts
- **Context Truncation**: Limits memory length to keep prompts manageable
- **Relevance Ranking**: Orders memories by relevance to current task

### 4. Automatic Learning Integration (`organizer.py`)

Triggers learning after each agent interaction:

- **Interaction Analysis**: Analyzes doer responses and tool outputs
- **Learning Decision**: Uses heuristics to determine if learning should be triggered
- **Knowledge Storage**: Stores valuable insights in the RAG system
- **Performance Monitoring**: Tracks learning activities in logs

## Usage

### Testing the Learning System

Run the test script to verify all components work together:

```bash
python test_learning_system.py
```

### Manual Learning Trigger

You can manually add knowledge to the system:

```python
from server.agent.rag_manager import rag_manager

rag_manager.add_knowledge(
    text="The system uses Flask for the web interface",
    source="manual_entry"
)
rag_manager.save()
```

### Retrieving Memories

Search for relevant memories:

```python
from server.agent.rag_manager import rag_manager

results = rag_manager.retrieve_knowledge("Flask web interface", top_k=3)
for result in results:
    print(f"Memory: {result['text']}")
    print(f"Source: {result['source']}")
```

### Building Prompts with Memory

Use the context builder to inject memories:

```python
from client.context_builder import ContextBuilder

builder = ContextBuilder()
prompt = await builder.build_system_prompt_planner("Analyze Flask application")
# Prompt now includes relevant memories in the {memory_context} section
```

## Configuration

### Learning Sensitivity

Adjust the learning identifier's sensitivity by modifying confidence thresholds:

```python
# In learning_identifier.py
learning_identifier.min_confidence = 0.5  # Lower = more sensitive
```

### Memory Limits

Configure memory management settings:

```python
# In rag_manager_enhanced.py
rag_manager = SimpleRAGManager(max_entries=2000)  # Increase memory capacity
```

### Pattern Customization

Add custom learning patterns:

```python
# In learning_identifier.py
custom_patterns = {
    "custom_discovery": {
        "pattern": r"(?:noticed|observed)\s+(.{10,200})",
        "weight": 0.7,
        "keywords": ["noticed", "observed", "pattern"]
    }
}
learning_identifier.insight_patterns.update(custom_patterns)
```

## Architecture Benefits

1. **Automatic Learning**: No manual intervention required for knowledge acquisition
2. **Robust Fallback**: Works in any environment, with or without ML dependencies
3. **Performance Optimized**: Minimal impact on agent response times
4. **Memory Efficient**: Automatic cleanup prevents unbounded growth
5. **Contextual Retrieval**: Memories are injected based on task relevance
6. **Extensible**: Easy to add new learning patterns and storage backends

## Files Modified

- `server/agent/learning_identifier.py` - New learning identification component
- `server/agent/rag_manager_enhanced.py` - New enhanced RAG implementation
- `server/agent/rag_manager.py` - Updated to use enhanced version with fallback
- `server/agent/organizer.py` - Added learning analysis after each interaction
- `server/agent/tool_executor.py` - Added tool output tracking for learning
- `client/context_builder.py` - Added memory injection into prompts
- `test_learning_system.py` - Comprehensive test suite for learning system

## Future Enhancements

1. **Adaptive Learning**: Adjust learning sensitivity based on task success rates
2. **Memory Clustering**: Group related memories for better organization
3. **Export/Import**: Support for sharing learned knowledge between instances
4. **Analytics Dashboard**: Web interface for viewing and managing learned knowledge
5. **Integration with External Knowledge**: Connect to documentation, wikis, etc.