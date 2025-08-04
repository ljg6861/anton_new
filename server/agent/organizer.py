import json
import re
import time

import ollama
import psutil
from typing import AsyncGenerator, Any, Tuple

from vllm.third_party.pynvml import NVMLError, nvmlShutdown

from torch import cuda

from client.context_builder import ContextBuilder
from metrics import initialize_nvml, MetricsTracker, NVML_INITIALIZED
from server.agent import config
from server.agent.config import SYSTEM_ROLE, ASSISTANT_ROLE, USER_ROLE
from server.agent.doer import execute_turn, run_doer_loop
from server.agent.knowledge_handler import save_reflection
from server.agent.message_handler import prepare_initial_messages
from server.agent.prompts import get_evaluator_prompt
from server.model_server import AgentChatRequest, OLLAMA_HOST, SMALL_MODEL_ID


# --- END METRICS ---

# --- Helper Functions ---

def needs_clarification(message: str) -> bool:
    """Check if message needs user clarification rather than agent action."""
    clarification_keywords = [
        "what do you mean", "can you clarify", "not sure", "unclear",
        "which one", "help me understand", "what would you like",
        "could you specify", "what exactly", "confused"
    ]
    return any(keyword in message.lower() for keyword in clarification_keywords)


def is_simple_request(message: str) -> bool:
    """Check if this is a simple request that can bypass multi-stage architecture."""
    message_lower = message.lower().strip()
    simple_patterns = [
        "hello", "hi", "thanks", "thank you", "what can you do",
        "how are you", "good morning", "good afternoon"
    ]
    # Check if message starts with simple greeting or is just "help"
    return (any(message_lower.startswith(pattern) for pattern in simple_patterns) or 
            message_lower == "help")


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using word overlap."""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0.0


def _get_manager_prompt(original_task: str, conversation_history: str) -> str:
    """Builds the system prompt for the Manager AI model."""
    return f"""
You are a meticulous AI Project Manager. Your sole responsibility is to determine if an AI agent has successfully and completely finished a user's request.

**USER'S ORIGINAL REQUEST:**
---
{original_task}
---

**CONVERSATION HISTORY:**
---
{conversation_history}
---

**YOUR TASK:**
Respond with JSON: `{{"is_finished": <bool>, "reason": "<one-sentence>"}}` only.
"""


async def _call_llm_for_json(
        prompt: str,
        role: str,
        logger: Any,
        api_base_url: str
) -> dict:
    """
    Calls the LLM and parses a JSON response.
    `role` should be 'system' for prompts providing instructions.
    """
    messages = [{"role": role, "content": prompt}]
    # invoke the LLM
    response_gen = execute_turn(api_base_url, messages, logger, [])
    full_response = "".join([token async for token in response_gen])

    try:
        # match JSON block or raw JSON
        match = re.search(r'```json\s*(\{.*?\})\s*```|^(\{.*\})', full_response, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response.")
        json_str = match.group(1) or match.group(2)
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"JSON parse failure: {e}. Response: {full_response}")
        return {"error": str(e), "raw_response": full_response}


async def _call_manager_model(
        messages: list,
        original_task: str,
        logger: Any,
        api_base_url: str
) -> Tuple[bool, str]:
    """Ask the Manager LLM if the conversation has completed the task."""
    conversation_str = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    prompt = _get_manager_prompt(original_task, conversation_str)
    result = await _call_llm_for_json(prompt, SYSTEM_ROLE, logger, api_base_url)
    if result.get("error"):
        return False, "Manager response malformed; continue working."
    return result.get("is_finished", False), result.get("reason", "No reason provided.")


async def handle_simple_request(
        request: AgentChatRequest,
        logger: Any,
        api_base_url: str
) -> AsyncGenerator[str, None]:
    """Handle simple requests with direct response, bypassing multi-stage architecture."""
    messages = prepare_initial_messages(request.messages)
    system_prompt = await ContextBuilder().build_system_prompt_doer()
    messages.insert(0, {"role": SYSTEM_ROLE, "content": system_prompt})
    
    logger.info("Handling simple request directly")
    async for token in execute_turn(api_base_url, messages, logger, request.tools, 0.7, False):
        yield token


# --- Main Orchestrator ---
async def run_organizer_loop(
        request: AgentChatRequest,
        logger: Any,
        api_base_url: str
) -> AsyncGenerator[str, None]:
    # Prepare initial messages and check conversation state
    organizer_messages = prepare_initial_messages(request.messages)
    original_task = organizer_messages[-1]["content"]
    
    # Conversation state management
    if needs_clarification(original_task):
        logger.info("Message needs clarification - pausing agent loop")
        yield "I need clarification to help you better. Could you provide more specific details about what you'd like me to do?"
        return
    
    if is_simple_request(original_task):
        logger.info("Simple request detected - using direct response path")
        async for token in handle_simple_request(request, logger, api_base_url):
            yield token
        return
    
    # --- METRICS: Initialization ---
    initialize_nvml(logger)
    metrics = MetricsTracker(logger)
    # --- END METRICS ---

    complex = True
    instruction_history = []  # Track instructions for loop detection

    # Create a context store to track accumulated information across steps
    context_store = {
        "explored_files": set(),
        "code_content": {},
        "task_progress": []
    }

    system_prompt = await ContextBuilder().build_system_prompt_planner()
    organizer_messages.insert(0, {"role": SYSTEM_ROLE, "content": system_prompt})

    try:
        logger.info("Starting organizer loop...")
        for turn in range(config.MAX_TURNS):
            doer_messages = []
            metrics.agent_step_count = turn + 1
            logger.info(f"Turn {turn + 1}/{config.MAX_TURNS}")

            # --- Planner Turn ---
            planner_step_name = f"Planner-Turn-{turn + 1}"
            planner_start_time = time.monotonic()
            planner_token_count = 0

            # Add context from previous steps to the planner's input
            context_summary = _build_context_summary(context_store)
            if turn > 0:
                organizer_messages.append({
                    "role": SYSTEM_ROLE,
                    "content": f"Previous step progress:\n{context_summary}"
                })

            response_buffer = ""
            logger.info(f"Planner: Calling model server at {api_base_url}/v1/chat/stream")
            async for token in execute_turn(api_base_url, organizer_messages, logger, request.tools, 0.6, complex):
                planner_token_count += 1
                response_buffer += token
                content = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()

            # --- METRICS: Log Planner Step ---
            metrics.step_latencies[planner_step_name] = time.monotonic() - planner_start_time
            metrics.step_token_counts[planner_step_name] = planner_token_count
            metrics.resource_snapshots[planner_step_name] = metrics.get_resource_usage()

            # Extract content after any thinking markers
            content = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
            logger.info("Planner said:\n" + response_buffer)
            
            # Loop detection and prevention
            instruction_history.append(content)
            if len(instruction_history) >= 2:
                last_instruction = instruction_history[-1]
                previous_instruction = instruction_history[-2]
                similarity = calculate_similarity(last_instruction, previous_instruction)
                
                if similarity > 0.85:
                    logger.warning(f"Loop detected! Similarity: {similarity:.2f}")
                    fallback_response = f"I notice I'm repeating similar instructions. Let me provide a direct response to your original request: {original_task}"
                    
                    # Direct fallback response
                    system_prompt = await ContextBuilder().build_system_prompt_doer()
                    fallback_messages = [
                        {"role": SYSTEM_ROLE, "content": system_prompt},
                        {"role": USER_ROLE, "content": fallback_response}
                    ]
                    
                    async for token in execute_turn(api_base_url, fallback_messages, logger, request.tools, 0.7, False):
                        yield token
                    return
            
            organizer_messages.append({"role": ASSISTANT_ROLE, "content": content})

            # --- Prepare Doer with Enhanced Context ---
            if len(doer_messages) == 0:
                system_prompt = await ContextBuilder().build_system_prompt_doer()
                doer_messages.append({"role": SYSTEM_ROLE, "content": system_prompt})

            # Add context about already explored files to the Doer
            if context_store["explored_files"]:
                explored_files_msg = "Previously explored files: " + ", ".join(context_store["explored_files"])
                doer_messages.append({"role": SYSTEM_ROLE, "content": explored_files_msg})

            # Add any previously retrieved file content as context
            for filename, content in context_store["code_content"].items():
                doer_messages.append({
                    "role": SYSTEM_ROLE,
                    "content": f"Content of file {filename}:\n```\n{content}\n```"
                })

            doer_messages.append({"role": USER_ROLE, "content": content})

            # --- METRICS: Doer Turn Timing & Token Count ---
            doer_step_name = f"Doer-Turn-{turn + 1}"
            doer_start_time = time.monotonic()
            doer_token_count = 0
            # --- END METRICS ---

            # Doer's turn
            response_buffer = ""
            async for token in run_doer_loop(doer_messages, request.tools, logger, api_base_url, complex):
                doer_token_count += 1  # METRICS: Count tokens
                yield token

            doer_result = doer_messages[-1]["content"]

            # --- METRICS: Log Doer Step Latency, Throughput and Resources ---
            metrics.step_latencies[doer_step_name] = time.monotonic() - doer_start_time
            metrics.step_token_counts[doer_step_name] = doer_token_count
            metrics.resource_snapshots[doer_step_name] = metrics.get_resource_usage()
            # --- END METRICS ---

            response_buffer = ""
            delegated_step = organizer_messages[-1]["content"]  # Get the last message from the Planner
            evaluator_system_prompt = get_evaluator_prompt() + (
                f"\n\nHere is the information to evaluate:"
                f"\nOriginal High-Level Task: {original_task}"
                f"\nDelegated Step for the Doer: {delegated_step}"
                f"\nDoer's Final Result: {doer_result}"
            )
            organizer_messages.append({
                "role": USER_ROLE,
                "content": f"The doer has completed the delegated task. Here is the result:\n\n{doer_result}"
            })
            evaluator_messages = [{"role": SYSTEM_ROLE, "content": evaluator_system_prompt}]
            async for token in execute_turn(api_base_url, evaluator_messages, logger, request.tools, 0.1, complex):
                response_buffer += token
                content = re.split(r"</think>", response_buffer, maxsplit=1)
                if len(content) == 2:
                    yield token
            logger.info("Evaluator response:\n" + response_buffer)
            evaluator_response = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
            
            if evaluator_response.startswith('SUCCESS:'):
                logger.info("Evaluator confirmed success. Task completed.")
                # Generate final summary and end loop
                summary_prompt = f"The task has been successfully completed. Based on our conversation, please provide a final summary for the original request: {original_task}"
                organizer_messages.append({"role": USER_ROLE, "content": summary_prompt})
                system_prompt = await ContextBuilder().build_system_prompt_doer()
                organizer_messages[0] = {"role": SYSTEM_ROLE, "content": system_prompt}
                
                async for token in execute_turn(api_base_url, organizer_messages, logger, request.tools, 0.7, False):
                    yield token
                return

            elif evaluator_response.startswith('FAILURE:'):
                logger.info("Evaluator reported failure. Planner must adjust.")
                organizer_messages.append({"role": USER_ROLE, "content": evaluator_response})

            elif evaluator_response.startswith('DONE:'):
                organizer_messages.append({"role": USER_ROLE, "content": evaluator_response})
                summary_prompt = f"The entire task has been successfully completed. Based on our conversation history, please summarize the steps taken and provide the final answer to the original high-level task: {original_task}"
                organizer_messages.append({"role": USER_ROLE, "content": summary_prompt})
                system_prompt = await ContextBuilder().build_system_prompt_doer()
                organizer_messages[0] = {"role": SYSTEM_ROLE, "content": system_prompt}
                
                async for token in execute_turn(api_base_url, organizer_messages, logger, request.tools, 0.7, False):
                    yield token
                return

        # Max turns reached
        logger.warning("Max turns reached; ending.")
        cuda.empty_cache()
        # --- METRICS: Mark task as failed due to max turns ---
        metrics.task_completed = False
        metrics.task_completion_reason = "Reached maximum turns"
        # --- END METRICS ---
        yield "\n\n[Reached maximum turns; stopping organizer.]"
    finally:
        metrics.log_final_metrics()
        if NVML_INITIALIZED:
            try:
                nvmlShutdown()
            except NVMLError:
                pass  # Already logged warnings if shutdown fails


def _build_context_summary(context_store):
    """Build a summary of the context for the planner"""
    summary_parts = []

    if context_store["explored_files"]:
        summary_parts.append("Explored files: " + ", ".join(context_store["explored_files"]))

    if context_store["code_content"]:
        summary_parts.append("Retrieved file contents:")
        for filename in context_store["code_content"].keys():
            summary_parts.append(f"- {filename}")

    if context_store["task_progress"]:
        summary_parts.append("Progress so far:")
        for step in context_store["task_progress"]:
            summary_parts.append(f"- {step}")

    return "\n".join(summary_parts)