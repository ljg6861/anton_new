import json
import re
import time
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
from server.model_server import AgentChatRequest


# --- END METRICS ---

# --- Helper Functions ---
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


# --- Reflection & Learning Loop ---
def _get_reflector_prompt(original_task: str, final_conversation: str) -> str:
    """Builds the system prompt for the Reflector AI model."""
    return f"""
You are a Reflector AI. Analyze a completed user-agent conversation to extract key insights.

**ORIGINAL REQUEST:**
---
{original_task}
---

**CONVERSATION:**
---
{final_conversation}
---

Generate JSON: {{
  "summary": "<1-2 sentences>",
  "key_takeaway": "<single lesson>",
  "strategy": "<high-level strategy>",
  "keywords": ["<keyword1>", ...]
}}
"""


async def _run_reflection_and_learning_loop(
        original_task: str,
        messages: list,
        logger: Any,
        api_base_url: str
) -> None:
    logger.info("Starting reflection loop...")
    final_convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    prompt = _get_reflector_prompt(original_task, final_convo)

    for attempt in range(3):
        logger.info(f"Reflection attempt {attempt + 1}")
        result = await _call_llm_for_json(prompt, SYSTEM_ROLE, logger, api_base_url)
        if all(k in result for k in ("summary", "key_takeaway", "strategy", "keywords")):
            await save_reflection(original_task, result, logger)
            logger.info("Reflection saved.")
            return
    logger.error("Failed to generate valid reflection after retries.")


# --- Main Orchestrator ---
async def run_organizer_loop(
        request: AgentChatRequest,
        logger: Any,
        api_base_url: str
) -> AsyncGenerator[str, None]:
    # --- METRICS: Initialization ---
    initialize_nvml(logger)
    metrics = MetricsTracker(logger)
    # --- END METRICS ---

    # Prepare initial messages
    organizer_messages = prepare_initial_messages(request.messages)
    original_task = organizer_messages[-1]["content"]

    # Build system prompt and insert
    system_prompt = await ContextBuilder().build_system_prompt_planner(original_task)
    organizer_messages.insert(0, {"role": SYSTEM_ROLE, "content": system_prompt})
    doer_messages = []

    try:  # --- METRICS: Use a try/finally block to ensure metrics are always logged ---
        logger.info("Starting organizer loop...")
        for turn in range(config.MAX_TURNS):
            metrics.agent_step_count = turn + 1  # METRICS: Update step count
            logger.info(f"Turn {turn + 1}/{config.MAX_TURNS}")

            # --- METRICS: Planner Turn Timing & Token Count ---
            planner_step_name = f"Planner-Turn-{turn + 1}"
            planner_start_time = time.monotonic()
            planner_token_count = 0
            # --- END METRICS ---

            # Planner's turn
            response_buffer = ""
            logger.info(f"Planner: Calling model server at {api_base_url}/v1/chat/stream")
            async for token in execute_turn(api_base_url, organizer_messages, logger, request.tools):
                planner_token_count += 1  # METRICS: Count tokens
                response_buffer += token
                content = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
                if content.startswith('FINAL ANSWER:'):
                    yield token

            # --- METRICS: Log Planner Step Latency, Throughput and Resources ---
            metrics.step_latencies[planner_step_name] = time.monotonic() - planner_start_time
            metrics.step_token_counts[planner_step_name] = planner_token_count
            metrics.resource_snapshots[planner_step_name] = metrics.get_resource_usage()
            # --- END METRICS ---

            # Extract content after any thinking markers
            content = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
            logger.info("Planner said:\n" + response_buffer)
            organizer_messages.append({"role": ASSISTANT_ROLE, "content": content})
            if content.startswith('FINAL ANSWER:'):
                cuda.empty_cache()
                # --- METRICS: Mark task as completed ---
                metrics.task_completed = True
                metrics.task_completion_reason = "Agent returned 'FINAL ANSWER:'"
                # --- END METRICS ---
                return

            if len(doer_messages) == 0:
                system_prompt = await ContextBuilder().build_system_prompt_planner(content)
                doer_messages.append({"role": SYSTEM_ROLE, "content": system_prompt})
            doer_messages.append({"role": USER_ROLE, "content": content})

            # --- METRICS: Doer Turn Timing & Token Count ---
            doer_step_name = f"Doer-Turn-{turn + 1}"
            doer_start_time = time.monotonic()
            doer_token_count = 0
            # --- END METRICS ---

            # Doer's turn
            response_buffer = ""
            async for token in run_doer_loop(doer_messages, request.tools, logger, api_base_url):
                doer_token_count += 1  # METRICS: Count tokens
                response_buffer += token
                yield token

            # --- METRICS: Log Doer Step Latency, Throughput and Resources ---
            metrics.step_latencies[doer_step_name] = time.monotonic() - doer_start_time
            metrics.step_token_counts[doer_step_name] = doer_token_count
            metrics.resource_snapshots[doer_step_name] = metrics.get_resource_usage()
            # --- END METRICS ---

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
