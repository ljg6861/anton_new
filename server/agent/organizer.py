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

# Optional learning components (may not be available in all environments)
try:
    from server.agent.learning_identifier import learning_identifier
    from server.agent.rag_manager import rag_manager
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False
    learning_identifier = None
    rag_manager = None


# --- END METRICS ---

# --- Helper Functions ---
def _calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple similarity between two instruction texts."""
    if not text1 or not text2:
        return 0.0
    
    # Simple word-based similarity
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
    complex = True

    # Workflow optimization: Add loop detection and iteration limits
    max_iterations = 10
    current_iteration = 0
    recent_instructions = []
    similarity_threshold = 0.85

    # Create a context store to track accumulated information across steps
    context_store = {
        "explored_files": set(),
        "code_content": {},
        "task_progress": [],
        "tool_outputs": []
    }

    system_prompt = await ContextBuilder().build_system_prompt_planner()
    organizer_messages.insert(0, {"role": SYSTEM_ROLE, "content": system_prompt})

    try:
        logger.info("Starting enhanced organizer loop...")
        for turn in range(min(config.MAX_TURNS, max_iterations)):
            doer_messages = []
            metrics.agent_step_count = turn + 1
            current_iteration = turn + 1
            logger.info(f"Turn {current_iteration}/{max_iterations}")

            # Check iteration limit
            if current_iteration > max_iterations:
                logger.warning(f"Maximum iterations ({max_iterations}) reached, stopping")
                yield f"\n[Maximum iterations reached. Task may need manual intervention.]\n"
                break

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
            
            # Enhanced execution with timeout protection
            try:
                async for token in execute_turn(api_base_url, organizer_messages, logger, request.tools, 0.6, complex):
                    planner_token_count += 1
                    response_buffer += token
                    content = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
            except Exception as e:
                logger.error(f"Planner execution failed: {e}")
                yield f"\n[Planner error: {str(e)}]\n"
                continue

            # --- METRICS: Log Planner Step ---
            metrics.step_latencies[planner_step_name] = time.monotonic() - planner_start_time
            metrics.step_token_counts[planner_step_name] = planner_token_count
            metrics.resource_snapshots[planner_step_name] = metrics.get_resource_usage()

            # Extract content after any thinking markers
            content = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
            logger.info("Planner said:\n" + response_buffer)
            
            # Loop detection: Check if this instruction is too similar to recent ones
            instruction_normalized = content.lower().strip()
            
            # Check for loops
            loop_detected = False
            if len(recent_instructions) >= 2:
                for recent_instruction in recent_instructions[-2:]:
                    similarity = _calculate_similarity(instruction_normalized, recent_instruction)
                    if similarity >= similarity_threshold:
                        loop_detected = True
                        logger.warning(f"Loop detected! Similarity: {similarity:.2f}")
                        break
            
            if loop_detected:
                # Inject pattern-breaking instruction
                pattern_break_msg = (
                    "PATTERN DETECTED: You are repeating similar instructions. "
                    "Try a completely different approach, use different tools, "
                    "or break down the problem differently."
                )
                organizer_messages.append({"role": USER_ROLE, "content": pattern_break_msg})
                yield f"\n[LOOP DETECTED - Trying different approach]\n"
                
                # Clear recent instructions and continue
                recent_instructions.clear()
                continue
            
            # Store this instruction for loop detection
            recent_instructions.append(instruction_normalized)
            if len(recent_instructions) > 3:
                recent_instructions.pop(0)
            
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

            # Doer's turn with enhanced error handling and timeout
            response_buffer = ""
            doer_success = False
            
            try:
                logger.info(f"Doer: Starting execution with timeout protection")
                async for token in run_doer_loop(doer_messages, request.tools, logger, api_base_url, complex):
                    doer_token_count += 1  # METRICS: Count tokens
                    yield token
                doer_success = True
                
            except Exception as e:
                logger.error(f"Doer execution failed: {e}")
                yield f"\n[Doer execution error: {str(e)}. Trying different approach...]\n"
                doer_success = False

            if not doer_success:
                # Skip to next iteration if doer failed
                continue

            doer_result = doer_messages[-1]["content"]

            # --- METRICS: Log Doer Step Latency, Throughput and Resources ---
            metrics.step_latencies[doer_step_name] = time.monotonic() - doer_start_time
            metrics.step_token_counts[doer_step_name] = doer_token_count
            metrics.resource_snapshots[doer_step_name] = metrics.get_resource_usage()
            # --- END METRICS ---

            # Enhanced evaluation with three-level assessment
            response_buffer = ""
            delegated_step = organizer_messages[-1]["content"]  # Get the last message from the Planner
            
            # Enhanced evaluator prompt with three-level assessment
            evaluator_system_prompt = get_evaluator_prompt() + (
                f"\n\nHere is the information to evaluate:"
                f"\nOriginal High-Level Task: {original_task}"
                f"\nDelegated Step for the Doer: {delegated_step}"
                f"\nDoer's Final Result: {doer_result}"
                f"\n\nProvide one of these verdicts:"
                f"\n- SUCCESS: The step was completed successfully and advances toward the goal"
                f"\n- PARTIAL: Some progress was made but the step is incomplete"
                f"\n- FAILURE: The step failed or made no progress"
                f"\n- DONE: The entire original task is now complete"
            )
            
            organizer_messages.append({
                "role": USER_ROLE,
                "content": f"The doer has completed the delegated task. Here is the result:\n\n{doer_result}"
            })
            
            evaluator_messages = [{"role": SYSTEM_ROLE, "content": evaluator_system_prompt}]
            
            try:
                async for token in execute_turn(api_base_url, evaluator_messages, logger, request.tools, 0.1, complex):
                    response_buffer += token
                    content = re.split(r"</think>", response_buffer, maxsplit=1)
                    if len(content) == 2:
                        yield token
            except Exception as e:
                logger.error(f"Evaluator execution failed: {e}")
                response_buffer = "PARTIAL: Evaluator failed, continuing with caution."
                
            logger.info("Evaluator response:\n" + response_buffer)
            evaluator_response = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
            
            # Store tool outputs for learning
            if "tool_outputs" in context_store:
                context_store["tool_outputs"].append({
                    "instruction": delegated_step,
                    "result": doer_result,
                    "evaluation": evaluator_response
                })

            # Enhanced three-level evaluation processing
            if evaluator_response.startswith('SUCCESS:'):
                logger.info("Evaluator reported success. Continuing to next step.")
                organizer_messages.append({"role": USER_ROLE, "content": evaluator_response})
                # Continue the loop
                
            elif evaluator_response.startswith('PARTIAL:'):
                logger.info("Evaluator reported partial success. Providing feedback and continuing.")
                organizer_messages.append({"role": USER_ROLE, "content": evaluator_response})
                # Continue the loop with feedback

            elif evaluator_response.startswith('FAILURE:'):
                logger.info("Evaluator reported failure. Planner must adjust approach.")
                organizer_messages.append({"role": USER_ROLE, "content": evaluator_response})
                # Add suggestion for different approach
                organizer_messages.append({
                    "role": USER_ROLE, 
                    "content": "The previous approach failed. Please try a different strategy or tool."
                })

            elif evaluator_response.startswith('DONE:'):
                logger.info("Evaluator reported task completion!")
                organizer_messages.append({"role": USER_ROLE, "content": evaluator_response})

                summary_prompt = f"The entire task has been successfully completed. Based on our conversation history, please summarize the steps taken and provide the final answer to the original high-level task: {original_task}"
                organizer_messages.append({"role": USER_ROLE, "content": summary_prompt})
                system_prompt = await ContextBuilder().build_system_prompt_doer()
                organizer_messages[0] = {"role": SYSTEM_ROLE, "content": system_prompt}
                final_response_buffer = ""
                async for token in execute_turn(api_base_url, organizer_messages, logger, request.tools, complex):
                    final_response_buffer += token
                    yield token
                    
                # Task completed successfully
                logger.info("Task completed successfully")
                return
                
            else:
                # Default handling for unclear evaluation
                logger.warning(f"Unclear evaluator response: {evaluator_response}")
                organizer_messages.append({
                    "role": USER_ROLE, 
                    "content": f"Evaluation unclear: {evaluator_response}. Please continue or clarify."
                })

            # Learning analysis: Capture insights from this turn
            try:
                if LEARNING_AVAILABLE and learning_identifier and hasattr(learning_identifier, 'analyze_interaction'):
                    doer_output = doer_result if 'doer_result' in locals() else ""
                    learning_result = learning_identifier.analyze_interaction(
                        instruction=delegated_step,
                        output=doer_output,
                        context=str(context_store)
                    )
                    
                    if learning_result.get("has_learning", False):
                        learning_text = learning_result.get("insight", "")
                        if rag_manager and hasattr(rag_manager, 'store_insight'):
                            rag_manager.store_insight(learning_text)
                            logger.info(f"Learning captured: {learning_text[:100]}...")
                        
            except Exception as e:
                logger.debug(f"Learning processing failed: {e}")
                # Don't fail the main loop for learning issues

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