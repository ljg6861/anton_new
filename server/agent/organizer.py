import json
import re
import time
from typing import AsyncGenerator, Any
from vllm.third_party.pynvml import NVMLError, nvmlShutdown
from torch import cuda
from client.context_builder import ContextBuilder
from metrics import initialize_nvml, MetricsTracker, NVML_INITIALIZED
from server.agent import config
from server.agent.config import SYSTEM_ROLE, ASSISTANT_ROLE, USER_ROLE
from server.agent.doer import execute_turn, run_doer_loop
from server.agent.knowledge_store import KnowledgeStore, ImportanceLevel
from server.agent.message_handler import prepare_initial_messages
from server.agent.prompts import get_evaluator_prompt
from server.model_server import AgentChatRequest

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
    is_complex = True

    # Create a centralized knowledge store to track and persist context across all components
    knowledge_store = KnowledgeStore()

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
            context_summary = knowledge_store.build_context_summary()
            if turn > 0:
                organizer_messages.append({
                    "role": SYSTEM_ROLE,
                    "content": f"Previous step progress:\n{context_summary}"
                })
            
            # Query relevant knowledge from past tasks
            if turn > 0:
                relevant_knowledge = knowledge_store.query_relevant_knowledge(original_task)
                if relevant_knowledge:
                    knowledge_context = "Relevant knowledge from past tasks:\n" + "\n".join(f"- {k[:200]}..." for k in relevant_knowledge[:3])
                    organizer_messages.append({
                        "role": SYSTEM_ROLE,
                        "content": knowledge_context
                    })

            response_buffer = ""
            logger.info(f"Planner: Calling model server at {api_base_url}/v1/chat/stream")
            async for token in execute_turn(api_base_url, organizer_messages, logger, request.tools, 0.6, is_complex):
                planner_token_count += 1
                response_buffer += token

            # --- METRICS: Log Planner Step ---
            metrics.step_latencies[planner_step_name] = time.monotonic() - planner_start_time
            metrics.step_token_counts[planner_step_name] = planner_token_count
            metrics.resource_snapshots[planner_step_name] = metrics.get_resource_usage()

            # Extract content after any thinking markers
            content = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()
            logger.info("Planner said:\n" + response_buffer)
            organizer_messages.append({"role": ASSISTANT_ROLE, "content": content})
            
            # Capture planner insights
            if content and not content.startswith("I need to"):  # Avoid capturing simple delegation
                knowledge_store.add_planner_insight(content[:500], ImportanceLevel.MEDIUM)

            # --- Prepare Doer with Enhanced Context ---
            if len(doer_messages) == 0:
                system_prompt = await ContextBuilder().build_system_prompt_doer()
                doer_messages.append({"role": SYSTEM_ROLE, "content": system_prompt})

            # Add context about already explored files to the Doer
            if knowledge_store.explored_files:
                explored_files_msg = "Previously explored files: " + ", ".join(list(knowledge_store.explored_files)[:10])
                doer_messages.append({"role": SYSTEM_ROLE, "content": explored_files_msg})

            # Add any previously retrieved file content as context
            for filename, content in list(knowledge_store.code_content.items())[:5]:  # Limit to avoid token overflow
                doer_messages.append({
                    "role": SYSTEM_ROLE,
                    "content": f"Content of file {filename}:\n```\n{content[:2000]}\n```"  # Truncate long files
                })

            doer_messages.append({"role": USER_ROLE, "content": content})

            await run_doer_loop(doer_messages, request.tools, logger, api_base_url, is_complex, knowledge_store)

            doer_result = doer_messages[-1]["content"]

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
            async for token in execute_turn(api_base_url, evaluator_messages, logger, request.tools, 0.6, True):
                response_buffer += token

            logger.info("Evaluator response:\n" + response_buffer)
            evaluator_response = re.split(r"</think>", response_buffer, maxsplit=1)[-1].strip()

            if evaluator_response.strip().startswith("```json"):
                evaluator_response = evaluator_response.strip()[7:-3].strip()

            parsed_intent = json.loads(evaluator_response)
            result = parsed_intent.get("result")
            explanation = parsed_intent.get("explanation")
            
            # Store evaluator feedback in knowledge store
            if evaluator_response:
                if result == 'FAILURE':
                    knowledge_store.add_evaluator_feedback(explanation, ImportanceLevel.HIGH)
                    logger.info("Evaluator reported failure. Planner must adjust.")
                    organizer_messages.append({"role": USER_ROLE, "content": evaluator_response})
                elif result == 'SUCCESS':
                    knowledge_store.add_evaluator_feedback(explanation, ImportanceLevel.MEDIUM)
                    logger.info("Evaluator confirmed success. Planner proceeds.")
                    pass  # Continue the loop to the next Planner turn
                elif result == 'DONE':
                    knowledge_store.add_evaluator_feedback(explanation, ImportanceLevel.CRITICAL)
                    organizer_messages.append({"role": USER_ROLE, "content": explanation})

                    #Done with task, begin creating summary for the user
                    summary_prompt = f"The entire task has been successfully completed. Based on our conversation history, please summarize the steps taken and provide the final answer to the original high-level task: {original_task}"
                    organizer_messages.append({"role": USER_ROLE, "content": summary_prompt})
                    system_prompt = await ContextBuilder().build_system_prompt_doer()
                    organizer_messages[0] = {"role": SYSTEM_ROLE, "content": system_prompt}
                    final_response_buffer = ""
                    async for token in execute_turn(api_base_url, organizer_messages, logger, request.tools, 0.6,
                                                    is_complex):
                        final_response_buffer += token
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

