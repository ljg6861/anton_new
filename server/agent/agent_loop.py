# agent/agent_loop.py

"""
The core logic for the multi-turn, streaming agent, now with a fast-path
complexity check.
"""
import json
import re
from typing import AsyncGenerator, Dict, Any, List

from server.agent.prompts import get_thinking_prompt
from server.agent.streaming import stream_response
from server.agent.tool_executor import execute_tool


def _get_generation_kwargs(request: Any, tokenizer: Any) -> Dict[str, Any]:
    """
    Constructs the generation arguments for the model for the main thinking loop.
    """
    return {
        "max_new_tokens": 1024,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p" : 0
    }


def _prepare_messages(request: Any) -> List[Dict[str, Any]]:
    """
    Prepares the list of messages for the agent, injecting the
    system prompt and tool definitions.
    """
    # Use a single, robust prompt for all tasks
    messages = []

    messages.extend([msg.model_dump() for msg in request.messages])
    return messages

async def run_agent_loop(request: Any, logger: Any, model: Any, tokenizer: Any) -> AsyncGenerator[str, None]:
    """
    Manages the agentic loop with a mechanism to detect and break reasoning loops.
    """
    messages = _prepare_messages(request)
    gen_kwargs = _get_generation_kwargs(request, tokenizer)
    MAX_TURNS = 10  # Reduced max turns as we are now smarter about loops
    tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    thought_summary_regex = re.compile(r"<thought_summary>(.*?)</thought_summary>", re.DOTALL)
    sentence_end_regex = re.compile(r"[.?!](?:\s|$)")

    recent_thoughts: list[str] = []
    LOOP_DETECTION_THRESHOLD = 2

    def is_paragraph_end(buf: list[str]) -> bool:
        """Heuristic: 2+ sentence enders AND a blank-line break."""
        joined = "".join(buf)
        if "</think" in joined.lower():
            return True
        sentence_count = len(sentence_end_regex.findall(joined))
        has_blank_line = "\n\n" in joined
        return sentence_count >= 2 and has_blank_line

    async def summarize_chunk(text: str):
        if not text.strip():
            return
        summary_kwargs = {**gen_kwargs, "max_new_tokens": 128}
        prompt = f"Summarize the following text:\n{text}"
        async for tok in stream_response(model, tokenizer, prompt, summary_kwargs):
            yield tok

    for turn in range(MAX_TURNS):
        logger.info(f"--- Agent Turn {turn + 1}/{MAX_TURNS} ---")

        response_buffer = ""
        thoughts_chunk: list[str] = []

        stream_generator = stream_response(model, tokenizer, messages, gen_kwargs)
        thinking = True

        async for token in stream_generator:
            stripped = token.strip().lower()

            if thinking:
                # collect raw token
                thoughts_chunk.append(token)

                # end of <think>?
                if "</think" in stripped:
                    thinking = False
                    # summarize any remaining chunk
                    text = "".join(thoughts_chunk).strip()
                    thoughts_chunk.clear()
                    async for s_tok in summarize_chunk(text):
                        yield s_tok

                # paragraph boundary inside <think>
                elif is_paragraph_end(thoughts_chunk):
                    text = "".join(thoughts_chunk).strip()
                    thoughts_chunk.clear()
                    async for s_tok in summarize_chunk(text):
                        yield s_tok

            # outside think, just stream normal reply
            response_buffer += token

        # After the full response is received, parse it
        thought_match = thought_summary_regex.search(response_buffer)
        current_thought = thought_match.group(1).strip() if thought_match else None

        if current_thought:
            normalized_thought = " ".join(current_thought.lower().split())
            recent_thoughts.append(normalized_thought)

            # Keep the history to a manageable size
            if len(recent_thoughts) > LOOP_DETECTION_THRESHOLD:
                recent_thoughts.pop(0)

            # Check if the agent is stuck
            if len(recent_thoughts) == LOOP_DETECTION_THRESHOLD and len(set(recent_thoughts)) == 1:
                logger.warning("Agent is stuck in a reasoning loop. Intervening.")
                yield "\n[INFO: Agent appears to be stuck. Forcing a new approach...]\n"

                # Inject a system message to break the loop
                messages.append({
                    "role": "user",
                    "content": "You are repeating the exact same thought process. This indicates you are stuck. You MUST change your plan. Re-evaluate the problem from the beginning, try a different tool, or ask the user for help."
                })

                # Clear the history to give the agent a fresh start after intervention
                recent_thoughts.clear()
                continue  # Skip the rest of this turn and let the model generate a new plan
        else:
            # If there's no thought, the agent is likely giving a final answer, so reset the tracker.
            recent_thoughts.clear()

        # Stream the thought to the user
        if current_thought:
            yield f"🤔 {current_thought}\n"

        # Append the full model response to the message history
        split_response = response_buffer.split("</think>")
        content = split_response[1] if len(split_response) > 1 else response_buffer

        messages.append({"role": "assistant", "content": content})
        tool_match = tool_call_regex.search(response_buffer)
        if not tool_match:
            # If no tool call, stream the final answer (excluding thoughts)
            final_answer = thought_summary_regex.sub("", response_buffer).strip()
            yield final_answer
            logger.info("No tool call detected. Ending conversation.")
            return

        # Execute the tool call
        tool_call_content = tool_match.group(1).strip()
        try:
            tool_data = json.loads(tool_call_content)
            tool_name = tool_data["name"]
            tool_args = tool_data.get("arguments", {})

            yield f"\n[INFO: Calling tool `{tool_name}`...]\n"
            tool_result = execute_tool(tool_name, tool_args, logger)
            tool_result_str = json.dumps({"result": tool_result})
            messages.append({"role": "tool", "content": tool_result_str})

        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Error: The model produced an invalid tool call. Reason: {e}"
            logger.error(f"{error_msg}\nContent: {tool_call_content}")
            # Feed the error back to the model so it can self-correct
            messages.append({"role": "tool", "content": json.dumps({"error": error_msg})})
            yield f"\n[{error_msg}]\n"

    logger.warning("Agent reached maximum turn limit.")
    yield "\n[Agent reached maximum turns. Ending conversation.]"
