import asyncio
import json
import re
from typing import List, Dict, Any, AsyncGenerator
from tools.tool_defs import ALL_TOOLS, TOOL_REGISTRY


# --- Tool Prompt Generation ---
def get_tool_system_prompt():
    tool_definitions = [tool.function for tool in ALL_TOOLS]
    return f"""
# Tool Usage Rules

You have a two-step process for answering requests:

1.  **Always Plan First:** Before using any tools, you **MUST** first write a brief, user-facing sentence or two explaining your overall plan.
2.  **Execute the Plan:** After you state your plan, use the most direct tool to accomplish it. For each tool you use, provide a `summary` field that explains that specific step. You may use more than one tool in your response IF NECESSARY. YOU MUST PUT EACH JSON TOOL IN XML <tool_call>...</tool_call>.

---

## Tool Result Substitution

You can use the output of an earlier tool in a later tool's arguments.
- Use the placeholder format `{{{{results[n]}}}}` where `n` is the zero-based index of the tool call whose result you want to use.
- You can use multiple placeholders in a single tool's arguments to combine results.

---

## Correct Usage Example (Writing a File)

**User:** "Write a short story about a fox and save it to `fox_story.txt`."

**Your Correct Response:**
Of course. I will write a short story about a fox and save it directly to `fox_story.txt` for you.
<tool_call>
{{
  "name": "write_file",
  "arguments": {{
    "file_path": "fox_story.txt",
    "content": "Felix the fox adjusted his monocle. The case of the missing acorns was proving to be a tough nut to crack. He followed a trail of tiny footprints to the old oak tree, ready to confront the culprit."
  }},
  "summary": "I am writing the story directly to the file 'fox_story.txt'."
}}
</tool_call>

---

## Available Tools
<tools>
{json.dumps(tool_definitions, indent=2)}
</tools>
"""
# --- LLM Invocation Helper ---
async def _invoke_llm(messages: List[Dict], temperature: float, model, tokenizer, think) -> AsyncGenerator[Any, Any]:
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking = think)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

    outputs_generator = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    full_response_ids = outputs_generator.sequences[0]

    previous_length = len(inputs.input_ids[0])
    for i in range(previous_length, len(full_response_ids)):
        current_token_id = full_response_ids[i]
        decoded_token = tokenizer.decode(current_token_id, skip_special_tokens=True)
        yield decoded_token
        await asyncio.sleep(0.01)


async def execute_agent(request, logger, model, tokenizer):
    MAX_TURNS = 5
    system_prompt = get_tool_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend([msg.model_dump() for msg in request.messages])

    for turn in range(MAX_TURNS):
        logger.info(f"Agent turn {turn + 1}/{MAX_TURNS}...")

        full_assistant_response_content = ""
        is_streaming_plan = True  # Flag to control streaming the initial text.

        # --- CHANGE 1: Stream the initial plan until a tool call is found. ---
        async for token in _invoke_llm(messages, request.temperature, model, tokenizer, False):
            full_assistant_response_content += token

            # Once we find a tool call, we stop streaming the raw tokens.
            if "<tool_call>" in full_assistant_response_content:
                is_streaming_plan = False

            if is_streaming_plan:
                yield token

        logger.info(f"LLM output: \"{full_assistant_response_content}\"")
        messages.append({"role": "assistant", "content": full_assistant_response_content})

        tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', full_assistant_response_content, re.DOTALL)

        if not tool_calls:
            logger.info("✅ No tool calls detected. Final response was streamed.")
            # If no tools were found, the entire response was the final answer and
            # has already been streamed. We can exit.
            return

        # --- CHANGE 2: Stream summaries with a fallback. ---
        logger.info(f"Detected {len(tool_calls)} tool call(s). Streaming summaries and executing...")

        for tool_json_str in tool_calls:
            try:
                tool_data = json.loads(tool_json_str)
                summary = tool_data.get("summary")

                if summary:
                    yield f"▶️ {summary}\n"
                else:
                    # Fallback: If model forgets summary, create a generic one.
                    tool_name = tool_data.get("name", "an unknown tool")
                    yield f"▶️ Executing tool: `{tool_name}`...\n"

                await asyncio.sleep(0.05)  # Helps ensure chunks are sent distinctly.

            except json.JSONDecodeError:
                logger.warning(f"Could not parse tool call: {tool_json_str}")
                yield "⚠️ Error processing a tool step.\n"
                continue

        # Execute the tool chain for the next turn.
        observations = _execute_tool_chain(tool_calls)
        for obs_str in observations:
            messages.append({"role": "tool", "content": obs_str})

    final_content = "Maximum tool call limit reached. Unable to get a final answer."
    logger.warning(final_content)
    yield final_content

def _execute_tool_chain(tool_calls: List[str]) -> List[str]:
    """Executes a chain of tool calls, allowing later calls to use earlier results."""
    executed_results = []
    observations_for_llm = []
    placeholder_pattern = re.compile(r'\{results\[(\d+)]}')

    for i, tool_json_str in enumerate(tool_calls):
        def replacer(match):
            index = int(match.group(1))
            if index < len(executed_results):
                return json.dumps(executed_results[index])[1:-1]
            return match.group(0)

        substituted_tool_json = placeholder_pattern.sub(replacer, tool_json_str)

        tool_output_json_str = _execute_tool(substituted_tool_json)
        observations_for_llm.append(tool_output_json_str)

        try:
            raw_result = json.loads(tool_output_json_str).get("result")
            executed_results.append(raw_result)
        except (json.JSONDecodeError, KeyError):
            # If parsing fails or 'result' key is missing, it's likely an error
            executed_results.append(json.loads(tool_output_json_str))

    return observations_for_llm

# --- Tool Execution Helper ---
def _execute_tool(tool_json_str: str) -> str:
    """Parses and executes a single tool call, returning the output as a JSON string."""
    try:
        tool_data = json.loads(tool_json_str)
        tool_name = tool_data.get("name")
        tool_args = tool_data.get("arguments", {})

        if not tool_name or tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Tool '{tool_name}' not found or name is invalid.")

        tool_to_execute = TOOL_REGISTRY[tool_name]
        tool_output = tool_to_execute(tool_args)
        tool_output_str = json.dumps({"result": tool_output})
        return tool_output_str

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        error_message = f"Error processing tool call: {e}"
        return json.dumps({"error": error_message})