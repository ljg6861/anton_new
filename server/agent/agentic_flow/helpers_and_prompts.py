import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List

import httpx
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_MODEL = "openai/gpt-oss-20b"

async def call_model_server(messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        request_payload = {
            "model": "anton",  # Use the served model name from vLLM
            "messages": messages,
            "temperature": 0.6,
            "stream": True,
            "tools": tools,
            "max_tokens": 9000
        }

        vllm_url = "http://localhost:8003"

        for attempt in range(3):
          try:
              async with httpx.AsyncClient(timeout=120.0) as client:
                  url = f"{vllm_url}/v1/chat/completions"
                  headers = {
                      "Content-Type": "application/json",
                      "Authorization": "Bearer anton-vllm-key"
                  }

                  # Start the streaming request
                  async with client.stream("POST", url, json=request_payload, headers=headers) as response:
                      response.raise_for_status()
                      
                      # Process each line of the stream
                      async for line in response.aiter_lines():
                          if line.startswith('data: '):
                              line = line[6:]

                          if line.strip() == "[DONE]":
                              break

                          if line.strip():
                              try:
                                  chunk_data = json.loads(line)
                                  choices = chunk_data.get("choices", [])
                                  if choices:
                                      delta = choices[0].get("delta", {})
                                      content = delta.get("content", "")
                                      if content:
                                          # Yields each chunk to the caller
                                          yield content

                              except json.JSONDecodeError as e:
                                  logger.warning(f"Failed to parse streaming chunk: {line} - {e}")
                                  continue

              # If the streaming completes successfully, break the retry loop
              return

          except httpx.RequestError as e:
              # Handle transient errors that would warrant a retry
              logger.info(f"Request error on attempt {attempt + 1}: {e}")
              if attempt < 2:  # Note: range(3) means 0, 1, 2. So we retry if attempt is less than 2.
                  delay = 2 ** attempt
                  logger.info(f"Waiting {delay} seconds before retrying...")
                  await asyncio.sleep(delay)
              else:
                  # If we've exhausted all attempts, re-raise the error.
                  raise


async def call_model_for_summarization(conversation_history: str) -> str:
    """Call the model server to get an intelligent summarization of conversation history"""
    messages = [
        {"role": "system", "content": "You are an expert at intelligently summarizing ReAct agent conversations while preserving critical context."},
        {"role": "user", "content": REACT_SUMMARIZATION_PROMPT.format(conversation_history=conversation_history)}
    ]
    
    request_payload = {
        "model": "anton",
        "messages": messages,
        "temperature": 0.1,  # Low temperature for consistent summaries
        "stream": False,
        "max_tokens": 2000
    }

    vllm_url = "http://localhost:8003"
    
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            url = f"{vllm_url}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer anton-vllm-key"
            }
            
            response = await client.post(url, json=request_payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip()
            
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return f"[Summary failed: {str(e)}]"


SIMPLE_PLANNER_PROMPT = """
Create a step-by-step plan using available tools. Return JSON only. You are not permitted to use any tools. You may only reference them in your plan.

Research findings: {research_findings}

Available tools:
{tools}

All research has been provided to you. Do not do your own research.

Output format:
{
  "plan": [
    {
      "step": 1,
      "thought": "Why this step is needed",
      "tool": "tool_name",
      "args": {"key": "value"}
    }
  ]
}

Always end with final_answer tool.
"""

REPLANNING_PROMPT_ADDENDUM = """
Your previous plan failed. Create a new plan to achieve the original goal.

Failed plan: {original_plan}
Failure reason: {failure_reasoning}
Original goal: {user_goal}

Create a different approach.
"""

EXECUTOR_PROMPT = """
Execute the current step from the plan.

Goal: {overall_goal}
Current step: {current_step}
Previous results: {previous_steps}
Full plan: {full_plan}

Use the available tools to complete this step. Ensure that you ONLY work on the current step. You may mark your progress as done by not calling any other tools. 
"""

ASSESSMENT_PROMPT = """
Assess if the core tools can handle this request.

Return JSON only:
{"assessment": "Sufficient"} - if core tools can handle it
{"assessment": "Requires_Discovery"} - if more tools needed

Examples:
- "Write python code" -> Sufficient
- "Book a flight" -> Requires_Discovery
"""

RESEARCHER_PROMPT = """
As a Researcher, your sole purpose is to gather information to fulfill a user's request. Your task is to turn the user's query into a research plan and execute it. You have access to search and file reading tools.

CRITICAL RULES:
1.  **Focus on Research Only**: Do not attempt to execute, create, or modify files. Your output must be a JSON object containing your research findings.
2.  **Be Efficient**: Stop researching once you have sufficient information. Avoid repeating searches with the same terms.
3.  **Prioritize Abstract Contracts**: If you find an abstract base class (ABC) that defines a contract for components (like tools), its requirements MUST be treated as the primary source of truth. If you find other concrete examples that do NOT follow this contract, they should be considered legacy and their patterns should NOT be copied for new components.
4.  **Synthesize and Exhaust**: Synthesize all findings into actionable guidance. For any class that must be implemented, you MUST provide an exhaustive, itemized list of all abstract methods from its parent class that need to be overridden, including their full signatures (arguments and return types).
5.  **Validate and Reuse Existing Components**: Before writing code, you MUST validate the full context of the system. This includes:
    * **Validating Paths**: Ensure all file paths for reading or writing are correct.
    * **Inspecting Imports**: When a class or function signature references another type (e.g., `-> ToolMetadata`), you MUST inspect the file where the base class is defined (`base_tool.py` in this case) to find and import that type.
    * **Forbidding Redefinition**: You are strictly forbidden from redefining classes, functions, or data structures that already exist within the project. Always prefer importing over redefining.
6.  **File Path References**: If you reference a file path in your output, it MUST be relative to the project root.

Research Process:
1.  Begin with broad searches to understand the system architecture.
2.  Follow up with specific searches to find exact implementation details.
3.  Synthesize all findings into the final JSON output.

Return your findings in the following JSON format:
{
    "research_findings": {
        "summary": "Overview of findings",
        "system_architecture": "How systems work", 
        "tool_requirements": "Tool structure and location requirements",
        "api_specifications": "API details if relevant",
        "implementation_guidance": "Step-by-step guidance",
        "dependencies": "Required dependencies",
        "examples": "Code examples and patterns"
    }
}
"""

PLAN_RESULT_EVALUATOR_PROMPT = """
Evaluate if the plan execution achieved the user's goal.

User goal: {user_goal}
Plan: {original_plan}
Results: {execution_results}

Return JSON only:
{"result": "success"} - if goal was achieved
{"result": "failure", "reasoning": "Why it failed and how to fix it"} - if goal was not achieved
"""

REACT_SUMMARIZATION_PROMPT = """
Intelligently summarize this ReAct conversation history to preserve the most important information.

Combine related findings, decisions, and context. Focus on:
- Key user requests and goals
- Important tool execution results and patterns
- Critical decisions and their reasoning
- Essential context that affects future actions
- Outstanding issues or incomplete tasks

Merge similar information rather than repeating it. Be concise but preserve all actionable insights.

Return as a single coherent summary in chronological order. Ensure you preserve tool calls and arguments, but please summarize the tool calls results. 

Original conversation:
{conversation_history}
"""