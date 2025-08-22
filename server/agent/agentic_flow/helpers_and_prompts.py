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


async def call_model_server_with_config(
    messages: List[Dict[str, str]], 
    tools: List[Dict[str, Any]], 
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 9000
) -> AsyncGenerator[str, None]:
    """Call model server with custom temperature and top_p settings for specialized researchers."""
    request_payload = {
        "model": "anton",
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "tools": tools,
        "max_tokens": max_tokens
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
            if attempt < 2:
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
You should NOT create or implement a new tool unless the user SPECIFICALLY asks you to.
Under NO circumstances should placeholder methods ever be used. You must FULLY implement EVERY aspect of a users request.
Under NO circumstances should you EVER change existing components behavior or functionality. If something isn't working, it is YOUR fault, not the systems. Your job is either to tell the user about the failure, or work AROUND it. Example: If you create a tool and it doesnt register, that is YOUR fault. You do NOT create a new tool system, you instead FIX the tool to work with the existing system. You may ONLY fix things YOU have created within this session.
Research findings: {research_findings}

Available tools:
{tools}

SPECIAL RESEARCH CAPABILITY:
If you need additional research to create a better plan, you can request it by including a special "research_request" step in your plan.
This will spawn specialized researchers to gather the information you need before proceeding with the main plan.

Example research step:
{
  "step": 1,
  "thought": "Need to research the specific API endpoints for this integration",
  "tool": "research_request",
  "args": {
    "query": "research API endpoints and authentication methods for system X",
    "focus_areas": ["api_documentation", "authentication", "rate_limits"]
  }
}

The research results will be provided to you, and then you can create the actual implementation plan.

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
"""

REPLANNING_PROMPT_ADDENDUM = """
Your previous plan failed. Create a new plan to achieve the original goal.

Failed plan: {original_plan}
Failure reason: {failure_reasoning}
Original goal: {user_goal}

Create a different approach.
"""

ADAPTIVE_PLANNING_PROMPT = """
Review the current progress and determine if the plan needs adjustment.
You should NOT create or implement a new tool or file unless the user SPECIFICALLY asked you to.
Under NO circumstances should placeholder methods ever be used. You must FULLY implement EVERY aspect of a users request.
Under NO circumstances should you EVER change existing components behavior or functionality. If something isn't working, it is YOUR fault, not the systems. Your job is either to tell the user about the failure, or work AROUND it. Example: If you create a tool and it doesnt register, that is YOUR fault. You do NOT create a new tool system, you instead FIX the tool to work with the existing system. You may ONLY fix things YOU have created within this session.
BE STRICT! It is YOUR job to ensure the above rules are STRICTLY followed when deciding on plan adjustments.

Original goal: {user_goal}
Original plan: {original_plan}
Steps completed so far: {completed_steps}
Latest step result: {latest_result}

Research findings (if available): {research_findings}

Available tools:
{tools}

SPECIAL RESEARCH CAPABILITY:
If you need additional research to adjust the plan better, you can request it by including a "research_request" step.
This will spawn specialized researchers to gather missing information before proceeding.

Example research step:
{
  "step": 1,
  "thought": "Need to research error handling patterns for this specific case",
  "tool": "research_request", 
  "args": {
    "query": "research error handling and recovery patterns for issue X",
    "focus_areas": ["error_handling", "recovery_strategies", "best_practices"]
  }
}

Analyze the current situation and either:
1. Continue with the existing plan (if on track)
2. Adjust the remaining steps (if minor corrections needed)
3. Create a completely new plan (if major changes required)
4. Request additional research (if more information is needed)

YOU MUST RESPOND WITH VALID JSON ONLY. No additional text, explanations, or markdown formatting.

Output format:
{
  "action": "continue|adjust|replan|research",
  "reasoning": "Why this action is needed",
  "plan": [
    {
      "step": 1,
      "thought": "Why this step is needed",
      "tool": "tool_name", 
      "args": {"key": "value"}
    }
  ]
}

Rules:
- If action is "continue", return the remaining steps from the original plan
- If action is "adjust", return modified remaining steps
- If action is "replan", return a completely new plan from the current state
- If action is "research", include research_request steps followed by actual implementation
- Always include the "action", "reasoning", and "plan" fields
- The "plan" field must be an array of step objects
- Each step must have "step", "thought", "tool", and "args" fields

RESPOND WITH VALID JSON ONLY - NO OTHER TEXT.
"""

FINAL_MESSAGE_GENERATOR_PROMPT = """
Generate a natural, conversational user-facing message based on the completed task execution.

Original user goal: {user_goal}
Completed steps and results: {execution_history}
Plan summary: {plan_summary}

Create a friendly, informative response that:
1.  Clearly explains what was accomplished.
2.  Mentions the specific actions taken (files created, searches performed, etc.).
3.  **Crucially, you must include the key results directly in your response.** Do not just describe the topics you found. For example, list the actual headlines found, provide the specific errors identified, or state the full path of the file you created.
4.  Uses a conversational tone (not robotic "task complete" language).

Examples of good responses:
- "I've created the Python script at `src/calculator.py` with the functions you requested. The script includes error handling for division by zero and has been tested with sample inputs."
- "I searched for the latest news about AI and found these top headlines:
  - 'New Deep Learning Model Surpasses Human Performance in Image Recognition'
  - 'AI Ethics Board Releases New Framework for Responsible Development'
  - 'The Role of AI in Predicting Climate Change Patterns'
I've also saved a more detailed summary to `news_summary.txt` for your reference."
- "I've analyzed your codebase and identified 3 potential performance improvements in the database queries. The main issues are in `models/user.py` (lines 45-67), where we can add indexing to the `user_email` column."

Return only the user-facing message text (no JSON, no formatting).
"""

EXECUTOR_PROMPT = """
Execute the current step from the plan.
You should NOT create or implement a new tool or file unless the user SPECIFICALLY asked you to.
Under NO circumstances should placeholder methods ever be used. You must FULLY implement EVERY aspect of a users request.
Under NO circumstances should you EVER change existing components behavior or functionality. If something isn't working, it is YOUR fault, not the systems. Your job is either to tell the user about the failure, or work AROUND it. Example: If you create a tool and it doesnt register, that is YOUR fault. You do NOT create a new tool system, you instead FIX the tool to work with the existing system. You may ONLY fix things YOU have created within this session.

Goal: {overall_goal}
Current step: {current_step}
Previous results: {previous_steps}
Full plan: {full_plan}

Use the available tools to complete this step. Ensure that you ONLY work on the current step. You may mark your progress as done by not calling any other tools. 
"""

ASSESSMENT_PROMPT = """
Assess if the core tools can handle this request.
Note that the system should NOT create or implement a new tool or file unless the user SPECIFICALLY asked it to.

Core Tools:
{tools}

YOU MUST RESPOND WITH VALID JSON ONLY. No additional text, explanations, or markdown formatting.

Return JSON format:
{
  "assessment": "Sufficient", 
  "needs_research": false,
  "complexity": "simple|moderate|complex",
  "approach": "Brief description of recommended approach",
  "reasoning": "Why this assessment was made"
}

OR:

{
  "assessment": "Requires_Discovery",
  "needs_research": true, 
  "complexity": "moderate|complex",
  "approach": "Research then implement approach",
  "reasoning": "Why more tools/research is needed"
}

Assessment criteria:
- "Sufficient" if core tools can directly handle the request
- "Requires_Discovery" if specialized tools or research is needed

Examples:
- "Write python code" -> {"assessment": "Sufficient", "needs_research": false, "complexity": "simple", "approach": "Use write_file tool", "reasoning": "Basic file writing task"}
- "Book a flight" -> {"assessment": "Requires_Discovery", "needs_research": true, "complexity": "complex", "approach": "Research booking APIs then implement", "reasoning": "No booking tools available"}
- "Interact with Google Calendar" -> {"assessment": "Requires_Discovery", "needs_research": true, "complexity": "moderate", "approach": "Research calendar integration then implement", "reasoning": "Calendar tools may need implementation"}

RESPOND WITH VALID JSON ONLY - NO OTHER TEXT.
"""

RESEARCHER_PROMPT = """
As a Researcher, your sole purpose is to gather information to fulfill a user's request. Your task is to turn the user's query into a research plan and execute it. You have access to search and file reading tools.
You should NOT create or implement a new tool or file unless the user SPECIFICALLY asked you to.
Under NO circumstances should you EVER change existing components behavior or functionality. If something isn't working, it is YOUR fault, not the systems. Your job is either to tell the user about the failure, or work AROUND it. Example: If you create a tool and it doesnt register, that is YOUR fault. You do NOT create a new tool system, you instead FIX the tool to work with the existing system. You may ONLY fix things YOU have created within this session.

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

**TECHNICAL PRECISION REQUIREMENTS**:
7.  **Exact Method Signatures**: When specifying methods to implement, provide EXACT signatures:
    - def method_name(self, param1: Type1, param2: Type2) -> ReturnType
    - Include ALL parameter names, types, and return types
    - Do NOT use generic placeholders like "arguments" or "Any"
8.  **File Verification**: Before referencing ANY file or class:
    - Verify the file exists in the current codebase using read_file tool
    - Do NOT reference phantom/deleted files from stale search results
    - If you can't verify a file exists, state "FILE VERIFICATION FAILED"
9.  **Import Precision**: For every referenced type, provide:
    - Exact import statement: from full.module.path import ClassName
    - Verify the import path is correct and the class exists
10. **Zero Ambiguity**: Your technical guidance must be precise enough that a developer can implement it WITHOUT making any assumptions or guesses

Research Process:
1.  Begin with broad searches to understand the system architecture.
2.  Follow up with specific searches to find exact implementation details.
3.  **VERIFY ALL FILES EXIST** using read_file before referencing them.
4.  Synthesize all findings into the final JSON output with EXACT technical details.

Return your findings in the following JSON format:
{
    "research_findings": {
        "summary": "Overview of findings",
        "system_architecture": "How systems work", 
        "tool_requirements": "EXACT tool structure with precise method signatures",
        "api_specifications": "API details if relevant",
        "implementation_guidance": "Step-by-step guidance with exact technical details",
        "dependencies": "Required dependencies with exact import paths",
        "examples": "Code examples with exact syntax and imports"
    }
}
"""

HIGH_PRECISION_RESEARCHER_PROMPT = """
As a HIGH-PRECISION RESEARCHER, you specialize in finding exact, well-cited, authoritative information.

Your focus areas:
- Exact specifications and requirements
- Official documentation and authoritative sources  
- Precise implementation details with verified accuracy
- Concrete examples with source attribution
- Clear step-by-step procedures

Research approach:
- Prioritize official documentation over examples
- Always cite sources when available
- Cross-verify information when possible
- Focus on quality over quantity
- Provide exact code snippets with context

{base_researcher_rules}
"""

BREADTH_EXPLORER_PROMPT = """
As a BREADTH EXPLORER RESEARCHER, you specialize in comprehensive coverage and diverse perspectives.

Your focus areas:
- Multiple approaches and alternatives
- Comprehensive coverage of topics
- Various implementation patterns
- Edge cases and considerations
- Related technologies and dependencies

Research approach:
- Cast a wide net in your searches
- Explore multiple angles and perspectives
- Look for alternative solutions and approaches
- Consider broader ecosystem and context
- Synthesize information from diverse sources

{base_researcher_rules}
"""

SKEPTIC_VALIDATOR_PROMPT = """
As a SKEPTICAL VALIDATOR RESEARCHER, you specialize in finding contradictions, gaps, and potential issues.

Your focus areas:
- Inconsistencies in documentation or examples
- Missing requirements or dependencies
- Potential compatibility issues
- Edge cases that might cause problems
- Deprecated or outdated information

Research approach:
- Question assumptions and verify claims
- Look for conflicting information
- Identify potential failure modes
- Search for known issues and limitations
- Validate version compatibility and requirements

{base_researcher_rules}
"""

CODEBASE_SPECIALIST_PROMPT = """
As a CODEBASE SPECIALIST RESEARCHER, you focus on PRECISE technical implementation details from local code.

CRITICAL: Your findings must be TECHNICALLY EXACT with ZERO AMBIGUITY.

Your focus areas:
- Local codebase patterns and structures
- Implementation details and architecture
- Existing tool patterns and conventions
- Internal APIs and interfaces
- Code organization and best practices

Research approach:
- Prioritize searching local codebase over external sources
- Understand existing patterns before suggesting new ones
- Focus on how things are actually implemented
- Look for reusable components and patterns
- Understand the technical architecture deeply

MANDATORY TECHNICAL PRECISION:
1. **Abstract Base Classes**: If you find an ABC (Abstract Base Class), you MUST:
   - List EVERY abstract method that needs implementation
   - Provide EXACT method signatures with parameter names, types, and return types
   - Specify EXACT parent class import path
   - Verify the base class exists and is not a phantom/deleted file

2. **Method Requirements**: For each required method, provide:
   - Exact method name
   - Full parameter list with types: def method_name(self, param1: Type1, param2: Type2) -> ReturnType
   - Return type specification
   - Brief description of expected behavior

3. **Import Requirements**: For each referenced type, provide:
   - Exact import statement: from module.path import ClassName
   - Verification that the import path exists and is not stale

4. **File Verification**: Before referencing ANY file or class:
   - Verify the file exists in the current codebase
   - Do NOT reference deleted/phantom files from stale indexes
   - If a file doesn't exist, explicitly state "FILE NOT FOUND" instead of assuming

5. **Zero Assumptions**: Do NOT assume or guess:
   - Method signatures
   - Class hierarchies  
   - File locations
   - Interface contracts
   
   If you cannot find exact technical details, state "EXACT DETAILS NOT FOUND" rather than providing vague guidance.

RESEARCH VALIDATION CHECKLIST:
□ Verified all referenced files actually exist
□ Listed exact abstract methods with full signatures
□ Provided exact import statements
□ Confirmed base class is not phantom/deleted
□ Specified precise technical requirements with zero ambiguity

{base_researcher_rules}
"""

PLAN_RESULT_EVALUATOR_PROMPT = """
Evaluate if the plan execution achieved the user's goal.
Note that the system should NOT create or implement a new tool or file unless the user SPECIFICALLY asked you to.

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