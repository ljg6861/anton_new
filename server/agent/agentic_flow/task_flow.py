import asyncio
import datetime
import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional
from server.agent.tool_learning_store import tool_learning_store
from server.agent.agentic_flow.helpers_and_prompts import ADAPTIVE_PLANNING_PROMPT, ASSESSMENT_PROMPT, EXECUTOR_PROMPT, FINAL_MESSAGE_GENERATOR_PROMPT, PLAN_RESULT_EVALUATOR_PROMPT, REPLANNING_PROMPT_ADDENDUM, RESEARCHER_PROMPT, SIMPLE_PLANNER_PROMPT, call_model_server
from server.agent.config import MODEL_SERVER_URL, USER_ROLE
from server.agent.knowledge_store import KnowledgeStore
from server.agent.react.react_agent import ReActAgent
from server.agent.react.token_budget import TokenBudget
from server.agent.tools.tool_manager import tool_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _extract_domain_from_query(query: str) -> str:
    """
    Extract domain from user query using keyword matching and heuristics.
    
    Args:
        query: User query text
        
    Returns:
        Extracted domain name
    """
    query_lower = query.lower()
    
    # Domain mapping based on common keywords
    domain_keywords = {
        "chess": ["chess", "chess game", "board game", "strategy game"],
        "code_dev": ["code", "programming", "development", "software", "debug", "bug", "function", "class", "api"],
        "tool_dev": ["tool", "utility", "script", "automation", "cli", "command line"],
        "data_analysis": ["data", "analysis", "dataset", "csv", "statistics", "analytics", "visualization"],
        "web_dev": ["web", "website", "html", "css", "javascript", "frontend", "backend", "server"],
        "devops": ["deploy", "deployment", "docker", "kubernetes", "ci/cd", "pipeline", "infrastructure"],
        "research": ["research", "study", "investigate", "analyze", "explore", "documentation"],
        "file_management": ["file", "directory", "folder", "organize", "move", "copy", "delete"],
        "git_workflow": ["git", "commit", "branch", "merge", "pull request", "repository", "version control"]
    }
    
    # Check for explicit domain keywords
    for domain, keywords in domain_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return domain
    
    # Extract potential domain from specific patterns
    # Look for "in <domain>" or "for <domain>" patterns
    domain_patterns = [
        r"in (\w+)",
        r"for (\w+)",
        r"with (\w+)",
        r"using (\w+)"
    ]
    
    for pattern in domain_patterns:
        matches = re.findall(pattern, query_lower)
        if matches:
            candidate = matches[0]
            # Filter out common words that aren't domains
            if candidate not in ["the", "a", "an", "this", "that", "me", "you", "it", "is", "was", "are", "were"]:
                return candidate
    
    # Default domain for general queries
    return "general"

async def stream_step_content(text: str) -> AsyncGenerator[str, None]:
    """Stream text character by character with a typing effect for step content."""
    for char in text:
        yield f"<step_content>{char}</step_content>"
        await asyncio.sleep(0.02)  # Small delay for typing effect

async def _call_and_parse_model(chat_messages: List[Dict[str, str]], tools: List[Dict[str, Any]], expected_key: Optional[str] = None) -> Any:
    """
    Handles the common logic of calling the model, parsing the response,
    and handling potential errors.
    """
    raw_response = ""
    async for token in call_model_server(chat_messages, tools):
        raw_response += token

    cleaned_response = raw_response
    if "</think>" in cleaned_response:
        cleaned_response = cleaned_response.split("</think>")[-1].strip()
        logger.info("Stripped <think> tags from model response.")

    try:
        parsed_json = json.loads(cleaned_response)
        if expected_key in parsed_json:
            logger.info(f"Router successfully parsed JSON: {parsed_json}")
            return parsed_json[expected_key]
        elif not expected_key and parsed_json:
            return parsed_json
        else:
            logger.warning(f"JSON is valid but missing '{expected_key}' key: {parsed_json}")
            return None # Or raise an error
    except json.JSONDecodeError:
        # "Bump" the agent with a reminder on the next attempt
        reminder = "Your previous response was not valid JSON. You MUST respond with only a JSON object."
        chat_messages.append({"role": "assistant", "content": raw_response})
        chat_messages.append({"role": "user", "content": reminder})
        # This function could return None or raise an exception to handle the error state.
        return None 

async def initial_assessment(messages: List[Dict[str, str]], knowledge_store: Optional[KnowledgeStore] = None) -> str:
    """Assesses a task based on the provided messages."""
    tools = tool_manager.get_tool_schemas()
    logger.info("Assessing Task")
    
    # Extract domain from user query for episodic memory
    user_content = " ".join([msg.get("content", "") for msg in messages if msg.get("role") == "user"])
    
    # If knowledge store provided, start episodic run and get relevant past assessments
    episodic_context = ""
    if knowledge_store:
        # Try to determine domain from query
        domain = _extract_domain_from_query(user_content)
        knowledge_store.start_episodic_run(domain)
        
        # Get relevant past assessment experiences
        past_assessments = knowledge_store.search_episodes_by_query(
            f"assessment task evaluation {user_content[:100]}", 
            role="assessor", 
            limit=3
        )
        
        if past_assessments:
            episodic_context = "\n\nPast Assessment Experiences:\n"
            for i, episode in enumerate(past_assessments, 1):
                episodic_context += f"{i}. {episode.summary} (confidence: {episode.confidence:.2f})\n"
    
    chat_prompt = ASSESSMENT_PROMPT.replace('{tools}', json.dumps(tools)) + episodic_context
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages

    assessment = await _call_and_parse_model(chat_messages, [], "assessment")
    
    # Record assessment episode
    if knowledge_store and assessment:
        knowledge_store.record_episode(
            role="assessor",
            summary=f"Assessed task: {assessment[:100]}",
            tags=["assessment", "task_evaluation"],
            entities={"query": user_content[:200], "assessment": assessment},
            outcome={"status": "pass", "assessment_result": assessment},
            confidence=0.8
        )
    
    return assessment or ""

async def execute_researcher(messages: List[Dict[str, str]], knowledge_store: Optional[KnowledgeStore] = None) -> AsyncGenerator[Dict, None]:
    """Executes the relentless researcher - an LLM that analyzes scraped data comprehensively."""
    logger.info("Executing RELENTLESS researcher...")
    
    # Step 1: Collect initial data based on request type
    user_content = " ".join([msg.get("content", "") for msg in messages if msg.get("role") == "user"])
    
    # Get relevant past research experiences
    episodic_context = ""
    if knowledge_store:
        past_research = knowledge_store.search_episodes_by_query(
            f"research investigation data collection {user_content[:100]}", 
            role="researcher", 
            limit=2
        )
        
        if past_research:
            episodic_context = "\n\nPast Research Experiences:\n"
            for episode in past_research:
                episodic_context += f"- {episode.summary} (outcome: {episode.outcome.get('status', 'unknown')})\n"
    
    # Get available research tools for the prompt
    meta_tools = tool_manager.get_tools_by_names([
        "search_codebase",
        "web_search",
        "read_file"
    ])
    
    # Create researcher prompt with collected data
    research_prompt = RESEARCHER_PROMPT.replace(
        "{current_date}", datetime.datetime.now().isoformat()
    ) + episodic_context
    
    research_messages = [
        {"role": "system", "content": research_prompt},
        {"role": "user", "content": 'Perform research related to the following user query: ' + user_content}
    ]
    
    # Step 3: Call LLM researcher to analyze the data
    logger.info("Calling LLM researcher to analyze collected data...")
    temp_knowledge_store = knowledge_store or KnowledgeStore()
        
    budget = TokenBudget(total_budget=135000)

    react_agent = ReActAgent(
        api_base_url=MODEL_SERVER_URL,
        tools=meta_tools,
        knowledge_store=temp_knowledge_store,
        max_iterations=75,
        user_id='',
        token_budget=budget
    )
    
    research_result = ""
    async for chunk in react_agent.execute_react_loop_streaming(research_messages, logger):
        if chunk.startswith("<step>") or chunk.startswith("<step_content>"):
            yield chunk
        else:
            research_result += chunk
    
    # Record research episode
    research_successful = False
    try:
        research_data = json.loads(research_result)
        research_successful = True
        yield research_data
    except Exception as e:
        logger.error('Failed decoding research: %s', e)
        # Fallback: try to get the raw response and extract findings
        logger.info("Attempting to extract research findings from raw LLM response...")
        
        import re
        # Clean control characters that break JSON parsing
        cleaned_result = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', research_result)
        
        # Try parsing the cleaned result first
        try:
            research_data = json.loads(cleaned_result)
            research_successful = True
            yield research_data
        except:
            # If that fails, try to extract just the research_findings section
            json_match = re.search(r'\{.*?"research_findings".*?\}', cleaned_result, re.DOTALL)
            if json_match:
                try:
                    research_data = json.loads(json_match.group())
                    research_successful = True
                    yield research_data.get("research_findings", {"summary": research_result})
                except:
                    pass
            
            # Final fallback: return the raw result as summary
            yield {"summary": research_result}
    
    # Record research episode
    if knowledge_store:
        outcome_status = "pass" if research_successful else "partial"
        knowledge_store.record_episode(
            role="researcher",
            summary=f"Research conducted for query: {user_content[:100]}",
            tags=["research", "data_collection", "investigation"],
            entities={"query": user_content[:200], "tools_used": [tool["function"]["name"] for tool in meta_tools]},
            outcome={"status": outcome_status, "notes": f"Research result length: {len(research_result)}"},
            confidence=0.8 if research_successful else 0.5
        )

async def execute_planner(prompt: str, messages: List[Dict[str, str]], research_findings: Dict = None, knowledge_store: Optional[KnowledgeStore] = None) -> str:
    """Executes the planner to create a simple plan."""
    tools = tool_manager.get_tool_schemas()
    logger.info("Creating Plan")
    
    # Get relevant past planning experiences
    episodic_context = ""
    if knowledge_store:
        user_content = " ".join([msg.get("content", "") for msg in messages if msg.get("role") == "user"])
        past_plans = knowledge_store.search_episodes_by_query(
            f"planning strategy approach {user_content[:100]}", 
            role="planner", 
            limit=2
        )
        
        if past_plans:
            episodic_context = "\n\nPast Planning Experiences:\n"
            for episode in past_plans:
                outcome_status = episode.outcome.get("status", "unknown")
                episodic_context += f"- {episode.summary} (outcome: {outcome_status})\n"
    
    # Include research findings in the prompt
    research_context = json.dumps(research_findings, indent=2) if research_findings else "No prior research conducted."
    
    chat_prompt = prompt.replace(
        "{current_date}", datetime.datetime.now().isoformat()
    ).replace("{research_findings}", research_context).replace("{tools}", json.dumps(tools, indent=2)) + episodic_context
    
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages
    
    plan = await _call_and_parse_model(chat_messages, [], "plan")
    logger.info(f"Generated Plan: {plan}")
    
    # Record planning episode
    if knowledge_store and plan:
        plan_complexity = len(plan) if isinstance(plan, list) else 1
        knowledge_store.record_episode(
            role="planner",
            summary=f"Created plan with {plan_complexity} steps",
            tags=["planning", "strategy", "task_breakdown"],
            entities={"plan_steps": plan_complexity, "research_used": bool(research_findings)},
            outcome={"status": "pass", "plan_complexity": plan_complexity},
            confidence=0.8
        )
    
    return plan or ""

async def execute_executor(overall_goal: str, full_plan: str, previous_steps: List[Dict[str, str]], current_step: Dict[str, str], knowledge_store: Optional[KnowledgeStore] = None) -> AsyncGenerator[str, None]:
    """Executes the executor to carry out the plan."""
    logger.info("Executing Plan")
    
    # Get relevant past execution experiences
    episodic_context = ""
    if knowledge_store:
        step_description = current_step.get("thought", str(current_step))
        past_executions = knowledge_store.search_episodes_by_query(
            f"execution implementation {step_description[:100]}", 
            role="executor", 
            limit=2
        )
        
        if past_executions:
            episodic_context = "\n\nPast Execution Experiences:\n"
            for episode in past_executions:
                outcome_status = episode.outcome.get("status", "unknown")
                episodic_context += f"- {episode.summary} (outcome: {outcome_status})\n"
    
    chat_prompt = EXECUTOR_PROMPT.replace("{overall_goal}", overall_goal).replace("{full_plan}", json.dumps(full_plan)).replace("{previous_steps}", json.dumps(previous_steps)).replace("{current_step}", json.dumps(current_step)) + episodic_context
    chat_messages = [{"role": "system", "content": chat_prompt}]

    temp_knowledge_store = knowledge_store or KnowledgeStore()
    
    available_tools = tool_manager.get_tool_schemas()
    
    budget = TokenBudget(total_budget=135000)

    react_agent = ReActAgent(
        api_base_url=MODEL_SERVER_URL,
        tools=available_tools,
        knowledge_store=temp_knowledge_store,
        max_iterations=50,
        user_id='',
        token_budget=budget
    )

    result = ""
    execution_successful = True
    async for chunk in react_agent.execute_react_loop_streaming(chat_messages, logger):
        if chunk.startswith("<step>") or chunk.startswith("<step_content>"):
            yield chunk
        else:
            result += chunk
    
    # Check if execution was successful (simple heuristic)
    if "error" in result.lower() or "failed" in result.lower():
        execution_successful = False
    
    # Record execution episode
    if knowledge_store:
        step_name = current_step.get("thought", "Unknown step")
        outcome_status = "pass" if execution_successful else "fail"
        knowledge_store.record_episode(
            role="executor",
            summary=f"Executed step: {step_name[:100]}",
            tags=["execution", "implementation", "step_completion"],
            entities={"step": current_step, "result_length": len(result)},
            outcome={"status": outcome_status, "notes": result[:200] if result else "No result"},
            confidence=0.8 if execution_successful else 0.4
        )
    
    yield result or ""

async def execute_adaptive_planner(goal: str, original_plan: List[Dict], completed_steps: List[Dict], latest_result: str, messages: List[Dict[str, str]], research_findings: Dict = None) -> Dict:
    """Executes adaptive planning to determine if plan adjustments are needed."""
    tools = tool_manager.get_tool_schemas()
    logger.info("Executing adaptive planning...")
    
    # Include research findings in the prompt
    research_context = json.dumps(research_findings, indent=2) if research_findings else "No prior research conducted."
    
    chat_prompt = ADAPTIVE_PLANNING_PROMPT.replace(
        "{current_date}", datetime.datetime.now().isoformat()
    ).replace(
        "{user_goal}", goal
    ).replace(
        "{original_plan}", json.dumps(original_plan, indent=2)
    ).replace(
        "{completed_steps}", json.dumps(completed_steps, indent=2)
    ).replace(
        "{latest_result}", latest_result
    ).replace(
        "{research_findings}", research_context
    ).replace(
        "{tools}", json.dumps(tools, indent=2)
    )
    
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages
    
    result = await _call_and_parse_model(chat_messages, [], None)
    logger.info(f"Adaptive planning result: {result}")
    return result or {"action": "continue", "reasoning": "Failed to parse planning response", "plan": original_plan}

async def generate_final_user_message(goal: str, step_history: List[Dict], plan_summary: str) -> AsyncGenerator[str, None]:
    """Generate a natural user-facing message based on the completed execution."""
    logger.info("Generating final user-facing message...")
    
    # First stream the step content to show progress
    progress_msg = "Crafting personalized response..."
    async for content in stream_step_content(progress_msg):
        yield content
    
    chat_prompt = FINAL_MESSAGE_GENERATOR_PROMPT.replace(
        "{user_goal}", goal
    ).replace(
        "{execution_history}", json.dumps(step_history, indent=2)
    ).replace(
        "{plan_summary}", plan_summary
    )
    
    chat_messages = [{"role": "system", "content": chat_prompt}]
    
    # Now stream the actual response tokens
    async for token in call_model_server(chat_messages, []):
        yield token

async def evaluate_plan_success(overall_goal: str, full_plan: str, previous_steps: List[Dict[str, str]]) -> Dict:
    logger.info("Evaluating plan success...")
    chat_prompt = PLAN_RESULT_EVALUATOR_PROMPT.replace("{user_goal}", overall_goal).replace("{original_plan}", json.dumps(full_plan)).replace("{execution_results}", json.dumps(previous_steps))
    result = await _call_and_parse_model([{'role' : 'system', 'content' : chat_prompt}], [])
    logger.info(f"Plan evaluation result: {result}")
    return result

async def handle_task_route(messages: List[Dict[str, str]], goal: str, knowledge_store: Optional[KnowledgeStore] = None) -> AsyncGenerator[str, None]:
    """Complex task agent workflow with research phase and episodic memory."""
    logger.info("Executing task route...")

    # Initialize knowledge store if not provided
    if not knowledge_store:
        knowledge_store = KnowledgeStore()

    # Step 1: Assessment
    yield "<step>Assessing Task Requirements</step>"
    assessment = await initial_assessment(messages, knowledge_store)
    async for content in stream_step_content(f"Assessment result: {assessment}"):
        yield content
    
    research_findings = []
    if assessment != 'Sufficient':
        # Step 2: RELENTLESS RESEARCH PHASE
        yield "<step>Conducting Research</step>"
        logger.info("Starting RELENTLESS research phase...")
        
        async for chunk in execute_researcher(messages, knowledge_store):
            if isinstance(chunk, dict):
                # This is the final research result
                research_findings = chunk
            elif chunk.startswith("<step>") or chunk.startswith("<step_content>"):
                yield chunk
        
        logger.info(f"Research completed with findings: {research_findings}")
        async for content in stream_step_content("Research completed with findings"):
            yield content
    else:
        research_findings = []

    # Step 3: Planning with Research Context
    yield "<step>Generating Plan</step>"
    prompt = SIMPLE_PLANNER_PROMPT
    plan = await execute_planner(prompt, messages, research_findings, knowledge_store)
    plan_msg = f"Plan generated with {len(plan) if plan else 0} steps"
    async for content in stream_step_content(plan_msg):
        yield content
    
    # Step 4: Execute the plan
    async for result in execute_control_loop(plan, goal, messages, research_findings, knowledge_store):
        yield result

async def execute_control_loop(plan, goal, messages, research_findings=None, knowledge_store: Optional[KnowledgeStore] = None):
    step_history = []
    current_plan = plan
    step_index = 0
    
    def truncate_step_name(name: str, max_length: int = 30) -> str:
        """Truncate step name to max_length characters"""
        if len(name) <= max_length:
            return name
        return name[:max_length-3] + "..."
    
    while step_index < len(current_plan):
        step = current_plan[step_index]
        
        # Execute the current step
        step_name = f"Executing Step {step['step']}: {step.get('thought', 'Processing step')}"
        truncated_name = truncate_step_name(step_name)
        yield f"<step>{truncated_name}</step>"

        result = ""
        async for chunk in execute_executor(goal, current_plan, step_history, step, knowledge_store):
            if chunk.startswith("<step>") or chunk.startswith("<step_content>"):
                yield chunk
            else:
                result = chunk  # The final result comes as the last chunk
        
        step_history.append({"step": step['step'], "result": result})
        
        completion_msg = f"Step {step['step']} completed"
        async for content in stream_step_content(completion_msg):
            yield content
        step_index += 1
        
        # After each step (except the last), check if we need to adjust the plan
        if step_index < len(current_plan):
            yield f"<step>Evaluating Plan Progress</step>"
            logger.info(f"Checking if plan adjustment needed after step {step['step']}")
            adaptive_result = await execute_adaptive_planner(
                goal, current_plan, step_history, result, messages, research_findings
            )
            
            if adaptive_result.get("action") == "replan":
                logger.info(f"Replanning: {adaptive_result.get('reasoning')}")
                replan_msg = f"Replanning required: {adaptive_result.get('reasoning')}"
                async for content in stream_step_content(replan_msg):
                    yield content
                current_plan = adaptive_result.get("plan", current_plan)
                step_index = 0  # Start from beginning with new plan
            elif adaptive_result.get("action") == "adjust":
                logger.info(f"Adjusting plan: {adaptive_result.get('reasoning')}")
                adjust_msg = f"Adjusting plan: {adaptive_result.get('reasoning')}"
                async for content in stream_step_content(adjust_msg):
                    yield content
                # Replace remaining steps with adjusted ones
                adjusted_plan = adaptive_result.get("plan", [])
                current_plan = adjusted_plan
                step_index = 0  # Start from beginning with adjusted plan
            else:
                logger.info(f"Continuing with original plan: {adaptive_result.get('reasoning')}")
                continue_msg = f"Continuing with plan: {adaptive_result.get('reasoning')}"
                async for content in stream_step_content(continue_msg):
                    yield content
                # Continue with current plan as-is
    
    yield f"<step>Evaluating Final Results</step>"
    result = await evaluate_plan_success(goal, current_plan, step_history)
    if result['result'] == 'success':
        success_msg = "Task completed successfully"
        async for content in stream_step_content(success_msg):
            yield content
        
        # Record successful completion
        if knowledge_store:
            knowledge_store.record_episode(
                role="evaluator",
                summary=f"Task completed successfully: {goal[:100]}",
                tags=["task_completion", "success", "evaluation"],
                entities={"goal": goal, "steps_completed": len(step_history)},
                outcome={"status": "pass", "notes": "All steps completed successfully"},
                confidence=0.9
            )
        
        # Generate a natural user-facing message instead of yielding the raw summary
        yield f"<step>Generating Final Response</step>"
        async for token in generate_final_user_message(
            goal, 
            step_history, 
            step.get('args', {}).get('summary', 'Task completed successfully')
        ):
            yield token
        return
    else:
        logger.info("Replanning due to failure... " + result['reasoning'])
        failure_msg = f"Task failed, replanning: {result['reasoning']}"
        async for content in stream_step_content(failure_msg):
            yield content
        
        # Record failure
        if knowledge_store:
            knowledge_store.record_episode(
                role="evaluator",
                summary=f"Task failed: {result['reasoning'][:100]}",
                tags=["task_failure", "evaluation", "replanning"],
                entities={"goal": goal, "failure_reason": result['reasoning']},
                outcome={"status": "fail", "notes": result['reasoning']},
                confidence=0.7
            )
        
        yield f"<step>Replanning Strategy</step>"
        prompt_string = SIMPLE_PLANNER_PROMPT + REPLANNING_PROMPT_ADDENDUM

        prompt_string = prompt_string.replace("{user_goal}", goal)
        prompt_string = prompt_string.replace("{original_plan}", json.dumps(current_plan))
        prompt_string = prompt_string.replace("{failure_reasoning}", result['reasoning'])
        prompt_string = prompt_string.replace("{steps_taken}", str(json.dumps(step_history)))

        new_plan = await execute_planner(prompt_string, messages, research_findings, knowledge_store)
        new_plan_msg = "New plan created"
        async for content in stream_step_content(new_plan_msg):
            yield content
        async for token in execute_control_loop(new_plan, goal, messages, research_findings, knowledge_store):
            yield token 