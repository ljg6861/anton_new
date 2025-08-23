import asyncio
import datetime
import json
import logging
import re
import time
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
    """Stream text with detailed step content for user visibility."""
    # Stream all at once instead of character by character for better readability
    yield f"<step_content>{text}</step_content>"
    await asyncio.sleep(0.1)  # Brief pause for better UX

async def _call_and_parse_model(chat_messages: List[Dict[str, str]], tools: List[Dict[str, Any]], expected_key: Optional[str] = None) -> Any:
    """
    Handles the common logic of calling the model, parsing the response,
    and handling potential errors.
    """
    raw_response = ""
    async for token in call_model_server(chat_messages, tools):
        raw_response += token

    logger.info(f"🔧 RAW MODEL RESPONSE: {raw_response[:500]}...")

    cleaned_response = raw_response
    if "</think>" in cleaned_response:
        cleaned_response = cleaned_response.split("</think>")[-1].strip()
        logger.info("Stripped <think> tags from model response.")

    # Try to extract JSON from the response if it's wrapped in other text
    json_patterns = [
        r'\{.*\}',  # Match JSON object
        r'\[.*\]'   # Match JSON array
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, cleaned_response, re.DOTALL)
        if matches:
            # Try the largest match first (most complete JSON)
            for match in sorted(matches, key=len, reverse=True):
                try:
                    parsed_json = json.loads(match)
                    logger.info(f"✅ SUCCESSFULLY PARSED JSON from pattern: {parsed_json}")
                    
                    if expected_key and expected_key in parsed_json:
                        return parsed_json[expected_key]
                    elif not expected_key and parsed_json:
                        return parsed_json
                    elif expected_key:
                        logger.warning(f"JSON valid but missing '{expected_key}' key: {parsed_json}")
                        return None
                    
                except json.JSONDecodeError:
                    continue

    # If no JSON patterns worked, try the original approach
    try:
        parsed_json = json.loads(cleaned_response)
        if expected_key in parsed_json:
            return parsed_json[expected_key]
        elif not expected_key and parsed_json:
            return parsed_json
        else:
            logger.warning(f"JSON is valid but missing '{expected_key}' key: {parsed_json}")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON DECODE ERROR: {e}")
        logger.error(f"   Cleaned response: {cleaned_response[:200]}...")
        
        # "Bump" the agent with a reminder on the next attempt
        reminder = "Your previous response was not valid JSON. You MUST respond with only a JSON object."
        chat_messages.append({"role": "assistant", "content": raw_response})
        chat_messages.append({"role": "user", "content": reminder})
        return None 

async def initial_assessment(messages: List[Dict[str, str]], knowledge_store: Optional[KnowledgeStore] = None) -> str:
    """Assesses a task based on the provided messages."""
    tools = tool_manager.get_tool_schemas()
    user_content = " ".join([msg.get("content", "") for msg in messages if msg.get("role") == "user"])
    
    logger.info(f"⚡ INITIAL ASSESSMENT STARTING")
    logger.info(f"   User request: {user_content[:100]}...")
    logger.info(f"   Available tools: {len(tools)}")
    logger.info(f"   Knowledge store: {'available' if knowledge_store else 'none'}")
    
    # If knowledge store provided, start episodic run and get relevant past assessments
    episodic_context = ""
    semantic_context = ""
    if knowledge_store:
        # Try to determine domain from query
        domain = _extract_domain_from_query(user_content)
        knowledge_store.start_episodic_run(domain)
        logger.info(f"   Domain identified: {domain}")
        
        # Get relevant past assessment experiences (episodic memory)
        past_assessments = knowledge_store.search_episodes_by_query(
            f"assessment task evaluation {user_content[:100]}", 
            role="assessor", 
            limit=3
        )
        
        if past_assessments:
            episodic_context = "\n\nPast Assessment Experiences:\n"
            for i, episode in enumerate(past_assessments, 1):
                episodic_context += f"{i}. {episode.summary} (confidence: {episode.confidence:.2f})\n"
            logger.info(f"   Found {len(past_assessments)} past assessment episodes")
        
        # Get relevant semantic facts for this domain
        semantic_context = await knowledge_store.build_semantic_context(
            f"assessment task {user_content[:100]}", 
            max_facts=3
        )
        if semantic_context:
            logger.info(f"   Found semantic context: {len(semantic_context)} chars")
    
    full_context = episodic_context + "\n" + semantic_context
    chat_prompt = ASSESSMENT_PROMPT.replace('{tools}', json.dumps(tools)) + full_context
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages

    # Try up to 2 times to get valid JSON assessment
    assessment = None
    for attempt in range(2):
        logger.info(f"🔍 ASSESSMENT ATTEMPT {attempt + 1}/2")
        
        result = await _call_and_parse_model(chat_messages, [], None)  # Don't expect specific key
        
        if result and isinstance(result, dict):
            # Validate the result has required fields
            if "assessment" in result and "needs_research" in result:
                assessment = result
                assessment_value = assessment.get("assessment", "unknown")
                complexity = assessment.get("complexity", "unknown")
                approach = assessment.get("approach", "unknown")
                needs_research = assessment.get("needs_research", False)
                reasoning = assessment.get("reasoning", "No reasoning provided")
                
                logger.info(f"📋 ASSESSMENT SUCCESS:")
                logger.info(f"   Assessment: {assessment_value}")
                logger.info(f"   Complexity: {complexity}")
                logger.info(f"   Approach: {approach}")
                logger.info(f"   Needs research: {'✅ YES' if needs_research else '❌ NO'}")
                logger.info(f"   Reasoning: {reasoning[:100]}...")
                break
            else:
                logger.warning(f"⚠️ ASSESSMENT INCOMPLETE - missing required fields: {result}")
                if attempt == 0:  # Only retry once
                    chat_messages.append({
                        "role": "user", 
                        "content": "Your response must include 'assessment', 'needs_research', 'complexity', 'approach', and 'reasoning' fields. Please provide a complete JSON response."
                    })
        else:
            logger.warning(f"❌ ASSESSMENT FAILED - attempt {attempt + 1}, result: {result}")
            if attempt == 0:  # Only retry once
                chat_messages.append({
                    "role": "user", 
                    "content": "Please respond with valid JSON only. Include assessment, needs_research, complexity, approach, and reasoning fields."
                })
    
    if not assessment:
        logger.warning("❌ ASSESSMENT FAILED - using fallback after all attempts")
        # Provide fallback assessment
        assessment = {
            "assessment": "Requires_Discovery",
            "needs_research": True,
            "complexity": "moderate",
            "approach": "Research then implement",
            "reasoning": "Failed to parse assessment, defaulting to research mode"
        }
    
    # Record assessment episode
    if knowledge_store and assessment:
        knowledge_store.record_episode(
            role="assessor",
            summary=f"Assessed task: {assessment.get('assessment', 'unknown')} - {assessment.get('reasoning', '')[:50]}",
            tags=["assessment", "task_evaluation"],
            entities={"query": user_content[:200], "assessment": assessment},
            outcome={"status": "pass", "assessment_result": assessment},
            confidence=0.8
        )
    
    return assessment or ""

async def execute_researcher(messages: List[Dict[str, str]], knowledge_store: Optional[KnowledgeStore] = None) -> AsyncGenerator[Dict, None]:
    """
    Executes multi-instance researcher system with diverse search strategies.
    Fan-out approach with specialized researchers for comprehensive information gathering.
    """
    user_content = " ".join([msg.get("content", "") for msg in messages if msg.get("role") == "user"])
    
    logger.info(f"🔬 MULTI-RESEARCHER SYSTEM STARTING")
    logger.info(f"   User request: {user_content[:100]}...")
    logger.info(f"   Knowledge store: {'available' if knowledge_store else 'none'}")
    
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
            logger.info(f"   Found {len(past_research)} past research episodes")
    
    # Get available research tools
    meta_tools = tool_manager.get_tools_by_names([
        "search_codebase",
        "web_search", 
        "fetch_web_content",
        "read_file"
    ])
    
    logger.info(f"   Available research tools: {[tool['function']['name'] for tool in meta_tools]}")
    
    # Fan-out multiple researchers with different specializations
    researcher_configs = [
        {
            "name": "high_precision",
            "prompt_role": "High-Precision Researcher - Focus on exact, well-cited information with strict verification",
            "temperature": 0.1,
            "top_p": 0.3,
            "search_strategy": "precise_citations",
            "max_iterations": 15
        },
        {
            "name": "breadth_explorer", 
            "prompt_role": "Breadth Explorer - Cast a wide net to find diverse perspectives and comprehensive coverage",
            "temperature": 0.7,
            "top_p": 0.95,
            "search_strategy": "broad_coverage",
            "max_iterations": 20
        },
        {
            "name": "skeptic_validator",
            "prompt_role": "Skeptical Validator - Find contradictions, gaps, edge cases, and potential issues",
            "temperature": 0.2,
            "top_p": 0.5,
            "search_strategy": "contradiction_finding",
            "max_iterations": 12
        },
        {
            "name": "codebase_specialist",
            "prompt_role": "Codebase Specialist - Deep dive into local code, implementation patterns, and technical details",
            "temperature": 0.3,
            "top_p": 0.4,
            "search_strategy": "local_code_focus",
            "max_iterations": 18
        }
    ]
    
    # Execute researchers in parallel
    logger.info(f"🚀 LAUNCHING {len(researcher_configs)} specialized researchers")
    for config in researcher_configs:
        logger.info(f"   • {config['name']}: {config['prompt_role'][:50]}...")
    
    research_tasks = []
    
    for config in researcher_configs:
        task = asyncio.create_task(
            _execute_single_researcher(
                config, user_content, episodic_context, meta_tools, knowledge_store
            )
        )
        research_tasks.append((config["name"], task))
    
    # Collect results from all researchers
    researcher_results = []
    for name, task in research_tasks:
        try:
            result = await task
            if result:
                researcher_results.append({
                    "researcher": name,
                    "findings": result,
                    "confidence": result.get("confidence", 0.5)
                })
                logger.info(f"✅ RESEARCHER {name} completed with confidence {result.get('confidence', 0.5):.2f}")
                
                # Stream progress updates
                async for content in stream_step_content(f"✓ {name.replace('_', ' ').title()} researcher completed\n"):
                    yield content
            else:
                logger.warning(f"❌ RESEARCHER {name} returned no results")
        except Exception as e:
            logger.error(f"💥 RESEARCHER {name} failed: {str(e)}")
            async for content in stream_step_content(f"⚠ {name.replace('_', ' ').title()} researcher encountered issues"):
                yield content
    
    logger.info(f"📊 RESEARCH COMPLETE: {len(researcher_results)}/{len(researcher_configs)} researchers succeeded")
    
    # Aggregate and synthesize results
    logger.info("🔄 AGGREGATING research findings...")
    async for content in stream_step_content("Synthesizing findings from multiple researchers..."):
        yield content
    
    aggregated_result = await _aggregate_research_findings(researcher_results, user_content)
    
    # Record research episode
    if knowledge_store:
        outcome_status = "pass" if aggregated_result else "partial"
        knowledge_store.record_episode(
            role="researcher",
            summary=f"Multi-researcher investigation: {user_content[:100]}",
            tags=["research", "multi_instance", "comprehensive_analysis"],
            entities={
                "query": user_content[:200], 
                "researchers_used": len(researcher_results),
                "tools_used": [tool["function"]["name"] for tool in meta_tools]
            },
            outcome={
                "status": outcome_status, 
                "researchers_completed": len(researcher_results),
                "total_researchers": len(researcher_configs)
            },
            confidence=0.9 if len(researcher_results) >= 3 else 0.6
        )
    
    yield aggregated_result

async def _execute_single_researcher(
    config: Dict, 
    user_content: str, 
    episodic_context: str, 
    meta_tools: List[Dict], 
    knowledge_store: Optional[KnowledgeStore]
) -> Dict:
    """Execute a single researcher with specific configuration."""
    from server.agent.agentic_flow.helpers_and_prompts import (
        RESEARCHER_PROMPT, HIGH_PRECISION_RESEARCHER_PROMPT, 
        BREADTH_EXPLORER_PROMPT, SKEPTIC_VALIDATOR_PROMPT, 
        CODEBASE_SPECIALIST_PROMPT
    )
    
    try:
        # Select specialized prompt based on researcher type
        base_rules = RESEARCHER_PROMPT
        
        if config["name"] == "high_precision":
            specialized_prompt = HIGH_PRECISION_RESEARCHER_PROMPT.replace("{base_researcher_rules}", base_rules)
        elif config["name"] == "breadth_explorer":
            specialized_prompt = BREADTH_EXPLORER_PROMPT.replace("{base_researcher_rules}", base_rules)
        elif config["name"] == "skeptic_validator":
            specialized_prompt = SKEPTIC_VALIDATOR_PROMPT.replace("{base_researcher_rules}", base_rules)
        elif config["name"] == "codebase_specialist":
            specialized_prompt = CODEBASE_SPECIALIST_PROMPT.replace("{base_researcher_rules}", base_rules)
        else:
            specialized_prompt = base_rules
        
        specialized_prompt += episodic_context
        
        research_messages = [
            {"role": "system", "content": specialized_prompt},
            {"role": "user", "content": f'Perform research related to the following user query: {user_content}'}
        ]
        
        temp_knowledge_store = knowledge_store or KnowledgeStore()
        budget = TokenBudget(total_budget=45000)  # Smaller budget per researcher
        
        # Create researcher agent with specialized settings
        react_agent = ReActAgent(
            api_base_url=MODEL_SERVER_URL,
            tools=meta_tools,
            knowledge_store=temp_knowledge_store,
            max_iterations=config["max_iterations"],
            user_id='',
            token_budget=budget,
            temperature=config["temperature"]
        )
        
        research_result = ""
        async for chunk in react_agent.execute_react_loop_streaming(research_messages, logger):
            if not (chunk.startswith("<step>") or chunk.startswith("<step_content>")):
                research_result += chunk
        
        # Parse result
        try:
            research_data = json.loads(research_result)
            research_data["researcher_type"] = config["name"]
            research_data["confidence"] = _calculate_researcher_confidence(research_data, config)
            return research_data
        except json.JSONDecodeError:
            # Clean and retry parsing
            import re
            cleaned_result = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', research_result)
            try:
                research_data = json.loads(cleaned_result)
                research_data["researcher_type"] = config["name"]
                research_data["confidence"] = 0.5
                return research_data
            except:
                # Return fallback structure
                return {
                    "research_findings": {
                        "summary": research_result[:500] if research_result else "No findings",
                        "researcher_type": config["name"],
                        "raw_response": True
                    },
                    "confidence": 0.3
                }
                
    except Exception as e:
        logger.error(f"Researcher {config['name']} failed: {e}")
        return None

def _calculate_researcher_confidence(research_data: Dict, config: Dict) -> float:
    """Calculate confidence score for researcher results."""
    base_confidence = 0.5
    
    findings = research_data.get("research_findings", {})
    
    # Boost confidence based on completeness
    if findings.get("summary") and len(findings.get("summary", "")) > 100:
        base_confidence += 0.1
    if findings.get("system_architecture"):
        base_confidence += 0.1  
    if findings.get("implementation_guidance"):
        base_confidence += 0.1
    if findings.get("examples"):
        base_confidence += 0.1
        
    # Adjust by researcher type
    if config["name"] == "high_precision":
        base_confidence += 0.1  # High precision gets bonus
    elif config["name"] == "breadth_explorer" and len(findings.get("summary", "")) > 200:
        base_confidence += 0.1  # Breadth gets bonus for comprehensive coverage
        
    return min(base_confidence, 1.0)

async def _aggregate_research_findings(researcher_results: List[Dict], user_query: str) -> Dict:
    """Aggregate findings from multiple researchers using map-rank-reduce approach."""
    if not researcher_results:
        return {"research_findings": {"summary": "No research findings available"}}
    
    # Map: Extract claims and sources from each researcher
    all_claims = []
    
    def _safe_content_extract(content) -> str:
        """Safely extract string content from various data types."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return " ".join(str(item) for item in content)
        elif isinstance(content, dict):
            return str(content)
        elif content is None:
            return ""
        else:
            return str(content)
    
    for result in researcher_results:
        findings = result.get("findings", {}).get("research_findings", {})
        researcher_type = result.get("researcher", "unknown")
        confidence = result.get("confidence", 0.5)
        
        # Extract claims with safe content handling
        summary = findings.get("summary", "")
        if summary:
            all_claims.append({
                "content": _safe_content_extract(summary),
                "source_researcher": researcher_type,
                "confidence": confidence,
                "type": "summary"
            })
            
        # Extract other structured findings
        for key in ["system_architecture", "implementation_guidance", "api_specifications", "examples"]:
            if findings.get(key):
                all_claims.append({
                    "content": _safe_content_extract(findings[key]),
                    "source_researcher": researcher_type, 
                    "confidence": confidence,
                    "type": key
                })
    
    # Rank: Score claims using simple agreement and confidence metrics
    claim_scores = []
    for claim in all_claims:
        score = claim["confidence"]
        
        # Boost score if multiple researchers found similar info
        similar_count = sum(1 for other in all_claims 
                          if other != claim and _similarity_check(claim["content"], other["content"]))
        score += similar_count * 0.2
        
        # Boost high-precision researcher findings
        if claim["source_researcher"] == "high_precision":
            score += 0.15
            
        claim_scores.append((claim, score))
    
    # Sort by score
    claim_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Reduce: Synthesize top findings into final structure
    top_claims = [item[0] for item in claim_scores[:10]]  # Take top 10 claims
    
    # Organize by type
    final_findings = {
        "summary": "",
        "system_architecture": "",
        "implementation_guidance": "",
        "api_specifications": "",
        "examples": "",
        "confidence_score": 0.0,
        "research_quality": "comprehensive" if len(researcher_results) >= 3 else "limited"
    }
    
    # Aggregate by type
    type_contents = {}
    for claim in top_claims:
        claim_type = claim["type"]
        if claim_type not in type_contents:
            type_contents[claim_type] = []
        type_contents[claim_type].append(claim["content"])
    
    # Combine content for each type
    for claim_type, contents in type_contents.items():
        if claim_type in final_findings:
            final_findings[claim_type] = " ".join(contents)
    
    # Calculate overall confidence
    if claim_scores:
        final_findings["confidence_score"] = sum(score for _, score in claim_scores) / len(claim_scores)
    
    return {"research_findings": final_findings}

def _similarity_check(text1, text2) -> bool:
    """Simple similarity check between two text strings or other content types."""
    # Convert inputs to strings if they aren't already
    if isinstance(text1, list):
        text1 = " ".join(str(item) for item in text1)
    elif text1 is None:
        text1 = ""
    else:
        text1 = str(text1)
        
    if isinstance(text2, list):
        text2 = " ".join(str(item) for item in text2)
    elif text2 is None:
        text2 = ""
    else:
        text2 = str(text2)
    
    if not text1 or not text2:
        return False
        
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Check for significant overlap
    if len(words1) == 0 or len(words2) == 0:
        return False
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    # Jaccard similarity > 0.3 indicates similarity
    return len(intersection) / len(union) > 0.3

def _extract_research_requests(plan: List[Dict]) -> List[Dict]:
    """Extract research request steps from a plan."""
    if not plan or not isinstance(plan, list):
        return []
    return [step for step in plan if step.get("tool") == "research_request"]

def _filter_implementation_steps(plan: List[Dict]) -> List[Dict]:
    """Filter out research request steps to get implementation steps."""
    if not plan or not isinstance(plan, list):
        return []
    return [step for step in plan if step.get("tool") != "research_request"]

async def _execute_research_requests(research_requests: List[Dict], knowledge_store: Optional[KnowledgeStore] = None) -> Dict:
    """Execute a list of research requests and return consolidated findings."""
    additional_research = {}
    
    for i, research_step in enumerate(research_requests):
        args = research_step.get("args", {})
        query = args.get("query", "")
        focus_areas = args.get("focus_areas", [])
        
        logger.info(f"Executing research request {i+1}: {query}")
        
        # Create focused research messages
        research_messages = [
            {"role": "user", "content": f"{query}. Focus on: {', '.join(focus_areas)}"}
        ]
        
        # Execute targeted research
        research_result = {}
        async for chunk in execute_researcher(research_messages, knowledge_store):
            if isinstance(chunk, dict):
                research_result = chunk
                break
        
        additional_research[f"research_{i+1}"] = {
            "query": query,
            "focus_areas": focus_areas,
            "findings": research_result
        }
    
    return additional_research

async def execute_planner_with_research(prompt: str, messages: List[Dict[str, str]], research_findings: Dict = None, knowledge_store: Optional[KnowledgeStore] = None) -> str:
    """Enhanced planner that can spawn researchers when needed."""
    tools = tool_manager.get_tool_schemas()
    logger.info("Creating Plan with Research Capability")
    
    # Get relevant past planning experiences and semantic facts
    episodic_context = ""
    semantic_context = ""
    if knowledge_store:
        user_content = " ".join([msg.get("content", "") for msg in messages if msg.get("role") == "user"])
        
        # Get episodic planning experiences
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
        
        # Get relevant semantic facts for planning
        semantic_context = await knowledge_store.build_semantic_context(
            f"planning strategy {user_content[:100]}", 
            max_facts=3
        )
    
    # Include research findings in the prompt
    research_context = json.dumps(research_findings, indent=2) if research_findings else "No prior research conducted."
    
    full_context = episodic_context + "\n" + semantic_context
    chat_prompt = prompt.replace(
        "{current_date}", datetime.datetime.now().isoformat()
    ).replace("{research_findings}", research_context).replace("{tools}", json.dumps(tools, indent=2)) + full_context
    
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages

    plan = await _call_and_parse_model(chat_messages, [], "plan")
    logger.info(f"📝 PLAN GENERATED: {len(plan) if plan else 0} steps")
    
    if plan and isinstance(plan, list):
        for i, step in enumerate(plan, 1):
            logger.info(f"  Step {i}: {step.get('tool', 'unknown')} - {step.get('thought', 'no description')}")
    
    # Check if plan contains research requests
    if plan and isinstance(plan, list):
        research_requests = _extract_research_requests(plan)
        
        if research_requests:
            logger.info(f"🔍 FOUND {len(research_requests)} research requests in plan")
            
            # Execute research requests
            additional_research = await _execute_research_requests(research_requests, knowledge_store)
            
            # Update research findings with new data
            if research_findings:
                research_findings.update(additional_research)
            else:
                research_findings = additional_research
            
            # Re-plan with the additional research
            logger.info("📝 RE-PLANNING with additional research findings")
            updated_research_context = json.dumps(research_findings, indent=2)
            updated_prompt = prompt.replace("{research_findings}", updated_research_context)
            updated_chat_messages = [{"role": "system", "content": updated_prompt + full_context}] + messages
            
            # Request a new plan now that research is complete
            updated_messages = updated_chat_messages + [
                {"role": "user", "content": "Now that the research has been completed, create the final implementation plan without any more research requests."}
            ]
            
            plan = await _call_and_parse_model(updated_messages, [], "plan")
            logger.info(f"Updated Plan with Research: {plan}")
    
    # Record planning episode
    if knowledge_store and plan:
        plan_complexity = len(plan) if isinstance(plan, list) else 1
        research_used = bool(research_findings)
        knowledge_store.record_episode(
            role="planner",
            summary=f"Created plan with {plan_complexity} steps (research: {research_used})",
            tags=["planning", "strategy", "task_breakdown", "research_enabled"],
            entities={"plan_steps": plan_complexity, "research_used": research_used},
            outcome={"status": "pass", "plan_complexity": plan_complexity},
            confidence=0.9 if research_used else 0.8
        )
    
    return plan or []
    
# Backward compatibility alias
execute_planner = execute_planner_with_research

async def execute_executor(overall_goal: str, full_plan: str, previous_steps: List[Dict[str, str]], current_step: Dict[str, str], knowledge_store: Optional[KnowledgeStore] = None, domain_knowledge: str = "", pack_name: str = "") -> AsyncGenerator[str, None]:
    """Executes the executor to carry out the plan."""
    step_num = current_step.get('step', 'unknown')
    step_tool = current_step.get('tool', 'unknown')
    step_args = current_step.get('args', {})
    
    logger.info(f"🚀 EXECUTOR STARTING for step {step_num}")
    logger.info(f"   Tool: {step_tool}")
    logger.info(f"   Goal: {overall_goal[:100]}...")
    logger.info(f"   Previous steps completed: {len(previous_steps)}")
    
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
    
    # Include domain knowledge if available
    domain_context = ""
    if domain_knowledge and pack_name:
        domain_context = f"\n\nDOMAIN KNOWLEDGE (from {pack_name.split('/')[-1]} pack):\n{domain_knowledge}\n"
        logger.info(f"   Using domain knowledge from pack: {pack_name.split('/')[-1]}")
    
    chat_prompt = EXECUTOR_PROMPT.replace("{overall_goal}", overall_goal).replace("{full_plan}", json.dumps(full_plan)).replace("{previous_steps}", json.dumps(previous_steps)).replace("{current_step}", json.dumps(current_step)) + episodic_context + domain_context
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

    # Show executor initialization
    async for content in stream_step_content(f"EXECUTOR INITIALIZED\nStep: {step_num}\nTool: {step_tool}\nArgs: {str(step_args)[:200]}"):
        yield content

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
        logger.warning(f"❌ STEP {step_num} FAILED - detected error in result")
    else:
        logger.info(f"✅ STEP {step_num} SUCCESS")
    
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
    
    logger.info(f"🔄 ADAPTIVE PLANNER STARTING")
    logger.info(f"   Goal: {goal[:100]}...")
    logger.info(f"   Original plan steps: {len(original_plan)}")
    logger.info(f"   Completed steps: {len(completed_steps)}")
    logger.info(f"   Latest result length: {len(latest_result)} chars")
    logger.info(f"   Research available: {'yes' if research_findings else 'no'}")
    
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
    
    # Try up to 2 times to get valid JSON
    for attempt in range(2):
        logger.info(f"🔄 ADAPTIVE PLANNING ATTEMPT {attempt + 1}/2")
        
        result = await _call_and_parse_model(chat_messages, [], None)
        
        if result and isinstance(result, dict):
            # Validate the result has required fields
            if "action" in result and "reasoning" in result and "plan" in result:
                action = result.get("action", "unknown")
                reasoning = result.get("reasoning", "No reasoning provided")
                plan_changes = len(result.get("plan", [])) if result.get("plan") else 0
                
                logger.info(f"📋 ADAPTIVE PLANNING SUCCESS:")
                logger.info(f"   Action: {action}")
                logger.info(f"   Reasoning: {reasoning[:100]}...")
                logger.info(f"   Plan changes: {plan_changes} steps")
                
                return result
            else:
                logger.warning(f"⚠️ ADAPTIVE PLANNING INCOMPLETE - missing required fields: {result}")
                if attempt == 0:  # Only retry once
                    # Add clarification for next attempt
                    chat_messages.append({
                        "role": "user", 
                        "content": "Your response must include 'action', 'reasoning', and 'plan' fields. Please provide a complete JSON response."
                    })
        else:
            logger.warning(f"❌ ADAPTIVE PLANNING FAILED - attempt {attempt + 1}, result: {result}")
            if attempt == 0:  # Only retry once
                # Add clarification for next attempt
                chat_messages.append({
                    "role": "user", 
                    "content": "Please respond with valid JSON only. Include action, reasoning, and plan fields."
                })
    
    logger.warning("❌ ADAPTIVE PLANNING FAILED - using fallback after all attempts")
    return {"action": "continue", "reasoning": "Failed to parse planning response after multiple attempts", "plan": original_plan}

async def generate_final_user_message(goal: str, step_history: List[Dict], plan_summary: str) -> AsyncGenerator[str, None]:
    """Generate a natural user-facing message based on the completed execution."""
    
    logger.info(f"📝 GENERATING FINAL USER MESSAGE")
    logger.info(f"   Goal: {goal[:100]}...")
    logger.info(f"   Steps completed: {len(step_history)}")
    logger.info(f"   Plan summary length: {len(plan_summary)} chars")
    
    # Analyze execution outcomes for logging
    successful_steps = sum(1 for step in step_history if step.get("success", True))
    total_steps = len(step_history)
    success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0
    
    logger.info(f"   Execution success rate: {success_rate:.1f}% ({successful_steps}/{total_steps})")
    
    # First stream the step content to show progress
    progress_msg = f"Crafting personalized response... (analyzed {total_steps} steps, {success_rate:.0f}% success)"
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
    
    logger.info("🎯 CALLING MODEL for final message generation")
    
    # Now stream the actual response tokens
    token_count = 0
    async for token in call_model_server(chat_messages, []):
        token_count += 1
        yield token
    
    logger.info(f"✅ FINAL MESSAGE COMPLETE - generated {token_count} tokens")

async def evaluate_plan_success(overall_goal: str, full_plan: str, previous_steps: List[Dict[str, str]], knowledge_store: Optional[KnowledgeStore] = None) -> Dict:
    """Evaluate plan success and optionally promote insights to semantic memory"""
    
    logger.info(f"📊 EVALUATING PLAN SUCCESS")
    logger.info(f"   Goal: {overall_goal[:100]}...")
    logger.info(f"   Plan length: {len(full_plan) if isinstance(full_plan, str) else 'not string'}")
    logger.info(f"   Steps executed: {len(previous_steps)}")
    
    # Analyze step outcomes
    successful_steps = 0
    failed_steps = 0
    tools_used = set()
    
    for step in previous_steps:
        if step.get("success", True):  # Default to success if not specified
            successful_steps += 1
        else:
            failed_steps += 1
        
        # Track tools used
        if "tool" in step:
            tools_used.add(step["tool"])
    
    success_rate = (successful_steps / len(previous_steps) * 100) if previous_steps else 0
    
    logger.info(f"   Success metrics:")
    logger.info(f"     Success rate: {success_rate:.1f}% ({successful_steps}/{len(previous_steps)})")
    logger.info(f"     Failed steps: {failed_steps}")
    logger.info(f"     Tools used: {', '.join(sorted(tools_used))}")
    
    # Determine overall success
    overall_success = success_rate >= 70  # 70% threshold for success
    
    evaluation = {
        "overall_success": overall_success,
        "success_rate": success_rate,
        "successful_steps": successful_steps,
        "failed_steps": failed_steps,
        "total_steps": len(previous_steps),
        "tools_used": list(tools_used),
        "insights": []
    }
    
    # Generate insights based on performance
    if success_rate >= 90:
        evaluation["insights"].append("Excellent execution with minimal errors")
    elif success_rate >= 70:
        evaluation["insights"].append("Good execution with some challenges overcome")
    elif success_rate >= 50:
        evaluation["insights"].append("Moderate success but significant room for improvement")
    else:
        evaluation["insights"].append("Poor execution - need to identify failure patterns")
    
    if len(tools_used) > 5:
        evaluation["insights"].append(f"Complex task requiring {len(tools_used)} different tools")
    
    if failed_steps > 0:
        evaluation["insights"].append(f"Had {failed_steps} failures that needed recovery")
    
    logger.info(f"   Overall success: {'✅ YES' if overall_success else '❌ NO'}")
    logger.info(f"   Key insights: {'; '.join(evaluation['insights'])}")
    
    # Record evaluation episode if knowledge store available
    if knowledge_store:
        confidence = success_rate / 100.0  # Convert percentage to 0-1 scale
        knowledge_store.record_episode(
            role="evaluator",
            summary=f"Plan evaluation: {success_rate:.1f}% success ({successful_steps}/{len(previous_steps)} steps)",
            tags=["evaluation", "plan_assessment", "performance_metrics"],
            entities={"goal": overall_goal[:100], "tools": list(tools_used), "steps": len(previous_steps)},
            outcome={"success": overall_success, "rate": success_rate, "insights": evaluation["insights"]},
            confidence=confidence
        )
        logger.info(f"   Recorded evaluation episode with confidence {confidence:.2f}")
    
    chat_prompt = PLAN_RESULT_EVALUATOR_PROMPT.replace("{user_goal}", overall_goal).replace("{original_plan}", json.dumps(full_plan)).replace("{execution_results}", json.dumps(previous_steps))
    result = await _call_and_parse_model([{'role' : 'system', 'content' : chat_prompt}], [])
    logger.info(f"Plan evaluation result: {result}")
    
    # If successful and knowledge store available, promote key insights to semantic memory
    if result.get('result') == 'success' and knowledge_store:
        await _promote_successful_insights_to_semantic(overall_goal, previous_steps, knowledge_store)
    
    return result

async def _promote_successful_insights_to_semantic(goal: str, steps: List[Dict[str, str]], knowledge_store: KnowledgeStore):
    """Extract and promote successful patterns from completed task to semantic memory"""
    try:
        # Extract key insights from successful execution
        successful_tools = []
        key_outcomes = []
        
        for step in steps:
            if step.get('status') == 'success' and 'tool_calls' in step:
                for tool_call in step['tool_calls']:
                    tool_name = tool_call.get('function', {}).get('name')
                    if tool_name:
                        successful_tools.append(tool_name)
            
            if 'result' in step and 'error' not in step['result'].lower():
                key_outcomes.append(step['result'][:100])  # First 100 chars
        
        # Promote tool usage patterns
        if successful_tools:
            tool_pattern = f"For {goal[:50]} tasks, effective tools include: {', '.join(set(successful_tools))}"
            await knowledge_store.write_semantic_fact(
                text=tool_pattern,
                tags=["tool_pattern", "successful_execution"],
                entities={"tools": list(set(successful_tools)), "goal_type": goal[:50]},
                confidence=0.8
            )
        
        # Promote execution strategy if multiple steps were successful
        if len(key_outcomes) >= 2:
            strategy_insight = f"Multi-step approach effective for {goal[:50]} type tasks"
            await knowledge_store.write_semantic_fact(
                text=strategy_insight,
                tags=["execution_strategy", "multi_step"],
                entities={"strategy": "multi_step", "goal_type": goal[:50]},
                confidence=0.75
            )
        
        logger.info("Successfully promoted insights to semantic memory")
        
    except Exception as e:
        logger.error(f"Error promoting insights to semantic memory: {e}")

async def handle_task_route(messages: List[Dict[str, str]], goal: str, knowledge_store: Optional[KnowledgeStore] = None) -> AsyncGenerator[str, None]:
    """Complex task agent workflow with research phase and episodic memory."""
    
    logger.info(f"🎯 TASK ROUTE HANDLER STARTING")
    logger.info(f"   Goal: {goal[:100]}...")
    logger.info(f"   Message count: {len(messages)}")
    logger.info(f"   Knowledge store: {'provided' if knowledge_store else 'will create new'}")

    # Initialize knowledge store if not provided
    if not knowledge_store:
        knowledge_store = KnowledgeStore()
        logger.info("   Created new knowledge store instance")

    # Add pack integration for domain-specific knowledge
    domain_knowledge = ""
    pack_name = ""
    try:
        # Select appropriate learning pack based on user query
        pack_name = knowledge_store.select_pack_by_embedding(goal)
        if pack_name:
            logger.info(f"   Selected learning pack: {pack_name}")
            yield f"🎓 Using {pack_name.split('/')[-1]} pack for specialized knowledge..."
            
            # Build domain-specific knowledge context
            domain_knowledge = knowledge_store.build_domain_knowledge_context(goal, pack_name)
            logger.info(f"   Successfully integrated pack: {pack_name.split('/')[-1]}")
            
            async for content in stream_step_content(f"📚 Selected pack: {pack_name.split('/')[-1]}\n🧠 Retrieved relevant knowledge for your task"):
                yield content
        else:
            logger.info("   No specific learning pack selected")
    except Exception as e:
        logger.warning(f"   Pack integration failed: {e}")
        # Continue without pack integration

    # Step 1: Assessment
    yield "<step>Assessing Task Requirements</step>"
    logger.info("🔍 PHASE 1: ASSESSMENT")
    assessment = await initial_assessment(messages, knowledge_store)
    
    assessment_summary = str(assessment) if assessment else "No assessment"
    logger.info(f"   Assessment result: {assessment_summary[:100]}...")
    
    # Determine if research is needed based on the new assessment structure
    if isinstance(assessment, dict):
        needs_research = assessment.get('needs_research', True)
        assessment_value = assessment.get('assessment', 'unknown')
        logger.info(f"   Assessment value: {assessment_value}")
    else:
        # Fallback for old format or failed assessment
        needs_research = True
        logger.info("   Fallback: Research needed due to assessment failure")
    
    async for content in stream_step_content(f"ASSESSMENT COMPLETE\nResult: {assessment_summary[:200]}\nNext: {'Research phase' if needs_research else 'Direct to planning'}"):
        yield content
    
    research_findings = []
    if needs_research:
        # Step 2: RELENTLESS RESEARCH PHASE
        yield "<step>Conducting Research</step>"
        logger.info("🔬 PHASE 2: RESEARCH")
        logger.info("   Starting multi-researcher investigation...")
        
        research_start_time = time.time()
        async for chunk in execute_researcher(messages, knowledge_store):
            if isinstance(chunk, dict):
                # This is the final research result
                research_findings = chunk
                research_duration = time.time() - research_start_time
                logger.info(f"   Research completed in {research_duration:.1f}s with {len(chunk)} findings")
            elif chunk.startswith("<step>") or chunk.startswith("<step_content>"):
                yield chunk
        
        findings_summary = f"{len(research_findings)} findings" if research_findings else "No findings"
        async for content in stream_step_content(f"RESEARCH COMPLETE\nDuration: {research_duration:.1f}s\nFindings: {findings_summary}"):
            yield content
    else:
        logger.info("🔬 PHASE 2: RESEARCH SKIPPED (sufficient existing knowledge)")
        research_findings = []

    # Step 3: Planning with Research Context
    yield "<step>Generating Plan</step>"
    logger.info("📝 PHASE 3: PLANNING")
    logger.info(f"   Research context: {'available' if research_findings else 'none'}")
    
    prompt = SIMPLE_PLANNER_PROMPT
    plan = await execute_planner(prompt, messages, research_findings, knowledge_store)
    
    # Show the actual plan in step content
    if plan and isinstance(plan, list):
        plan_details = "\n".join([f"Step {step.get('step', i+1)}: {step.get('tool', 'unknown')} - {step.get('thought', 'no description')[:80]}" for i, step in enumerate(plan)])
        logger.info(f"   Plan created: {len(plan)} steps")
        for i, step in enumerate(plan[:3]):  # Log first 3 steps
            logger.info(f"     Step {i+1}: {step.get('tool', 'unknown')} - {step.get('thought', 'no desc')[:50]}...")
        
        async for content in stream_step_content(f"PLAN GENERATED\nSteps: {len(plan)}\n\nPlan Details:\n{plan_details[:500]}"):
            yield content
    else:
        logger.warning("⚠️ PLANNING FAILED - no valid plan generated")
        async for content in stream_step_content("PLAN GENERATION FAILED\nReason: No valid plan returned\nStatus: Proceeding with fallback approach"):
            yield content
    
    # Step 4: Execute the plan
    logger.info("🚀 PHASE 4: EXECUTION")
    logger.info(f"   Executing plan with {len(plan) if plan else 0} steps")
    logger.info(f"   Using learning pack: {pack_name.split('/')[-1] if pack_name else 'none'}")
    
    execution_start_time = time.time()
    async for result in execute_control_loop(plan, goal, messages, research_findings, knowledge_store, domain_knowledge, pack_name):
        yield result

async def execute_control_loop(plan, goal, messages, research_findings=None, knowledge_store: Optional[KnowledgeStore] = None, domain_knowledge: str = "", pack_name: str = ""):
    step_history = []
    current_plan = plan or []  # Ensure we have a list
    step_index = 0
    
    # Handle empty plan
    if not current_plan:
        logger.warning("Received empty plan, cannot execute")
        yield f"<step>Plan Generation Failed</step>"
        async for content in stream_step_content("Plan generation failed - no valid plan received"):
            yield content
        return
    
    def truncate_step_name(name: str, max_length: int = 30) -> str:
        """Truncate step name to max_length characters"""
        if len(name) <= max_length:
            return name
        return name[:max_length-3] + "..."
    
    while step_index < len(current_plan):
        step = current_plan[step_index]
        step_num = step.get('step', step_index + 1)
        step_tool = step.get('tool', 'unknown')
        step_thought = step.get('thought', 'No description')
        
        # Execute the current step
        step_name = f"Executing Step {step_num}: {step_tool}"
        truncated_name = truncate_step_name(step_name)
        yield f"<step>{truncated_name}</step>"
        
        logger.info(f"🔧 EXECUTING STEP {step_num}: {step_tool}")
        logger.info(f"   Thought: {step_thought}")
        logger.info(f"   Args: {step.get('args', {})}")

        result = ""
        async for chunk in execute_executor(goal, current_plan, step_history, step, knowledge_store, domain_knowledge, pack_name):
            if chunk.startswith("<step>") or chunk.startswith("<step_content>"):
                yield chunk
            else:
                result = chunk  # The final result comes as the last chunk
        
        logger.info(f"✅ STEP {step_num} COMPLETED. Result length: {len(result)} chars")
        step_history.append({"step": step_num, "tool": step_tool, "result": result})
        
        # Show step completion with result preview
        result_preview = result[:200] + "..." if len(result) > 200 else result
        async for content in stream_step_content(f"STEP {step_num} COMPLETED\nTool: {step_tool}\nResult Preview: {result_preview}"):
            yield content
        step_index += 1
        
        # After each step (except the last), check if we need to adjust the plan
        if step_index < len(current_plan):
            yield f"<step>Evaluating Plan Progress</step>"
            logger.info(f"Checking if plan adjustment needed after step {step['step']}")
            adaptive_result = await execute_adaptive_planner(
                goal, current_plan, step_history, result, messages, research_findings
            )
            
            if adaptive_result.get("action") == "research":
                logger.info(f"Research requested: {adaptive_result.get('reasoning')}")
                research_msg = f"Conducting additional research: {adaptive_result.get('reasoning')}"
                async for content in stream_step_content(research_msg):
                    yield content
                
                # Execute research requests from the adaptive plan
                updated_plan = adaptive_result.get("plan", [])
                research_requests = _extract_research_requests(updated_plan)
                
                if research_requests:
                    additional_research = {}
                    for i, research_step in enumerate(research_requests):
                        args = research_step.get("args", {})
                        query = args.get("query", "")
                        focus_areas = args.get("focus_areas", [])
                        
                        logger.info(f"Executing adaptive research request {i+1}: {query}")
                        async for content in stream_step_content(f"Researching: {query[:50]}..."):
                            yield content
                        
                        # Create focused research messages
                        research_messages = [
                            {"role": "user", "content": f"{query}. Focus on: {', '.join(focus_areas)}"}
                        ]
                        
                        # Execute targeted research
                        research_result = {}
                        async for chunk in execute_researcher(research_messages, knowledge_store):
                            if isinstance(chunk, dict):
                                research_result = chunk
                                break
                        
                        additional_research[f"adaptive_research_{i+1}"] = {
                            "query": query,
                            "focus_areas": focus_areas,
                            "findings": research_result
                        }
                    
                    # Update research findings
                    if research_findings:
                        research_findings.update(additional_research)
                    else:
                        research_findings = additional_research
                    
                    async for content in stream_step_content("Research completed, updating plan..."):
                        yield content
                
                # Filter out research_request steps and proceed with implementation steps
                implementation_plan = _filter_implementation_steps(updated_plan)
                current_plan = implementation_plan
                step_index = 0  # Start from beginning with research-informed plan
                
            elif adaptive_result.get("action") == "replan":
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
    result = await evaluate_plan_success(goal, current_plan, step_history, knowledge_store)
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