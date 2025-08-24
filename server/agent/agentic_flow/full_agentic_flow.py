import json
import logging
import os
from typing import AsyncGenerator, Dict, List, Optional
from pathlib import Path
import httpx
from server.agent.agentic_flow.helpers_and_prompts import call_model_server
from server.agent.config import USER_ROLE, MODEL_SERVER_URL
from server.agent.agentic_flow.task_flow import handle_task_route
from server.agent.tools.tool_manager import tool_manager
from server.agent.react.react_agent import ReActAgent
from server.agent.react.token_budget import TokenBudget
from server.agent.knowledge_store import KnowledgeStore
from server.agent.react.research_enhancer import ResearchEnhancer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit" 


FEW_SHOT_ROUTER_PROMPT = """You are a precise and logical AI router. Your only function is to classify the user's intent based on their most recent message and output a single, valid JSON object. Do not add any conversational text or explanations.

# ROUTES
- `chat`: For basic greetings, simple acknowledgments, and very simple factual questions.
- `task`: For any request requiring research, analysis, detailed explanations, tool use, or multi-step processes. This includes all analysis requests, explanations requiring expertise, and complex questions.

Based on the route descriptions, classify the user's message by following the pattern in the examples below.

--- EXAMPLES ---

User Message: "hello there how are you"
{"route": "chat"}

User Message: "thanks"
{"route": "chat"}

User Message: "What's 2+2?"
{"route": "chat"}

User Message: "Analyze this for me"
{"route": "task"}

User Message: "Explain how this works"
{"route": "task"}

User Message: "Break down this topic"
{"route": "task"}

User Message: "Research the latest developments"
{"route": "task"}

User Message: "Create a plan for me"
{"route": "task"}

User Message: "Write code to solve this"
{"route": "task"}

--- END EXAMPLES ---

Now, classify the following user message.
"""

def extract_user_prompt(messages: List[Dict[str, str]]) -> str:
    """Extract the latest user prompt from messages"""
    for msg in reversed(messages):
        if msg.get("role") == USER_ROLE:
            return msg.get("content", "")
    return ""

async def determine_route(conversation_history: List[Dict]) -> Optional[str]:
    MAX_ROUTER_RETRIES = 2
    router_messages = [
        {"role": "system", "content": FEW_SHOT_ROUTER_PROMPT},
    ] + conversation_history

    for attempt in range(MAX_ROUTER_RETRIES):
        logger.info(f"Router attempt {attempt + 1}/{MAX_ROUTER_RETRIES}...")
        
        raw_response = ""
        async for token in call_model_server(router_messages, []):
            raw_response += token
        
        # Pre-process the response to handle <think> tags
        cleaned_response = raw_response
        if "</think>" in cleaned_response:
            cleaned_response = cleaned_response.split("</think>")[-1].strip()
            logger.info("Stripped <think> tags from model response.")

        try:
            parsed_json = json.loads(cleaned_response)
            if "route" in parsed_json:
                logger.info(f"Router successfully parsed JSON: {parsed_json}")
                return parsed_json["route"]
            else:
                logger.warning(f"JSON is valid but missing 'route' key: {parsed_json}")

        except json.JSONDecodeError:
            logger.warning(f"Attempt {attempt + 1} failed: Could not decode JSON from response: '{raw_response}'")
            # "Bump" the agent with a reminder on the next attempt
            reminder = "Your previous response was not valid JSON. You MUST respond with only a JSON object."
            router_messages.append({"role": "assistant", "content": raw_response})
            router_messages.append({"role": "user", "content": reminder})
    
    logger.error("Router failed to produce a valid route after multiple attempts.")
    return None


async def execute_agentic_flow(initial_messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Execute the main agentic flow"""
    logger.info("--- Starting Agentic Flow ---")

    user_prompt = extract_user_prompt(initial_messages)
    
    if user_prompt:
        # learning_loop.start_task(user_prompt)
        pass

    # --- Route to the appropriate handler ---
    route = await determine_route(initial_messages)
    logger.info(f"Routing to: {route}")

    if route == "chat":
        async for token in _handle_chat_route(initial_messages):
            yield token
    elif route == "task":
        async for token in handle_task_route(initial_messages, user_prompt):
            yield token
    else:
        yield f"Error: Unknown route '{route}' provided by the router."

async def _handle_chat_route(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Enhanced chat route with tool calling capabilities and LLM-based research validation"""
    logger.info("Handling chat route with research and learning capability")
    
    # Extract user query for learning context
    user_content = " ".join([msg.get("content", "") for msg in messages if msg.get("role") == "user"])
    
    # Create LLM callback for research enhancement
    async def llm_callback(prompt: str) -> str:
        """Callback for LLM-based research analysis"""
        try:
            vllm_port = os.getenv("VLLM_PORT", "8003")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://localhost:{vllm_port}/v1/chat/completions",
                    json={
                        "model": "anton",
                        "messages": [
                            {"role": "system", "content": "You are an expert research analyst. Respond with JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 1000
                    },
                    headers={"Content-Type": "application/json", "Authorization": "Bearer anton-vllm-key"}
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    logger.error(f"LLM callback failed: {response.status_code}")
                    return ""
        except Exception as e:
            logger.error(f"LLM callback error: {e}")
            return ""
    
    # Initialize research enhancement system with LLM callback
    research_enhancer = ResearchEnhancer(llm_callback=llm_callback)
    research_requirement = await research_enhancer.analyze_query_complexity(user_content)
    
    logger.info(f"🔍 Research requirement: min_sources={research_requirement.min_sources}, "
               f"fact_verification={research_requirement.fact_verification_needed}")
    
    # Initialize pack integration variables
    domain_knowledge = ""
    pack_name = ""
    
    # Pack selection and domain knowledge integration
    try:
        ks = KnowledgeStore()
        selected_pack = ks.select_pack_by_embedding(user_content)
        
        if selected_pack:
            pack_name = Path(selected_pack).name
            domain_knowledge = ks.build_domain_knowledge_context(user_content, selected_pack, topk=5)
            
            if domain_knowledge and domain_knowledge.strip():
                logger.info(f"Successfully integrated pack: {pack_name}")
                # Add step content to show pack selection
                yield f"<step>Knowledge Pack Selection</step>"
                yield f"<step_content>🎓 Using {pack_name} pack for specialized knowledge</step_content>"
            else:
                logger.info(f"Selected pack {pack_name} but no domain knowledge retrieved")
    except Exception as e:
        logger.warning(f"Pack integration failed: {e}")
    
    # Set up tools available for chat (including research tools)
    chat_tools = tool_manager.get_tools_by_names([
        "search_codebase",
        "web_search", 
        "fetch_web_content",
        "read_file"
    ])
    
    # Create enhanced chat prompt that includes domain knowledge and research requirements
    domain_knowledge_section = ""
    if domain_knowledge:
        domain_knowledge_section = f"""

🎓 DOMAIN KNOWLEDGE AVAILABLE:
{domain_knowledge}

Use this domain knowledge to provide detailed, accurate answers. This knowledge comes from the {pack_name} learning pack and should be your primary source for relevant questions.
"""

    # Add research enhancement guidance based on query complexity
    research_guidance = ""
    if research_requirement.min_sources > 1:
        research_guidance = f"""

🔍 RESEARCH REQUIREMENTS FOR THIS QUERY:
- MINIMUM {research_requirement.min_sources} different sources required
- {'FACT VERIFICATION mandatory for basic claims' if research_requirement.fact_verification_needed else ''}
- Use diverse search queries - avoid repetitive searches
- Cross-check information from multiple sources before responding
- For music/technical topics: verify album names, dates, artist details, and technical information

DO NOT provide your final answer until you have conducted thorough research meeting these requirements.
"""

    chat_prompt = f"""You are Anton, a helpful assistant with access to powerful research tools and domain knowledge.

🚨 CRITICAL INSTRUCTIONS 🚨

YOU HAVE SUCCESSFULLY USED TOOLS AND GOTTEN RESULTS - USE THEM!

WHEN YOU GET TOOL RESULTS:
1. ✅ YOU DO HAVE ACCESS to the information the tools just found
2. ✅ YOU CAN provide the information from the tool results  
3. ✅ YOU MUST use the tool results to answer the user's question
4. ❌ NEVER say "I cannot provide" or "I don't have access" after getting tool results

MANDATORY TOOL USAGE RULES:
- Weather questions → Use web_search → Use the weather links/data you found to answer
- Product questions → Use web_search → Use the product info you found to answer  
- Current events → Use web_search → Use the news results you found to answer
- Music/technical analysis → Use multiple web_search calls with different queries → ALWAYS verify facts before answering
- Song analysis → Search for "song name artist analysis", then "song name chord progression", then "song name meaning" → Use ALL results to build comprehensive answer

FOR MUSIC THEORY QUESTIONS:
1. NEVER guess at technical details (time signatures, keys, chord progressions)
2. ALWAYS search multiple times with different queries to verify information  
3. Look for authoritative music theory sources, not just fan sites
4. Cross-reference information from multiple sources before stating facts
5. If sources conflict, mention the discrepancy rather than picking one

EXAMPLE CORRECT BEHAVIOR:
User: "Weather in Orlando?"
You: [Use web_search] → Get AccuWeather links → "Based on my search, I found current weather information for Orlando. You can check AccuWeather and Weather.com for detailed forecasts, or here's what I found: [use the search results]"

EXAMPLE WRONG BEHAVIOR (DON'T DO THIS):
You: [Use web_search] → Get results → "I cannot provide real-time weather data" ❌ NO!

IF YOU JUST USED A TOOL AND GOT RESULTS - YOU HAVE ACCESS TO THAT INFORMATION!

Your available tools:
- web_search: Internet research (weather, products, news, etc.)
- search_codebase: Code-related questions
- read_file: File content questions

Remember: After using tools successfully, confidently provide information based on what you found!{domain_knowledge_section}{research_guidance}"""

    # Create chat messages with the enhanced prompt
    chat_messages = [{"role": "system", "content": chat_prompt}] + messages
    
    # Initialize knowledge store for this chat session with learning capabilities
    knowledge_store = KnowledgeStore()
    
    # Start learning tracking for this chat session
    from server.agent.learning_loop import learning_loop
    learning_loop.start_task(user_content)
    
    # Determine domain and start episodic run
    domain = _extract_domain_from_query(user_content)
    knowledge_store.start_episodic_run(domain)
    
    # Select relevant learning pack based on the user query
    selected_pack = knowledge_store.select_pack_by_embedding(user_content)
    pack_name = Path(selected_pack).name if selected_pack else "none"
    
    # Retrieve domain knowledge from the selected pack
    domain_knowledge = ""
    if selected_pack and selected_pack != "packs/anton_repo.v1":  # Skip default code pack for chat
        try:
            domain_knowledge = knowledge_store.build_domain_knowledge_context(
                query=user_content,
                pack_dir=selected_pack,
                topk=5,
                max_nodes=6
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve domain knowledge from {selected_pack}: {e}")
    
    # Show pack selection in step content
    if domain_knowledge:
        yield f"<step>Knowledge Pack Selection</step>"
        yield f"<step_content>📚 Selected pack: {pack_name}\n🧠 Retrieved relevant knowledge for your question</step_content>"
        logger.info(f"Chat using learning pack: {pack_name}")
    
    # Check for relevant past chat experiences
    past_chats = knowledge_store.search_episodes_by_query(
        f"chat conversation {user_content[:100]}", 
        role="chat_assistant", 
        limit=2
    )
    
    if past_chats:
        yield f"<step>Learning from Experience</step>"
        learning_info = f"Found {len(past_chats)} similar past conversations"
        yield f"<step_content>📚 {learning_info}</step_content>"
        logger.info(f"Chat learning: Found {len(past_chats)} relevant past conversations")
    
    # Set up token budget for chat (smaller than task execution)
    budget = TokenBudget(total_budget=50000)  # Smaller budget for chat
    
    # Create ReAct agent for tool-enabled chat
    react_agent = ReActAgent(
        api_base_url=MODEL_SERVER_URL,
        tools=chat_tools,
        knowledge_store=knowledge_store,
        max_iterations=10,  # Limit iterations for chat
        user_id='chat_user',
        token_budget=budget
    )
    
    # Execute chat with tool support using ReAct agent
    logger.info("Starting ReAct agent for chat with research and learning capability")
    
    tools_used = []
    research_conducted = False
    tool_calls_made = []
    research_validated = False
    
    async for chunk in react_agent.execute_react_loop_streaming(chat_messages, logger):
        # Track tool usage for learning and research validation
        if chunk.startswith("<step>Using "):
            tool_name = chunk.replace("<step>Using ", "").replace("</step>", "")
            tools_used.append(tool_name)
            if "web_search" in tool_name or "search_codebase" in tool_name:
                research_conducted = True
                
            # Track tool call for research validation (initialize with basic info)
            tool_calls_made.append({
                'name': tool_name,
                'status': 'started',
                'arguments': {}  # Will be filled when we get results
            })
        
        # Track successful tool completions and capture arguments from logs
        elif chunk.startswith("<step_content>✅") and "completed" in chunk:
            if tool_calls_made:
                tool_calls_made[-1]['status'] = 'success'
                # Note: We don't have access to the actual arguments here in the streaming,
                # but the research validation will handle missing arguments gracefully
        
        # Before generating response, validate research depth ONLY for music theory queries
        elif chunk.startswith("<step>Generating response</step>") and not research_validated:
            # Only validate for music theory queries that specifically need research
            is_music_theory = any(keyword in user_content.lower() for keyword in ['music theory', 'song', 'guitar', 'chord'])
            
            if research_requirement.min_sources > 1 and is_music_theory:
                # Validate research depth before allowing response
                is_sufficient, issues = await research_enhancer.validate_research_depth(
                    user_content, tool_calls_made, research_requirement
                )
                
                if not is_sufficient and len(tool_calls_made) < 2:  # Only block if very little research done
                    logger.warning(f"🔍 Research insufficient for music theory query: {issues}")
                    suggestions = research_enhancer.suggest_additional_research(
                        user_content, tool_calls_made, research_requirement
                    )
                    
                    # Provide research guidance instead of generating response
                    guidance = research_enhancer.generate_research_guidance(
                        user_content, issues, suggestions
                    )
                    
                    yield f"<step>Research Enhancement Required</step>"
                    yield f"<step_content>{guidance}</step_content>"
                    
                    # Don't yield the generating response step yet
                    continue
                else:
                    research_validated = True
                    logger.info("✅ Research validation passed")
            else:
                research_validated = True
                logger.info("✅ Research validation skipped (not a music theory query requiring validation)")
        
        # Pass through step markers and step content for UI processing
        if chunk.startswith("<step>") or chunk.startswith("<step_content>"):
            yield chunk
        elif chunk.startswith("<token>"):
            # Pass through token chunks as-is for proper streaming
            yield chunk
        else:
            # For any other content, skip it (shouldn't happen with proper tagging)
            continue
    
    # Record this chat interaction as a learning episode
    yield f"<step>Recording Learning</step>"
    
    # Determine success based on research quality and depth
    success = True
    confidence = 0.8
    
    # Evaluate research quality for knowledge-intensive queries
    if research_requirement.min_sources > 1:
        is_sufficient, issues = research_enhancer.validate_research_depth(
            user_content, tool_calls_made, research_requirement
        )
        
        if is_sufficient and research_conducted:
            learning_msg = f"✅ Conducted thorough research ({len(tool_calls_made)} sources) for complex query"
            confidence = 0.95
        elif research_conducted:
            learning_msg = f"⚠️ Conducted some research but could be more thorough: {'; '.join(issues)}"
            confidence = 0.7
        else:
            learning_msg = "❌ Failed to conduct required research for knowledge-intensive query"
            success = False
            confidence = 0.3
    elif any(keyword in user_content.lower() for keyword in ["what is", "tell me about", "weather", "news"]):
        # Questions that should trigger research
        if research_conducted:
            learning_msg = "✅ Successfully researched unknown topic before responding"
            confidence = 0.9
        else:
            learning_msg = "❌ Should have researched this topic first"
            success = False
            confidence = 0.4
    else:
        learning_msg = "✅ Handled conversational query appropriately"
    
    yield f"<step_content>📝 Learning: {learning_msg}</step_content>"
    
    # Record the chat episode for future learning
    knowledge_store.record_episode(
        role="chat_assistant",
        summary=f"Chat: {user_content[:100]} - {'researched' if research_conducted else 'direct answer'}",
        tags=["chat", "conversation", "research" if research_conducted else "direct"],
        entities={
            "query_type": "research_needed" if research_conducted else "conversational",
            "tools_used": tools_used,
            "domain": domain
        },
        outcome={
            "status": "pass" if success else "needs_improvement",
            "research_conducted": research_conducted,
            "tools_count": len(tools_used)
        },
        confidence=confidence
    )
    
    # Complete learning tracking
    learning_loop.complete_task(success, learning_msg)
    
    logger.info(f"Chat learning complete: {learning_msg}")


def _extract_domain_from_query(query: str) -> str:
    """Extract a domain/topic from the user query for categorization."""
    query_lower = query.lower()
    
    # Domain mapping based on keywords
    if any(word in query_lower for word in ["weather", "temperature", "forecast", "rain", "snow"]):
        return "weather"
    elif any(word in query_lower for word in ["code", "programming", "function", "class", "debug"]):
        return "coding"
    elif any(word in query_lower for word in ["product", "brand", "buy", "price", "review"]):
        return "products"
    elif any(word in query_lower for word in ["health", "medical", "symptoms", "drug", "medicine"]):
        return "health"
    elif any(word in query_lower for word in ["news", "current", "recent", "today", "latest"]):
        return "current_events"
    else:
        return "general"