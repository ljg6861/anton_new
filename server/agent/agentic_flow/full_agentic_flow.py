import json
import logging
from typing import AsyncGenerator, Dict, List, Optional
from pathlib import Path
from server.agent.agentic_flow.helpers_and_prompts import call_model_server
from server.agent.config import USER_ROLE, MODEL_SERVER_URL
from server.agent.agentic_flow.task_flow import handle_task_route
from server.agent.tools.tool_manager import tool_manager
from server.agent.react.react_agent import ReActAgent
from server.agent.react.token_budget import TokenBudget
from server.agent.knowledge_store import KnowledgeStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit" 


FEW_SHOT_ROUTER_PROMPT = """You are a precise and logical AI router. Your only function is to classify the user's intent based on their most recent message and output a single, valid JSON object. Do not add any conversational text or explanations.

# ROUTES
- `chat`: For general conversation, greetings, and simple questions that do not require action.
- `task`: For any request that requires an action, tool use, or a multi-step process. This includes simple commands like setting reminders and complex requests like writing code.

Based on the route descriptions, classify the user's message by following the pattern in the examples below.

--- EXAMPLES ---

User Message: "hello there how are you"
{"route": "chat"}

User Message: "What is the largest planet in our solar system?"
{"route": "chat"}

User Message: "Set a reminder to call Mom at 5pm."
{"route": "task"}

User Message: "Schedule a meeting with the design team for tomorrow morning."
{"route": "task"}

User Message: "Create a to-do list for my grocery shopping trip."
{"route": "task"}

User Message: "Write a python script to parse a CSV file."
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
    """Enhanced chat route with tool calling capabilities and mandatory research for unknown topics"""
    logger.info("Handling chat route with research and learning capability")
    
    # Extract user query for learning context
    user_content = " ".join([msg.get("content", "") for msg in messages if msg.get("role") == "user"])
    
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
                yield f"🎓 Using {pack_name} pack for specialized knowledge..."
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
    
    # Create enhanced chat prompt that includes domain knowledge
    domain_knowledge_section = ""
    if domain_knowledge:
        domain_knowledge_section = f"""

🎓 DOMAIN KNOWLEDGE AVAILABLE:
{domain_knowledge}

Use this domain knowledge to provide detailed, accurate answers. This knowledge comes from the {pack_name} learning pack and should be your primary source for relevant questions.
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

Remember: After using tools successfully, confidently provide information based on what you found!{domain_knowledge_section}"""

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
    
    async for chunk in react_agent.execute_react_loop_streaming(chat_messages, logger):
        # Track tool usage for learning
        if chunk.startswith("<step>Using "):
            tool_name = chunk.replace("<step>Using ", "").replace("</step>", "")
            tools_used.append(tool_name)
            if "web_search" in tool_name or "search_codebase" in tool_name:
                research_conducted = True
        
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
    
    # Determine success based on whether research was conducted when needed
    success = True
    confidence = 0.8
    
    if any(keyword in user_content.lower() for keyword in ["what is", "tell me about", "weather", "news"]):
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