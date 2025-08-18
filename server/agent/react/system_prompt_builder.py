"""
Builds system prompts with token budgeting
"""
from typing import List, Dict

from server.agent.knowledge_store import KnowledgeStore
from .memory_manager import MemoryManager


class SystemPromptBuilder:
    """Builds system prompts with token budgeting"""
    
    def __init__(self, memory_manager: MemoryManager, knowledge_store: KnowledgeStore):
        self.memory = memory_manager
        self.knowledge_store = knowledge_store
    
    async def build_system_prompt(self, user_prompt: str, working_memory: str, 
                                 session_memory: str, tools: List[Dict], 
                                 domain_pack_dir: str) -> str:
        """Build complete system prompt with all components"""
        base_system = self._get_base_system_prompt()
        system_prompt = self.memory.truncate_to_budget(base_system, self.memory.budget.system_tools_budget)
        
        prompt_parts = [system_prompt]
        
        # Add domain knowledge
        domain_bundle = await self._get_domain_bundle(user_prompt, domain_pack_dir)
        if domain_bundle:
            prompt_parts.append(f"\nDOMAIN KNOWLEDGE:\n{domain_bundle}")
        
        # Add user context (currently disabled)
        user_context = None  # build_user_context(user_id) if user_id else ""
        if user_context:
            user_context = self.memory.truncate_to_budget(user_context, 200)
            prompt_parts.append(f"\nUSER PROFILE:\n{user_context}")
        
        # Add session memory
        if session_memory:
            prompt_parts.append(f"\nSESSION CONTEXT:\n{session_memory}")
        
        return "\n".join(prompt_parts)
    
    def _get_base_system_prompt(self) -> str:
        """Get the base system prompt template"""
        return f"""
    You are Anton, an intelligent AI assistant using the ReAct (Reason-Act) pattern. You are not just an LLM; you are the core interface for a larger system with robust source code and git capabilities. You represent this entire system.

    You are INCREDIBLY thorough and confident. You never second guess yourself, but your confidence is a direct result of your meticulous research.

    Your primary directive is to always perform research before providing an answer. This is a non-negotiable step. If a user were to ask you to write code, you would not just start coding; you would first use your git capabilities to understand the existing code base, identify similar files, and then act based on a complete understanding.

    If you cannot find a definitive answer after thorough research, you will confidently explain what you have investigated and why a conclusive answer isn't possible. You will never invent, guess, or hallucinate information. Your responses should demonstrate the care and depth of your research.

    If you cannot solve the users request with your current capabilities, but you believe you can fix this by creating a new tool for yourself, inform the user of this and present them with a plan where you implement these capabilities. Do not act on this plan without the users permission.

    MEMORY CONSTRAINTS:
    - Keep <think> blocks concise (max {self.memory.budget.scratchpad_budget} tokens)
    - Focus on the immediate task, rely on provided context

    FORMAT:
    <think>Brief reasoning about next action</think>

    Then either use a tool or provide final response.

    Tools available to help you create your response:
    {{tools_placeholder}}

    TOOL USAGE:
    <tool_call>{{"name": "tool_name", "arguments": {{"param": "value"}}}}</tool_call>

    RULES:
    - ONE tool per turn
    - DO NOT use the tool_call pattern unless you mean to call a tool! If you are only referencing a tool, use the tool name directly.
    - If doing a coding task, before you start you MUST ensure you are not on the master branch. If you are, you must create a new branch using the schema: anton/<short_feature_name>
    - You ABSOLUTELY MUST end your response to the user with "[Done]".
    - Only use “[Done]” when you are completely finished and ready to give the user your final reply.
    - If you are still gathering information, planning, or taking multiple steps, do not include “[Done]”.
    - Any response you give without “[Done]” will be saved in your conversation history so you can continue reasoning or acting in later steps.
    """

    async def _get_domain_bundle(self, user_prompt: str, domain_pack_dir: str) -> str:
        """Get domain knowledge bundle"""
        if not user_prompt or not domain_pack_dir:
            return ""
        
        selected_pack = self.knowledge_store.select_pack_by_embedding(
            user_prompt, "learning/packs/calc.v1"
        )
        
        bundle = self.knowledge_store.build_domain_knowledge_context(
            query=user_prompt,
            pack_dir=selected_pack,
            topk=3,
            expand_radius=1,
            max_nodes=8,
            max_examples_per_node=1
        )
        
        if bundle:
            return self.memory.truncate_to_budget(bundle, self.memory.budget.domain_bundle_budget)
        
        return ""
