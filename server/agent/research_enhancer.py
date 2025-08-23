"""
Research Enhancement System for Anton Agent

This module implements LLM-based research quality enforcement to address weaknesses 
in agent responses, particularly for complex queries requiring detailed factual information.

Key improvements:
1. LLM-based research depth validation 
2. Dynamic fact verification requirements
3. Intelligent quality assessment
4. Multi-source research requirements
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)


@dataclass
class ResearchRequirement:
    """Defines what research is needed for quality response"""
    min_sources: int = 2
    fact_verification_needed: bool = True
    domain_specific_checks: List[str] = None
    required_information_types: List[str] = None


class ResearchEnhancer:
    """
    LLM-powered research enhancement system.
    
    Uses LLM to analyze queries and validate research quality rather than 
    brittle keyword matching.
    """
    
    def __init__(self, llm_callback: Optional[Callable] = None):
        self.llm_callback = llm_callback
        
    async def analyze_query_complexity(self, user_query: str) -> ResearchRequirement:
        """
        Use LLM to analyze query complexity and determine research requirements.
        """
        analysis_prompt = f"""Analyze this user query and determine research requirements:

Query: "{user_query}"

Determine:
1. Minimum number of sources needed (1-5)
2. Whether fact verification is needed (true/false)
3. Types of information required

Consider:
- Technical topics need more sources
- Factual claims need verification
- Analysis questions need multiple perspectives
- Simple questions need fewer sources

Respond with JSON only:
{{
    "min_sources": 1-5,
    "fact_verification_needed": true/false,
    "reasoning": "brief explanation"
}}"""

        try:
            if self.llm_callback:
                response = await self.llm_callback(analysis_prompt)
                parsed = json.loads(response.strip())
                
                return ResearchRequirement(
                    min_sources=parsed.get("min_sources", 2),
                    fact_verification_needed=parsed.get("fact_verification_needed", True),
                    required_information_types=["analysis", "facts"]
                )
        except Exception as e:
            logger.warning(f"LLM analysis failed, using fallback: {e}")
        
        # Fallback: always require thorough research
        return ResearchRequirement(
            min_sources=3,
            fact_verification_needed=True,
            required_information_types=["analysis", "facts"]
        )
    
    async def validate_research_depth(self, user_query: str, tool_calls_made: List[Dict[str, Any]], 
                                    requirement: ResearchRequirement) -> Tuple[bool, List[str]]:
        """
        Use LLM to validate research quality and depth.
        """
        # Count successful research tool calls
        research_calls = [
            call for call in tool_calls_made 
            if call.get('name') in ['web_search', 'fetch_web_page'] 
            and call.get('status') == 'success'
        ]
        
        # Basic count check
        if len(research_calls) < requirement.min_sources:
            return False, [f"Insufficient research sources: {len(research_calls)} < {requirement.min_sources}"]
        
        # Extract search queries for LLM analysis
        search_queries = []
        for call in research_calls:
            arguments = call.get('arguments', {})
            if isinstance(arguments, str):
                try:
                    parsed_args = json.loads(arguments)
                    if isinstance(parsed_args, dict):
                        search_queries.append(parsed_args.get('query', ''))
                except (json.JSONDecodeError, AttributeError):
                    continue
            elif isinstance(arguments, dict):
                search_queries.append(arguments.get('query', ''))
        
        if not search_queries:
            return False, ["No valid search queries found"]
        
        # Use LLM to validate research quality
        validation_prompt = f"""Evaluate research quality for this query:

Original Query: "{user_query}"

Research Conducted:
{chr(10).join([f"- {query}" for query in search_queries])}

Assess:
1. Are searches diverse enough?
2. Do they cover key aspects of the query?
3. Is basic fact verification included?
4. Are searches specific and targeted?

Respond with JSON only:
{{
    "sufficient": true/false,
    "issues": ["issue1", "issue2"],
    "missing_research": ["what else to search for"]
}}"""

        try:
            if self.llm_callback:
                response = await self.llm_callback(validation_prompt)
                parsed = json.loads(response.strip())
                
                is_sufficient = parsed.get("sufficient", False)
                issues = parsed.get("issues", [])
                
                return is_sufficient, issues
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
        
        # Fallback: basic diversity check
        if len(set(search_queries)) < len(search_queries) * 0.8:
            return False, ["Search queries too similar"]
        
        return True, []
    
    async def suggest_additional_research(self, user_query: str, existing_calls: List[Dict[str, Any]], 
                                        requirement: ResearchRequirement) -> List[str]:
        """
        Use LLM to suggest specific additional research queries.
        """
        existing_queries = []
        for call in existing_calls:
            arguments = call.get('arguments', {})
            if isinstance(arguments, str):
                try:
                    parsed_args = json.loads(arguments)
                    if isinstance(parsed_args, dict):
                        existing_queries.append(parsed_args.get('query', ''))
                except (json.JSONDecodeError, AttributeError):
                    continue
            elif isinstance(arguments, dict):
                existing_queries.append(arguments.get('query', ''))
        
        suggestion_prompt = f"""Suggest additional research for this query:

Original Query: "{user_query}"

Already Searched:
{chr(10).join([f"- {query}" for query in existing_queries])}

Suggest 2-3 specific additional searches to improve answer quality.
Focus on missing facts, verification, or different perspectives.

Respond with JSON only:
{{
    "suggestions": ["search query 1", "search query 2", "search query 3"]
}}"""

        try:
            if self.llm_callback:
                response = await self.llm_callback(suggestion_prompt)
                parsed = json.loads(response.strip())
                
                return parsed.get("suggestions", [])[:3]
        except Exception as e:
            logger.warning(f"LLM suggestion failed: {e}")
        
        # Fallback suggestions
        return [
            f"Verify basic facts about the topic in: {user_query}",
            f"Find additional sources for: {user_query}",
            f"Cross-check information about: {user_query}"
        ][:2]
    
    def generate_research_guidance(self, user_query: str, issues: List[str], 
                                 suggestions: List[str]) -> str:
        """
        Generate guidance message for agent to conduct more thorough research.
        """
        guidance = "🔍 RESEARCH ENHANCEMENT REQUIRED:\n\n"
        
        if issues:
            guidance += "Issues with current research:\n"
            for issue in issues:
                guidance += f"• {issue}\n"
            guidance += "\n"
        
        if suggestions:
            guidance += "Suggested additional research:\n"
            for suggestion in suggestions:
                guidance += f"• Search for: \"{suggestion}\"\n"
            guidance += "\n"
        
        guidance += ("Before generating your final response, ensure you have:\n"
                    "1. Verified basic factual claims\n"
                    "2. Gathered information from multiple reliable sources\n"
                    "3. Conducted diverse, specific searches\n"
                    "4. Cross-checked key details\n\n"
                    "Continue researching instead of providing a premature answer.")
        
        return guidance
