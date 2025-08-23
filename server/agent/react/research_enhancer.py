"""
Research Enhancement System for Anton Agent

This module implements research quality enforcement to address weaknesses in agent responses,
particularly for complex queries requiring detailed factual information.

Key improvements:
1. Research depth validation before response generation  
2. Fact verification for basic claims
3. Domain-specific quality checks
4. Multi-source research requirements
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

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
    Enhances ReAct agent with thorough research requirements.
    
    Analyzes queries to determine research depth needed and validates
    that sufficient research has been conducted before allowing response generation.
    """
    
    def __init__(self):
        self.music_keywords = [
            "song", "music", "guitar", "theory", "chord", "scale", "melody", 
            "harmony", "rhythm", "composition", "artist", "album", "band"
        ]
        self.factual_keywords = [
            "who", "what", "when", "where", "why", "how", "date", "year",
            "history", "biography", "facts", "details", "information"
        ]
        
    def analyze_query_complexity(self, user_query: str) -> ResearchRequirement:
        """
        Analyze user query to determine research requirements.
        
        Returns ResearchRequirement with appropriate depth for the query type.
        """
        query_lower = user_query.lower()
        
        # Music theory queries need specialized research
        if any(keyword in query_lower for keyword in self.music_keywords):
            if "theory" in query_lower or "analysis" in query_lower:
                return ResearchRequirement(
                    min_sources=3,
                    fact_verification_needed=True,
                    domain_specific_checks=["music_theory_accuracy", "artist_facts"],
                    required_information_types=["basic_facts", "technical_analysis", "historical_context"]
                )
        
        # General factual queries
        if any(keyword in query_lower for keyword in self.factual_keywords):
            return ResearchRequirement(
                min_sources=2,
                fact_verification_needed=True,
                required_information_types=["basic_facts", "verification"]
            )
        
        # Default for other queries
        return ResearchRequirement(
            min_sources=1,
            fact_verification_needed=False
        )
    
    def validate_research_depth(self, user_query: str, tool_calls_made: List[Dict[str, Any]], 
                              requirement: ResearchRequirement) -> Tuple[bool, List[str]]:
        """
        Validate that sufficient research has been conducted.
        
        Returns (is_sufficient, list_of_missing_requirements)
        """
        issues = []
        
        # Count actual research tool calls (not generic or repetitive)
        research_calls = [
            call for call in tool_calls_made 
            if call.get('name') in ['web_search', 'fetch_web_page'] 
            and call.get('status') == 'success'
        ]
        
        if len(research_calls) < requirement.min_sources:
            issues.append(f"Insufficient research sources: {len(research_calls)} < {requirement.min_sources}")
        
        # Check for basic fact verification if required
        if requirement.fact_verification_needed:
            if not self._has_basic_fact_verification(user_query, tool_calls_made):
                issues.append("Missing basic fact verification (album, date, artist details)")
        
        # Check for diverse search queries (not repetitive)
        if not self._has_diverse_research(tool_calls_made):
            issues.append("Research queries too similar - need diverse information gathering")
            
        return len(issues) == 0, issues
    
    def _has_basic_fact_verification(self, user_query: str, tool_calls: List[Dict[str, Any]]) -> bool:
        """Check if basic facts have been researched"""
        query_lower = user_query.lower()
        
        # For music queries, check if basic song/artist facts were researched
        if any(keyword in query_lower for keyword in self.music_keywords):
            search_queries = []
            for call in tool_calls:
                if call.get('name') == 'web_search' and call.get('arguments'):
                    # Handle both string and dict arguments
                    arguments = call.get('arguments')
                    if isinstance(arguments, dict):
                        search_queries.append(arguments.get('query', '').lower())
                    elif isinstance(arguments, str):
                        # Try to parse as JSON if it's a string
                        try:
                            import json
                            parsed_args = json.loads(arguments)
                            if isinstance(parsed_args, dict):
                                search_queries.append(parsed_args.get('query', '').lower())
                        except (json.JSONDecodeError, AttributeError):
                            # If parsing fails, skip this call
                            continue
            
            # Check if searches included basic factual queries
            has_basic_info = any(
                any(fact_term in query for fact_term in ['album', 'year', 'release', 'date', 'biography'])
                for query in search_queries
            )
            return has_basic_info
        
        return True  # Don't require for non-music queries
    
    def _has_diverse_research(self, tool_calls: List[Dict[str, Any]]) -> bool:
        """Check if research queries are sufficiently diverse"""
        search_queries = []
        for call in tool_calls:
            if call.get('name') == 'web_search' and call.get('arguments'):
                # Handle both string and dict arguments
                arguments = call.get('arguments')
                if isinstance(arguments, dict):
                    search_queries.append(arguments.get('query', '').lower())
                elif isinstance(arguments, str):
                    # Try to parse as JSON if it's a string
                    try:
                        import json
                        parsed_args = json.loads(arguments)
                        if isinstance(parsed_args, dict):
                            search_queries.append(parsed_args.get('query', '').lower())
                    except (json.JSONDecodeError, AttributeError):
                        # If parsing fails, skip this call
                        continue
        
        if len(search_queries) < 2:
            return False
            
        # Check that queries are sufficiently different
        for i, query1 in enumerate(search_queries):
            for query2 in search_queries[i+1:]:
                overlap = len(set(query1.split()) & set(query2.split()))
                total_words = len(set(query1.split()) | set(query2.split()))
                if total_words > 0 and overlap / total_words > 0.7:  # >70% overlap
                    return False
        
        return True
    
    def suggest_additional_research(self, user_query: str, existing_calls: List[Dict[str, Any]], 
                                  requirement: ResearchRequirement) -> List[str]:
        """
        Suggest specific additional research queries to improve response quality.
        """
        suggestions = []
        query_lower = user_query.lower()
        
        existing_queries = set()
        for call in existing_calls:
            if call.get('name') == 'web_search' and call.get('arguments'):
                # Handle both string and dict arguments
                arguments = call.get('arguments')
                if isinstance(arguments, dict):
                    existing_queries.add(arguments.get('query', '').lower())
                elif isinstance(arguments, str):
                    # Try to parse as JSON if it's a string
                    try:
                        import json
                        parsed_args = json.loads(arguments)
                        if isinstance(parsed_args, dict):
                            existing_queries.add(parsed_args.get('query', '').lower())
                    except (json.JSONDecodeError, AttributeError):
                        # If parsing fails, skip this call
                        continue
        
        # Music-specific suggestions
        if any(keyword in query_lower for keyword in self.music_keywords):
            # Extract song/artist from query for targeted research
            if "mr. crowley" in query_lower or "mr crowley" in query_lower:
                music_suggestions = [
                    "Mr. Crowley album release date Blizzard of Ozz",
                    "Randy Rhoads guitar techniques Mr. Crowley analysis",
                    "Mr. Crowley key signature chord progression music theory",
                    "Randy Rhoads biography guitar style"
                ]
                for suggestion in music_suggestions:
                    if suggestion.lower() not in existing_queries:
                        suggestions.append(suggestion)
        
        # General fact-checking suggestions
        if requirement.fact_verification_needed and not suggestions:
            suggestions.append("Verify basic facts and details about the topic")
        
        return suggestions[:2]  # Limit to 2 suggestions to avoid overwhelming
    
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
