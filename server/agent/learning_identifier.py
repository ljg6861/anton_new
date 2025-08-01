"""
Learning Identification Component for the Anton agent system.

This component analyzes agent interactions to identify novel and valuable
insights that could improve future performance.
"""
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LearningInsight:
    """Represents a novel insight identified from agent interactions."""
    insight_text: str
    source: str
    confidence: float
    context: str
    keywords: List[str]


class LearningIdentifier:
    """
    Analyzes agent interactions to identify novel and valuable insights.
    
    This component looks for patterns that indicate the agent has learned
    something new that could be valuable for future tasks.
    """
    
    def __init__(self):
        self.insight_patterns = self._initialize_insight_patterns()
        self.min_confidence = 0.6
        
    def _initialize_insight_patterns(self) -> Dict[str, Dict]:
        """Initialize patterns that indicate learning opportunities."""
        return {
            "discovery": {
                "pattern": r"(?:discovered|found|identified|realized|learned|determined|observed|detected)\s+(?:that\s+)?(.{10,200})",
                "weight": 0.8,
                "keywords": ["discovered", "found", "identified", "learned", "solution", "approach"]
            },
            "contains_has": {
                "pattern": r"(?:contains|has|includes|implements|uses|provides)\s+(.{10,200})",
                "weight": 0.7,
                "keywords": ["contains", "implements", "uses", "structure", "component"]
            },
            "successful_action": {
                "pattern": r"(?:successfully|effectively)\s+(?:read|analyzed|executed|processed|completed|handled)\s+(.{10,200})",
                "weight": 0.8,
                "keywords": ["successfully", "completed", "processed", "analyzed"]
            },
            "error_resolution": {
                "pattern": r"(?:error|issue|problem)\s+(?:was\s+)?(?:caused by|due to|resolved by|fixed by)\s+(.{10,200})",
                "weight": 0.9,
                "keywords": ["error", "issue", "problem", "fix", "resolve", "solution"]
            },
            "file_content": {
                "pattern": r"(?:the\s+)?file\s+(?:contains|has|includes|implements)\s+(.{10,200})",
                "weight": 0.6,
                "keywords": ["file", "contains", "code", "implementation", "configuration"]
            },
            "system_structure": {
                "pattern": r"(?:system|application|code|module)\s+(?:uses|implements|handles|manages)\s+(.{10,200})",
                "weight": 0.7,
                "keywords": ["system", "application", "handles", "manages", "architecture"]
            },
            "tool_result": {
                "pattern": r"(?:Tool:\s+\w+).*?Result:\s+(.{10,200})",
                "weight": 0.6,
                "keywords": ["tool", "result", "output", "data"]
            }
        }
    
    def analyze_interaction(
        self,
        task_description: str,
        doer_response: str,
        tool_outputs: List[str],
        context: Dict[str, Any]
    ) -> List[LearningInsight]:
        """
        Analyze a complete agent interaction to identify learning opportunities.
        
        Args:
            task_description: The task that was delegated to the doer
            doer_response: The doer's final response
            tool_outputs: List of tool execution outputs
            context: Additional context from the interaction
            
        Returns:
            List of identified learning insights
        """
        insights = []
        
        # Analyze doer response for insights
        doer_insights = self._extract_insights_from_text(
            doer_response, 
            f"doer_response_{len(insights)}", 
            task_description
        )
        insights.extend(doer_insights)
        
        # Analyze tool outputs for insights
        for i, tool_output in enumerate(tool_outputs):
            tool_insights = self._extract_insights_from_text(
                tool_output,
                f"tool_output_{i}",
                task_description
            )
            insights.extend(tool_insights)
        
        # Filter and rank insights
        filtered_insights = self._filter_and_rank_insights(insights)
        
        logger.info(f"Identified {len(filtered_insights)} learning insights from interaction")
        return filtered_insights
    
    def _extract_insights_from_text(
        self, 
        text: str, 
        source: str, 
        context: str
    ) -> List[LearningInsight]:
        """Extract potential insights from a text using pattern matching."""
        insights = []
        
        for pattern_name, pattern_config in self.insight_patterns.items():
            pattern = pattern_config["pattern"]
            weight = pattern_config["weight"]
            keywords = pattern_config["keywords"]
            
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                insight_text = match.group(1).strip()
                
                # Skip if too short or generic
                if len(insight_text) < 10 or self._is_too_generic(insight_text):
                    continue
                
                # Calculate confidence based on pattern weight and keyword presence
                confidence = self._calculate_confidence(insight_text, weight, keywords)
                
                if confidence >= self.min_confidence:
                    insight = LearningInsight(
                        insight_text=insight_text,
                        source=f"{source}_{pattern_name}",
                        confidence=confidence,
                        context=context,
                        keywords=self._extract_keywords(insight_text, keywords)
                    )
                    insights.append(insight)
        
        return insights
    
    def _is_too_generic(self, text: str) -> bool:
        """Check if the text is too generic to be valuable."""
        generic_phrases = [
            "no such file or directory",
            "file not found", 
            "permission denied",
            "command not found",
            "syntax error"
        ]
        
        text_lower = text.lower()
        # Only reject if it's very short or contains generic error messages
        if len(text.strip()) < 10:
            return True
            
        return any(phrase in text_lower for phrase in generic_phrases)
    
    def _calculate_confidence(
        self, 
        text: str, 
        base_weight: float, 
        keywords: List[str]
    ) -> float:
        """Calculate confidence score for an insight."""
        # Start with base weight from pattern
        confidence = base_weight
        
        # Boost confidence for keyword matches
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
        keyword_boost = min(keyword_matches * 0.1, 0.3)
        confidence += keyword_boost
        
        # Reduce confidence for very short insights
        if len(text) < 50:
            confidence *= 0.8
        
        # Boost confidence for structured information
        if any(char in text for char in [':', '-', '•', '1.', '2.']):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_keywords(self, text: str, pattern_keywords: List[str]) -> List[str]:
        """Extract relevant keywords from the insight text."""
        keywords = []
        text_lower = text.lower()
        
        # Add pattern-specific keywords that appear in text
        for keyword in pattern_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        # Extract additional technical keywords
        technical_terms = re.findall(r'\b(?:file|function|method|class|module|config|error|debug|test|code|tool|command)\b', text_lower)
        keywords.extend(list(set(technical_terms)))
        
        return list(set(keywords))  # Remove duplicates
    
    def _filter_and_rank_insights(self, insights: List[LearningInsight]) -> List[LearningInsight]:
        """Filter duplicate insights and rank by confidence."""
        # Remove near-duplicates
        unique_insights = []
        for insight in insights:
            is_duplicate = False
            for existing in unique_insights:
                similarity = self._calculate_similarity(insight.insight_text, existing.insight_text)
                if similarity > 0.8:
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if insight.confidence > existing.confidence:
                        unique_insights.remove(existing)
                        unique_insights.append(insight)
                    break
            
            if not is_duplicate:
                unique_insights.append(insight)
        
        # Sort by confidence descending
        unique_insights.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit to top insights to avoid noise
        return unique_insights[:3]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def should_trigger_learning(
        self, 
        insights: List[LearningInsight],
        task_success: bool,
        novel_information_discovered: bool
    ) -> bool:
        """
        Determine if learning should be triggered based on the interaction analysis.
        
        Args:
            insights: List of identified insights
            task_success: Whether the task was completed successfully
            novel_information_discovered: Whether new information was discovered
            
        Returns:
            True if learning should be triggered
        """
        # Trigger learning if we have high-confidence insights
        high_confidence_insights = [i for i in insights if i.confidence > 0.8]
        if high_confidence_insights:
            return True
        
        # Trigger learning for successful problem-solving with new information
        if task_success and novel_information_discovered and insights:
            return True
        
        # Trigger learning if we have multiple medium-confidence insights
        medium_confidence_insights = [i for i in insights if i.confidence > 0.7]
        if len(medium_confidence_insights) >= 2:
            return True
        
        return False


# Create a singleton instance
learning_identifier = LearningIdentifier()