"""
Evaluator Node: Gate final answers with tests/rubrics before returning.

This module implements the evaluator_node system that scores output against acceptance criteria
and provides one-shot self-repair if the score is below threshold.
"""
import logging
import json
import re
from typing import Dict, Any, Optional, Tuple, Protocol
from dataclasses import dataclass
from enum import Enum

from server.agent.state import State


logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM client interface"""
    async def complete(self, prompt: str) -> str:
        ...


class FactualityResult(Enum):
    """Results from factuality checking"""
    ACCURATE = "accurate"
    INACCURATE = "inaccurate" 
    UNVERIFIABLE = "unverifiable"


@dataclass
class AcceptanceCriteria:
    """Defines the acceptance criteria for evaluating responses"""
    min_score: float = 0.7
    required_elements: list[str] = None
    prohibited_elements: list[str] = None
    domain_specific_checks: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.required_elements is None:
            self.required_elements = []
        if self.prohibited_elements is None:
            self.prohibited_elements = []
        if self.domain_specific_checks is None:
            self.domain_specific_checks = {}


@dataclass  
class EvaluationResult:
    """Result from evaluating an answer"""
    llm_score: float
    fact_score: float
    overall_score: float
    feedback: str
    passed: bool
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class EvaluatorNode:
    """
    Evaluates outputs against acceptance criteria and provides self-repair capability.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
    async def evaluate(self, output: str, state: State, acceptance: AcceptanceCriteria) -> EvaluationResult:
        """
        Evaluate output against acceptance criteria.
        
        Args:
            output: The answer to evaluate
            state: Current agent state with context
            acceptance: Acceptance criteria to check against
            
        Returns:
            EvaluationResult with scores and feedback
        """
        # Get LLM-based score
        llm_score, llm_feedback = await self._llm_score(output, acceptance)
        
        # Get factuality score (stub implementation for now)
        fact_score = await self._check_factuality(output, state.context)
        
        # Calculate overall score (0.7 LLM + 0.3 factuality as specified)
        overall_score = 0.7 * llm_score + 0.3 * fact_score
        
        # Check if passed
        passed = overall_score >= acceptance.min_score
        
        # Build detailed feedback
        feedback = self._build_feedback(llm_feedback, fact_score, overall_score, acceptance)
        
        return EvaluationResult(
            llm_score=llm_score,
            fact_score=fact_score, 
            overall_score=overall_score,
            feedback=feedback,
            passed=passed,
            details={
                "required_elements_check": self._check_required_elements(output, acceptance),
                "prohibited_elements_check": self._check_prohibited_elements(output, acceptance),
                "domain_checks": self._run_domain_checks(output, acceptance)
            }
        )
    
    async def _llm_score(self, output: str, acceptance: AcceptanceCriteria) -> Tuple[float, str]:
        """
        Use LLM to score the output against acceptance criteria.
        
        Returns:
            Tuple of (score, feedback)
        """
        judge_prompt = self._build_judge_prompt(output, acceptance)
        
        try:
            response = await self.llm_client.complete(judge_prompt)
            return self._parse_judge_response(response)
            
        except Exception as e:
            logger.error(f"Error in LLM scoring: {e}")
            # Fallback to basic checks
            return self._fallback_scoring(output, acceptance)
    
    def _build_judge_prompt(self, output: str, acceptance: AcceptanceCriteria) -> str:
        """Build the judge prompt for LLM evaluation"""
        
        criteria_text = ""
        if acceptance.required_elements:
            criteria_text += f"\nRequired elements: {', '.join(acceptance.required_elements)}"
        if acceptance.prohibited_elements:
            criteria_text += f"\nProhibited elements: {', '.join(acceptance.prohibited_elements)}"
        if acceptance.domain_specific_checks:
            criteria_text += f"\nDomain-specific requirements: {json.dumps(acceptance.domain_specific_checks, indent=2)}"
        
        return f"""You are an expert evaluator. Score the following answer on a scale of 0.0 to 1.0 based on the acceptance criteria.

ACCEPTANCE CRITERIA:
- Minimum score required: {acceptance.min_score}{criteria_text}

ANSWER TO EVALUATE:
{output}

Please provide your evaluation in this exact format:
SCORE: [0.0-1.0]
REASONING: [detailed explanation of why you gave this score]
STRENGTHS: [what the answer does well]
WEAKNESSES: [what could be improved]
SUGGESTIONS: [specific suggestions for improvement]

Be thorough and objective in your evaluation."""

    def _parse_judge_response(self, response: str) -> Tuple[float, str]:
        """Parse LLM judge response to extract score and feedback"""
        try:
            # Extract score
            score_match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))  # Clamp to [0,1]
            else:
                score = 0.5  # Default if can't parse
                
            return score, response
            
        except Exception as e:
            logger.error(f"Error parsing judge response: {e}")
            return 0.5, f"Failed to parse judge response: {response}"
    
    def _fallback_scoring(self, output: str, acceptance: AcceptanceCriteria) -> Tuple[float, str]:
        """Fallback scoring when LLM evaluation fails"""
        score = 0.5  # Start with neutral
        feedback_parts = []
        
        # Check required elements
        required_check = self._check_required_elements(output, acceptance)
        if required_check["all_present"]:
            score += 0.2
            feedback_parts.append("✓ All required elements present")
        else:
            feedback_parts.append(f"✗ Missing required elements: {required_check['missing']}")
        
        # Check prohibited elements
        prohibited_check = self._check_prohibited_elements(output, acceptance)
        if prohibited_check["none_present"]:
            score += 0.1
            feedback_parts.append("✓ No prohibited elements found")
        else:
            score -= 0.2
            feedback_parts.append(f"✗ Prohibited elements found: {prohibited_check['found']}")
        
        # Basic length and completeness check
        if len(output.strip()) > 50:
            score += 0.1
            feedback_parts.append("✓ Adequate response length")
        else:
            feedback_parts.append("✗ Response too short")
            
        score = max(0.0, min(1.0, score))
        feedback = "FALLBACK SCORING:\n" + "\n".join(feedback_parts)
        
        return score, feedback
    
    async def _check_factuality(self, output: str, context: str) -> float:
        """
        Check factuality of output against context (stub implementation).
        
        Returns:
            Score between 0.0 and 1.0
        """
        # TODO: This is a stub implementation. In a real system, this could:
        # - Check facts against a knowledge base
        # - Verify claims against retrieved context
        # - Use fact-checking models
        # - Check for logical consistency
        
        # For now, we'll do basic checks:
        if not output or not output.strip():
            return 0.0
            
        # Check if output contradicts itself
        output_lower = output.lower()
        if any(contradiction in output_lower for contradiction in [
            "yes and no", "true and false", "both true and false",
            "i don't know but", "unclear but definitely"
        ]):
            return 0.3
            
        # Check for uncertainty markers (good for factuality)
        uncertainty_markers = ["likely", "probably", "seems", "appears", "might", "could be"]
        has_uncertainty = any(marker in output_lower for marker in uncertainty_markers)
        
        # Check for overconfident claims
        overconfident = any(claim in output_lower for claim in [
            "definitely", "absolutely", "certainly", "without doubt", "100%"
        ])
        
        if has_uncertainty and not overconfident:
            return 0.8  # Good factual awareness
        elif overconfident:
            return 0.5  # Potentially overconfident
        else:
            return 0.7  # Neutral
    
    def _check_required_elements(self, output: str, acceptance: AcceptanceCriteria) -> Dict[str, Any]:
        """Check if all required elements are present in output"""
        if not acceptance.required_elements:
            return {"all_present": True, "missing": [], "found": acceptance.required_elements}
            
        output_lower = output.lower()
        missing = []
        found = []
        
        for element in acceptance.required_elements:
            if element.lower() in output_lower:
                found.append(element)
            else:
                missing.append(element)
                
        return {
            "all_present": len(missing) == 0,
            "missing": missing,
            "found": found
        }
    
    def _check_prohibited_elements(self, output: str, acceptance: AcceptanceCriteria) -> Dict[str, Any]:
        """Check if any prohibited elements are present in output"""
        if not acceptance.prohibited_elements:
            return {"none_present": True, "found": [], "not_found": acceptance.prohibited_elements}
            
        output_lower = output.lower()
        found = []
        not_found = []
        
        for element in acceptance.prohibited_elements:
            if element.lower() in output_lower:
                found.append(element)
            else:
                not_found.append(element)
                
        return {
            "none_present": len(found) == 0,
            "found": found,
            "not_found": not_found
        }
    
    def _run_domain_checks(self, output: str, acceptance: AcceptanceCriteria) -> Dict[str, Any]:
        """Run domain-specific checks"""
        results = {}
        
        for check_name, check_config in acceptance.domain_specific_checks.items():
            try:
                if check_name == "math_notation":
                    results[check_name] = self._check_math_notation(output, check_config)
                elif check_name == "code_quality":
                    results[check_name] = self._check_code_quality(output, check_config)
                elif check_name == "citation_format":
                    results[check_name] = self._check_citation_format(output, check_config)
                else:
                    results[check_name] = {"status": "unknown_check", "config": check_config}
            except Exception as e:
                results[check_name] = {"status": "error", "error": str(e)}
                
        return results
    
    def _check_math_notation(self, output: str, config: Dict) -> Dict[str, Any]:
        """Check mathematical notation requirements"""
        # Look for LaTeX math notation
        latex_patterns = [r'\$.*?\$', r'\\\(.*?\\\)', r'\\\[.*?\\\]']
        has_latex = any(re.search(pattern, output) for pattern in latex_patterns)
        
        return {
            "has_latex": has_latex,
            "required": config.get("require_latex", False),
            "passed": not config.get("require_latex", False) or has_latex
        }
    
    def _check_code_quality(self, output: str, config: Dict) -> Dict[str, Any]:
        """Check code quality requirements"""
        # Look for code blocks
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = re.findall(code_block_pattern, output)
        
        return {
            "has_code_blocks": len(code_blocks) > 0,
            "code_block_count": len(code_blocks),
            "required": config.get("require_code", False),
            "passed": not config.get("require_code", False) or len(code_blocks) > 0
        }
    
    def _check_citation_format(self, output: str, config: Dict) -> Dict[str, Any]:
        """Check citation format requirements"""
        # Look for citations in various formats
        citation_patterns = [
            r'\[[\d,\s-]+\]',  # [1], [1,2], [1-3]
            r'\([\w\s]+,?\s*\d{4}\)',  # (Author, 2023)
            r'@\w+',  # @author2023
        ]
        
        citations_found = []
        for pattern in citation_patterns:
            citations_found.extend(re.findall(pattern, output))
            
        return {
            "citations_found": citations_found,
            "citation_count": len(citations_found),
            "required": config.get("require_citations", False),
            "passed": not config.get("require_citations", False) or len(citations_found) > 0
        }
    
    def _build_feedback(self, llm_feedback: str, fact_score: float, overall_score: float, acceptance: AcceptanceCriteria) -> str:
        """Build comprehensive feedback message"""
        feedback_parts = [
            f"EVALUATION SUMMARY:",
            f"Overall Score: {overall_score:.2f} (threshold: {acceptance.min_score})",
            f"Factuality Score: {fact_score:.2f}",
            "",
            "LLM EVALUATION:",
            llm_feedback
        ]
        
        return "\n".join(feedback_parts)
    
    async def repair_answer(self, original_answer: str, state: State, evaluation: EvaluationResult) -> str:
        """
        One-shot self-repair attempt to improve the answer.
        
        Args:
            original_answer: The original answer that scored below threshold
            state: Current agent state with context
            evaluation: The evaluation result with feedback
            
        Returns:
            Repaired answer attempt
        """
        repair_prompt = self._build_repair_prompt(original_answer, state, evaluation)
        
        try:
            repaired_answer = await self.llm_client.complete(repair_prompt)
            return repaired_answer.strip()
            
        except Exception as e:
            logger.error(f"Error in answer repair: {e}")
            return original_answer  # Return original if repair fails
    
    def _build_repair_prompt(self, original_answer: str, state: State, evaluation: EvaluationResult) -> str:
        """Build the repair prompt for self-correction"""
        
        context_summary = state.context[:1000] + "..." if len(state.context) > 1000 else state.context
        
        return f"""You need to improve your previous answer based on evaluation feedback.

ORIGINAL GOAL:
{state.goal}

RELEVANT CONTEXT:
{context_summary}

YOUR ORIGINAL ANSWER:
{original_answer}

EVALUATION FEEDBACK (Score: {evaluation.overall_score:.2f}):
{evaluation.feedback}

IMPROVEMENT REQUIREMENTS:
- Address the specific weaknesses identified in the evaluation
- Maintain the strengths that were noted
- Follow the suggestions provided
- Ensure your improved answer is complete and directly addresses the original goal

Please provide an improved version of your answer that addresses the evaluation feedback:"""


# Global evaluator instance
evaluator_node: Optional[EvaluatorNode] = None


def get_evaluator_node(llm_client: LLMClient) -> EvaluatorNode:
    """Get or create the global evaluator node instance"""
    global evaluator_node
    if evaluator_node is None:
        evaluator_node = EvaluatorNode(llm_client)
    return evaluator_node
