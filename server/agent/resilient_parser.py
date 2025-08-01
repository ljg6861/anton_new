"""
Resilient Parsing System for the Anton agent.

This component addresses the critical weakness of brittle parsing logic by:
1. Moving away from rigid regex patterns to flexible parsing
2. Providing graceful handling of unexpected outputs
3. Supporting multiple output format variations
4. Implementing fallback parsing strategies
5. Adaptive parsing based on model output patterns
"""
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import ast

logger = logging.getLogger(__name__)


class ParseResult(Enum):
    """Result status of parsing operations."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    RETRY_NEEDED = "retry_needed"


class OutputFormat(Enum):
    """Expected output formats from the model."""
    JSON = "json"
    XML_TAGS = "xml_tags"
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    STRUCTURED_TEXT = "structured_text"
    CODE_BLOCK = "code_block"


@dataclass
class ParsedOutput:
    """Result of parsing operation with metadata."""
    result: ParseResult
    content: Any
    confidence: float
    format_detected: OutputFormat
    parser_used: str
    raw_input: str
    error_message: Optional[str] = None
    alternative_interpretations: List[Any] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ParsingContext:
    """Context information for parsing operations."""
    expected_format: Optional[OutputFormat] = None
    expected_fields: Optional[List[str]] = None
    model_name: str = "unknown"
    previous_outputs: List[str] = field(default_factory=list)
    error_tolerance: float = 0.5  # 0.0 = strict, 1.0 = very tolerant


class BaseParser(ABC):
    """Abstract base class for all parsers."""
    
    def __init__(self, name: str, confidence_threshold: float = 0.7):
        self.name = name
        self.confidence_threshold = confidence_threshold
        self.success_count = 0
        self.failure_count = 0
        self.patterns_learned = []
    
    @abstractmethod
    def can_parse(self, text: str, context: ParsingContext) -> float:
        """
        Determine if this parser can handle the given text.
        Returns confidence score 0.0-1.0.
        """
        pass
    
    @abstractmethod
    def parse(self, text: str, context: ParsingContext) -> ParsedOutput:
        """Parse the text and return structured result."""
        pass
    
    def get_success_rate(self) -> float:
        """Calculate success rate for this parser."""
        total = self.success_count + self.failure_count
        return (self.success_count / total) if total > 0 else 0.0
    
    def learn_from_success(self, pattern: str) -> None:
        """Learn from successful parsing patterns."""
        if pattern not in self.patterns_learned:
            self.patterns_learned.append(pattern)
            # Keep only recent patterns
            if len(self.patterns_learned) > 20:
                self.patterns_learned = self.patterns_learned[-20:]


class JSONParser(BaseParser):
    """Parser for JSON outputs with flexible handling."""
    
    def __init__(self):
        super().__init__("JSONParser")
        self.json_patterns = [
            r'```json\s*(.*?)\s*```',  # Code block JSON
            r'```\s*(.*?)\s*```',      # Generic code block
            r'\{.*?\}',                 # Direct JSON object
            r'\[.*?\]',                 # JSON array
        ]
    
    def can_parse(self, text: str, context: ParsingContext) -> float:
        """Check if text contains JSON-like content."""
        confidence = 0.0
        
        # Look for JSON patterns
        for pattern in self.json_patterns:
            if re.search(pattern, text, re.DOTALL):
                confidence += 0.3
        
        # Check for JSON indicators
        json_indicators = ['{', '}', '":', '","', '[', ']']
        for indicator in json_indicators:
            if indicator in text:
                confidence += 0.1
        
        # Boost confidence if expecting JSON
        if context.expected_format == OutputFormat.JSON:
            confidence += 0.4
        
        return min(1.0, confidence)
    
    def parse(self, text: str, context: ParsingContext) -> ParsedOutput:
        """Parse JSON content with multiple fallback strategies."""
        try:
            # Strategy 1: Try direct JSON parsing
            result = self._try_direct_json(text)
            if result:
                self.success_count += 1
                return ParsedOutput(
                    result=ParseResult.SUCCESS,
                    content=result,
                    confidence=0.9,
                    format_detected=OutputFormat.JSON,
                    parser_used=self.name,
                    raw_input=text
                )
            
            # Strategy 2: Extract from code blocks
            result = self._extract_from_code_blocks(text)
            if result:
                self.success_count += 1
                return ParsedOutput(
                    result=ParseResult.SUCCESS,
                    content=result,
                    confidence=0.8,
                    format_detected=OutputFormat.JSON,
                    parser_used=self.name,
                    raw_input=text
                )
            
            # Strategy 3: Fuzzy JSON parsing
            result = self._fuzzy_json_parse(text, context)
            if result:
                self.success_count += 1
                return ParsedOutput(
                    result=ParseResult.PARTIAL_SUCCESS,
                    content=result,
                    confidence=0.6,
                    format_detected=OutputFormat.JSON,
                    parser_used=self.name,
                    raw_input=text,
                    suggestions=["Consider using proper JSON formatting"]
                )
            
            # All strategies failed
            self.failure_count += 1
            return ParsedOutput(
                result=ParseResult.FAILED,
                content=None,
                confidence=0.0,
                format_detected=OutputFormat.PLAIN_TEXT,
                parser_used=self.name,
                raw_input=text,
                error_message="Could not extract valid JSON"
            )
            
        except Exception as e:
            self.failure_count += 1
            return ParsedOutput(
                result=ParseResult.FAILED,
                content=None,
                confidence=0.0,
                format_detected=OutputFormat.PLAIN_TEXT,
                parser_used=self.name,
                raw_input=text,
                error_message=f"JSON parsing error: {str(e)}"
            )
    
    def _try_direct_json(self, text: str) -> Optional[Any]:
        """Try parsing text directly as JSON."""
        try:
            return json.loads(text.strip())
        except:
            return None
    
    def _extract_from_code_blocks(self, text: str) -> Optional[Any]:
        """Extract JSON from code blocks."""
        for pattern in self.json_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json_text = match.group(1) if match.groups() else match.group(0)
                    return json.loads(json_text.strip())
                except:
                    continue
        return None
    
    def _fuzzy_json_parse(self, text: str, context: ParsingContext) -> Optional[Any]:
        """Attempt fuzzy JSON parsing with corrections."""
        try:
            # Clean common JSON formatting issues
            cleaned = text.strip()
            
            # Fix common issues
            fixes = [
                (r"'\s*:\s*'", '":"'),          # Single quotes to double quotes
                (r"'\s*:\s*([^',}]+)", r'":"\1"'),  # Mixed quotes
                (r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3'),  # Unquoted keys
                (r':\s*([^",{\[\]}\s]+)(?=[,}])', r': "\1"'),  # Unquoted values
                (r',\s*}', '}'),                # Trailing commas
                (r',\s*]', ']'),                # Trailing commas in arrays
            ]
            
            for pattern, replacement in fixes:
                cleaned = re.sub(pattern, replacement, cleaned)
            
            # Try parsing again
            return json.loads(cleaned)
            
        except:
            # If expected fields are provided, try to construct JSON
            if context.expected_fields:
                return self._construct_from_expected_fields(text, context.expected_fields)
            return None
    
    def _construct_from_expected_fields(self, text: str, expected_fields: List[str]) -> Optional[Dict]:
        """Construct JSON object from expected fields found in text."""
        result = {}
        
        for field in expected_fields:
            # Look for field patterns
            patterns = [
                rf'{field}\s*:\s*"([^"]*)"',     # "field": "value"
                rf'{field}\s*:\s*([^,\s}}]+)',   # field: value
                rf'"{field}":\s*"([^"]*)"',      # Quoted field
                rf'{field}[:\s=]+([^\n,}}]+)',   # Flexible separator
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result[field] = match.group(1).strip()
                    break
        
        return result if result else None


class XMLTagParser(BaseParser):
    """Parser for XML-style tags in model output."""
    
    def __init__(self):
        super().__init__("XMLTagParser")
        self.common_tags = [
            'tool_code', 'think', 'response', 'answer', 'result',
            'content', 'data', 'output', 'summary', 'analysis'
        ]
    
    def can_parse(self, text: str, context: ParsingContext) -> float:
        """Check if text contains XML-style tags."""
        confidence = 0.0
        
        # Look for tag patterns
        tag_pattern = r'<(\w+)>(.*?)</\1>'
        matches = re.findall(tag_pattern, text, re.DOTALL)
        confidence += min(0.5, len(matches) * 0.2)
        
        # Check for specific known tags
        for tag in self.common_tags:
            if f'<{tag}>' in text and f'</{tag}>' in text:
                confidence += 0.2
        
        # Boost if expecting XML format
        if context.expected_format == OutputFormat.XML_TAGS:
            confidence += 0.4
        
        return min(1.0, confidence)
    
    def parse(self, text: str, context: ParsingContext) -> ParsedOutput:
        """Parse XML tags from text."""
        try:
            result = {}
            confidence = 0.0
            
            # Extract all tag pairs
            tag_pattern = r'<(\w+)>(.*?)</\1>'
            matches = re.findall(tag_pattern, text, re.DOTALL)
            
            for tag_name, tag_content in matches:
                result[tag_name] = tag_content.strip()
                confidence += 0.2
            
            # Extract self-closing tags
            self_closing_pattern = r'<(\w+)\s*/>'
            self_closing_matches = re.findall(self_closing_pattern, text)
            for tag_name in self_closing_matches:
                result[tag_name] = True
                confidence += 0.1
            
            if result:
                self.success_count += 1
                return ParsedOutput(
                    result=ParseResult.SUCCESS,
                    content=result,
                    confidence=min(1.0, confidence),
                    format_detected=OutputFormat.XML_TAGS,
                    parser_used=self.name,
                    raw_input=text
                )
            else:
                self.failure_count += 1
                return ParsedOutput(
                    result=ParseResult.FAILED,
                    content=None,
                    confidence=0.0,
                    format_detected=OutputFormat.PLAIN_TEXT,
                    parser_used=self.name,
                    raw_input=text,
                    error_message="No XML tags found"
                )
                
        except Exception as e:
            self.failure_count += 1
            return ParsedOutput(
                result=ParseResult.FAILED,
                content=None,
                confidence=0.0,
                format_detected=OutputFormat.PLAIN_TEXT,
                parser_used=self.name,
                raw_input=text,
                error_message=f"XML parsing error: {str(e)}"
            )


class StructuredTextParser(BaseParser):
    """Parser for structured text with key-value pairs."""
    
    def __init__(self):
        super().__init__("StructuredTextParser")
        self.key_patterns = [
            r'^(\w+):\s*(.+)$',                    # Key: Value
            r'^(\w+)\s*=\s*(.+)$',                 # Key = Value
            r'^-\s*(\w+):\s*(.+)$',                # - Key: Value
            r'^•\s*(\w+):\s*(.+)$',                # • Key: Value
            r'^(\d+)\.\s*(\w+):\s*(.+)$',          # 1. Key: Value
        ]
    
    def can_parse(self, text: str, context: ParsingContext) -> float:
        """Check if text contains structured key-value patterns."""
        confidence = 0.0
        lines = text.strip().split('\n')
        
        structured_lines = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in self.key_patterns:
                if re.match(pattern, line):
                    structured_lines += 1
                    break
        
        if len(lines) > 0:
            structure_ratio = structured_lines / len([l for l in lines if l.strip()])
            confidence = structure_ratio
        
        # Boost if expecting structured text
        if context.expected_format == OutputFormat.STRUCTURED_TEXT:
            confidence += 0.3
        
        return min(1.0, confidence)
    
    def parse(self, text: str, context: ParsingContext) -> ParsedOutput:
        """Parse structured text into key-value pairs."""
        try:
            result = {}
            lines = text.strip().split('\n')
            parsed_lines = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                for pattern in self.key_patterns:
                    match = re.match(pattern, line)
                    if match:
                        if len(match.groups()) == 2:
                            key, value = match.groups()
                        else:  # Pattern with numbering
                            _, key, value = match.groups()
                        
                        result[key.lower().replace(' ', '_')] = value.strip()
                        parsed_lines += 1
                        break
            
            confidence = parsed_lines / len(lines) if lines else 0.0
            
            if result:
                self.success_count += 1
                return ParsedOutput(
                    result=ParseResult.SUCCESS if confidence > 0.7 else ParseResult.PARTIAL_SUCCESS,
                    content=result,
                    confidence=confidence,
                    format_detected=OutputFormat.STRUCTURED_TEXT,
                    parser_used=self.name,
                    raw_input=text
                )
            else:
                self.failure_count += 1
                return ParsedOutput(
                    result=ParseResult.FAILED,
                    content=None,
                    confidence=0.0,
                    format_detected=OutputFormat.PLAIN_TEXT,
                    parser_used=self.name,
                    raw_input=text,
                    error_message="No structured patterns found"
                )
                
        except Exception as e:
            self.failure_count += 1
            return ParsedOutput(
                result=ParseResult.FAILED,
                content=None,
                confidence=0.0,
                format_detected=OutputFormat.PLAIN_TEXT,
                parser_used=self.name,
                raw_input=text,
                error_message=f"Structured text parsing error: {str(e)}"
            )


class CodeBlockParser(BaseParser):
    """Parser for code blocks in various formats."""
    
    def __init__(self):
        super().__init__("CodeBlockParser")
        self.code_patterns = [
            r'```(\w+)?\n(.*?)\n```',      # Fenced code blocks
            r'`([^`]+)`',                   # Inline code
            r'    (.+)',                    # Indented code (4 spaces)
            r'\t(.+)',                      # Tab-indented code
        ]
    
    def can_parse(self, text: str, context: ParsingContext) -> float:
        """Check if text contains code blocks."""
        confidence = 0.0
        
        # Look for code block patterns
        for pattern in self.code_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            confidence += min(0.3, len(matches) * 0.1)
        
        # Check for code indicators
        code_indicators = ['```', 'function', 'def ', 'class ', 'import ', 'const ', 'var ']
        for indicator in code_indicators:
            if indicator in text:
                confidence += 0.1
        
        # Boost if expecting code
        if context.expected_format == OutputFormat.CODE_BLOCK:
            confidence += 0.4
        
        return min(1.0, confidence)
    
    def parse(self, text: str, context: ParsingContext) -> ParsedOutput:
        """Parse code blocks from text."""
        try:
            result = {"code_blocks": [], "inline_code": []}
            
            # Extract fenced code blocks
            fenced_pattern = r'```(\w+)?\n(.*?)\n```'
            fenced_matches = re.findall(fenced_pattern, text, re.DOTALL)
            for language, code in fenced_matches:
                result["code_blocks"].append({
                    "language": language or "unknown",
                    "code": code.strip()
                })
            
            # Extract inline code
            inline_pattern = r'`([^`]+)`'
            inline_matches = re.findall(inline_pattern, text)
            result["inline_code"] = inline_matches
            
            # If only one code block, return it directly
            if len(result["code_blocks"]) == 1 and not result["inline_code"]:
                result = result["code_blocks"][0]["code"]
            
            confidence = 0.8 if result else 0.0
            
            if result:
                self.success_count += 1
                return ParsedOutput(
                    result=ParseResult.SUCCESS,
                    content=result,
                    confidence=confidence,
                    format_detected=OutputFormat.CODE_BLOCK,
                    parser_used=self.name,
                    raw_input=text
                )
            else:
                self.failure_count += 1
                return ParsedOutput(
                    result=ParseResult.FAILED,
                    content=None,
                    confidence=0.0,
                    format_detected=OutputFormat.PLAIN_TEXT,
                    parser_used=self.name,
                    raw_input=text,
                    error_message="No code blocks found"
                )
                
        except Exception as e:
            self.failure_count += 1
            return ParsedOutput(
                result=ParseResult.FAILED,
                content=None,
                confidence=0.0,
                format_detected=OutputFormat.PLAIN_TEXT,
                parser_used=self.name,
                raw_input=text,
                error_message=f"Code block parsing error: {str(e)}"
            )


class PlainTextParser(BaseParser):
    """Fallback parser for plain text content."""
    
    def __init__(self):
        super().__init__("PlainTextParser")
    
    def can_parse(self, text: str, context: ParsingContext) -> float:
        """Always can parse plain text (fallback parser)."""
        return 0.5  # Always available but low priority
    
    def parse(self, text: str, context: ParsingContext) -> ParsedOutput:
        """Parse plain text with basic structure extraction."""
        try:
            result = {
                "raw_text": text.strip(),
                "lines": [line.strip() for line in text.strip().split('\n') if line.strip()],
                "word_count": len(text.split()),
                "character_count": len(text)
            }
            
            # Try to extract some structure
            sentences = re.split(r'[.!?]+', text)
            result["sentences"] = [s.strip() for s in sentences if s.strip()]
            
            # Look for lists
            list_items = re.findall(r'^[-•*]\s*(.+)$', text, re.MULTILINE)
            if list_items:
                result["list_items"] = list_items
            
            # Look for numbered items
            numbered_items = re.findall(r'^\d+\.\s*(.+)$', text, re.MULTILINE)
            if numbered_items:
                result["numbered_items"] = numbered_items
            
            self.success_count += 1
            return ParsedOutput(
                result=ParseResult.SUCCESS,
                content=result,
                confidence=0.5,
                format_detected=OutputFormat.PLAIN_TEXT,
                parser_used=self.name,
                raw_input=text
            )
            
        except Exception as e:
            self.failure_count += 1
            return ParsedOutput(
                result=ParseResult.FAILED,
                content={"raw_text": text},
                confidence=0.2,
                format_detected=OutputFormat.PLAIN_TEXT,
                parser_used=self.name,
                raw_input=text,
                error_message=f"Plain text parsing error: {str(e)}"
            )


class ResilientParsingSystem:
    """
    Main parsing system that coordinates multiple parsers for resilient output handling.
    
    Features:
    - Multiple parsing strategies with automatic fallback
    - Adaptive parser selection based on content and context
    - Learning from successful parsing patterns
    - Performance tracking and optimization
    - Graceful error handling with helpful feedback
    """
    
    def __init__(self):
        self.parsers: List[BaseParser] = [
            JSONParser(),
            XMLTagParser(),
            StructuredTextParser(),
            CodeBlockParser(),
            PlainTextParser()  # Always last (fallback)
        ]
        
        # Performance tracking
        self.parsing_history = []
        self.parser_performance = {}
        
        # Adaptive learning
        self.model_patterns = {}  # Track patterns per model
        self.success_patterns = []
        
        logger.info(f"Resilient Parsing System initialized with {len(self.parsers)} parsers")
    
    def parse(
        self,
        text: str,
        context: Optional[ParsingContext] = None,
        preferred_format: Optional[OutputFormat] = None
    ) -> ParsedOutput:
        """
        Parse text using the most appropriate parser with fallback strategies.
        
        Args:
            text: Raw text to parse
            context: Parsing context with hints and requirements
            preferred_format: Preferred output format
            
        Returns:
            ParsedOutput with result and metadata
        """
        if context is None:
            context = ParsingContext()
        
        if preferred_format:
            context.expected_format = preferred_format
        
        # Track parsing attempt
        start_time = time.time()
        
        # Select best parser for this content
        parser_scores = []
        for parser in self.parsers:
            confidence = parser.can_parse(text, context)
            # Boost score based on parser's historical performance
            performance_boost = self._get_parser_performance_boost(parser)
            final_score = confidence + performance_boost
            parser_scores.append((final_score, parser))
        
        # Sort by score (highest first)
        parser_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Try parsers in order of confidence
        last_error = None
        best_result = None
        
        for score, parser in parser_scores:
            if score < 0.1:  # Skip very low confidence parsers
                continue
            
            try:
                result = parser.parse(text, context)
                
                # Track the attempt
                self._record_parsing_attempt(parser.name, result.result, time.time() - start_time)
                
                if result.result == ParseResult.SUCCESS:
                    # Learn from successful patterns
                    self._learn_from_success(parser, text, context)
                    return result
                elif result.result == ParseResult.PARTIAL_SUCCESS:
                    if best_result is None or result.confidence > best_result.confidence:
                        best_result = result
                
                last_error = result.error_message
                
            except Exception as e:
                logger.error(f"Parser {parser.name} failed: {e}")
                last_error = str(e)
                continue
        
        # Return best partial result if available
        if best_result:
            return best_result
        
        # All parsers failed - return failure with suggestions
        return ParsedOutput(
            result=ParseResult.FAILED,
            content=None,
            confidence=0.0,
            format_detected=OutputFormat.PLAIN_TEXT,
            parser_used="none",
            raw_input=text,
            error_message=last_error or "All parsing strategies failed",
            suggestions=self._generate_parsing_suggestions(text, context)
        )
    
    def _get_parser_performance_boost(self, parser: BaseParser) -> float:
        """Calculate performance boost for parser based on historical success."""
        success_rate = parser.get_success_rate()
        # Convert success rate to a small boost (0.0 to 0.2)
        return success_rate * 0.2
    
    def _record_parsing_attempt(self, parser_name: str, result: ParseResult, duration: float) -> None:
        """Record parsing attempt for performance tracking."""
        attempt = {
            "parser": parser_name,
            "result": result.value,
            "duration": duration,
            "timestamp": time.time()
        }
        
        self.parsing_history.append(attempt)
        
        # Keep only recent history
        if len(self.parsing_history) > 1000:
            self.parsing_history = self.parsing_history[-1000:]
        
        # Update parser performance stats
        if parser_name not in self.parser_performance:
            self.parser_performance[parser_name] = {
                "attempts": 0,
                "successes": 0,
                "partial_successes": 0,
                "failures": 0,
                "avg_duration": 0.0
            }
        
        stats = self.parser_performance[parser_name]
        stats["attempts"] += 1
        
        if result == ParseResult.SUCCESS:
            stats["successes"] += 1
        elif result == ParseResult.PARTIAL_SUCCESS:
            stats["partial_successes"] += 1
        else:
            stats["failures"] += 1
        
        # Update average duration
        stats["avg_duration"] = (stats["avg_duration"] * (stats["attempts"] - 1) + duration) / stats["attempts"]
    
    def _learn_from_success(self, parser: BaseParser, text: str, context: ParsingContext) -> None:
        """Learn from successful parsing to improve future attempts."""
        # Extract pattern from successful text
        pattern_signature = self._extract_pattern_signature(text)
        parser.learn_from_success(pattern_signature)
        
        # Track model-specific patterns
        if context.model_name != "unknown":
            if context.model_name not in self.model_patterns:
                self.model_patterns[context.model_name] = []
            
            self.model_patterns[context.model_name].append({
                "parser": parser.name,
                "pattern": pattern_signature,
                "timestamp": time.time()
            })
            
            # Keep only recent patterns per model
            if len(self.model_patterns[context.model_name]) > 50:
                self.model_patterns[context.model_name] = self.model_patterns[context.model_name][-50:]
    
    def _extract_pattern_signature(self, text: str) -> str:
        """Extract a signature pattern from text for learning."""
        # Simple pattern extraction - could be enhanced with ML
        signature_parts = []
        
        # Check for JSON patterns
        if '{' in text and '}' in text:
            signature_parts.append("json_structure")
        
        # Check for XML patterns
        if '<' in text and '>' in text:
            signature_parts.append("xml_tags")
        
        # Check for code blocks
        if '```' in text:
            signature_parts.append("code_blocks")
        
        # Check for structured text
        if re.search(r'^\w+:\s*', text, re.MULTILINE):
            signature_parts.append("key_value_pairs")
        
        return "_".join(signature_parts) if signature_parts else "plain_text"
    
    def _generate_parsing_suggestions(self, text: str, context: ParsingContext) -> List[str]:
        """Generate helpful suggestions for improving parsing success."""
        suggestions = []
        
        # Analyze text for common issues
        if '{' in text but not text.strip().startswith('{'):
            suggestions.append("Consider wrapping JSON in proper delimiters")
        
        if re.search(r"'\w+'\s*:", text):
            suggestions.append("Use double quotes for JSON keys and values")
        
        if re.search(r'},\s*}', text):
            suggestions.append("Remove trailing commas in JSON objects")
        
        if '<' in text and '>' in text but not re.search(r'<\w+>.*?</\w+>', text):
            suggestions.append("Ensure XML tags are properly closed")
        
        if context.expected_format:
            suggestions.append(f"Consider formatting output as {context.expected_format.value}")
        
        if not suggestions:
            suggestions.append("Consider using structured format like JSON or XML tags")
        
        return suggestions
    
    def get_parser_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about parser performance."""
        stats = {
            "total_attempts": len(self.parsing_history),
            "parser_performance": {},
            "recent_success_rate": 0.0,
            "most_successful_parser": "",
            "average_parsing_time": 0.0,
            "format_distribution": {}
        }
        
        # Calculate parser performance
        for parser_name, perf in self.parser_performance.items():
            success_rate = (perf["successes"] / perf["attempts"]) * 100 if perf["attempts"] > 0 else 0
            stats["parser_performance"][parser_name] = {
                "success_rate": success_rate,
                "attempts": perf["attempts"],
                "avg_duration": perf["avg_duration"]
            }
        
        # Find most successful parser
        best_parser = max(
            self.parser_performance.items(),
            key=lambda x: x[1]["successes"],
            default=("none", {"successes": 0})
        )
        stats["most_successful_parser"] = best_parser[0]
        
        # Calculate recent success rate (last 50 attempts)
        recent_attempts = self.parsing_history[-50:]
        if recent_attempts:
            recent_successes = sum(1 for attempt in recent_attempts 
                                 if attempt["result"] in ["success", "partial_success"])
            stats["recent_success_rate"] = (recent_successes / len(recent_attempts)) * 100
        
        # Calculate average parsing time
        if self.parsing_history:
            total_time = sum(attempt["duration"] for attempt in self.parsing_history)
            stats["average_parsing_time"] = total_time / len(self.parsing_history)
        
        return stats
    
    def optimize_parser_order(self) -> None:
        """Optimize parser order based on historical performance."""
        logger.info("Optimizing parser order based on performance data...")
        
        # Calculate overall performance scores for each parser
        parser_scores = []
        for parser in self.parsers:
            if parser.name == "PlainTextParser":
                # Keep PlainTextParser last (fallback)
                score = -1
            else:
                success_rate = parser.get_success_rate()
                avg_duration = self.parser_performance.get(parser.name, {}).get("avg_duration", 1.0)
                # Score combines success rate and speed (prefer fast, successful parsers)
                score = success_rate - (avg_duration * 0.1)  # Penalty for slow parsers
            
            parser_scores.append((score, parser))
        
        # Sort by score (highest first, except PlainTextParser)
        parser_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Reconstruct parser list (keep PlainTextParser last)
        new_order = []
        plain_text_parser = None
        
        for score, parser in parser_scores:
            if parser.name == "PlainTextParser":
                plain_text_parser = parser
            else:
                new_order.append(parser)
        
        if plain_text_parser:
            new_order.append(plain_text_parser)
        
        self.parsers = new_order
        logger.info("Parser optimization completed")
    
    def add_custom_parser(self, parser: BaseParser) -> None:
        """Add a custom parser to the system."""
        # Insert before PlainTextParser (fallback)
        if self.parsers and self.parsers[-1].name == "PlainTextParser":
            self.parsers.insert(-1, parser)
        else:
            self.parsers.append(parser)
        
        logger.info(f"Added custom parser: {parser.name}")


# Create singleton instance
resilient_parser = ResilientParsingSystem()