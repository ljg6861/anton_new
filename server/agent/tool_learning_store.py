"""
Tool Learning Store - Dedicated knowledge management for tool failures and successful alternatives.
This system allows the LLM to learn from tool failures and discover better approaches.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class ToolOutcome(Enum):
    """Possible outcomes of tool execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"

@dataclass
class ToolExecutionRecord:
    """Record of a single tool execution attempt"""
    tool_name: str
    arguments: Dict[str, Any]
    outcome: ToolOutcome
    result: str
    timestamp: float
    conversation_id: str
    execution_id: str
    error_details: Optional[str] = None
    success_indicators: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "outcome": self.outcome.value,
            "result": self.result,
            "timestamp": self.timestamp,
            "conversation_id": self.conversation_id,
            "execution_id": self.execution_id,
            "error_details": self.error_details,
            "success_indicators": self.success_indicators
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolExecutionRecord':
        """Create instance from dictionary"""
        return cls(
            tool_name=data["tool_name"],
            arguments=data["arguments"],
            outcome=ToolOutcome(data["outcome"]),
            result=data["result"],
            timestamp=data["timestamp"],
            conversation_id=data["conversation_id"],
            execution_id=data["execution_id"],
            error_details=data.get("error_details"),
            success_indicators=data.get("success_indicators")
        )

@dataclass
class ToolLearning:
    """A learned pattern about tool usage"""
    learning_id: str
    failure_pattern: str
    successful_alternative: str
    confidence: float
    context_pattern: str
    tool_names_involved: List[str]
    created_timestamp: float
    last_confirmed_timestamp: float
    confirmation_count: int
    llm_analysis: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolLearning':
        """Create instance from dictionary"""
        return cls(**data)

class ToolLearningStore:
    """
    Dedicated storage for tool failure patterns and successful alternatives.
    Uses SQLite for persistence and provides LLM-driven learning analysis.
    """
    
    def __init__(self, db_path: str = "tool_learning.db"):
        self.db_path = db_path
        self.current_conversation_id = None
        self.current_execution_sequence = []
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for tool learning storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Tool execution records table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tool_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tool_name TEXT NOT NULL,
                        arguments TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        result TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        conversation_id TEXT NOT NULL,
                        execution_id TEXT NOT NULL,
                        error_details TEXT,
                        success_indicators TEXT
                    )
                """)
                
                # Tool learnings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tool_learnings (
                        learning_id TEXT PRIMARY KEY,
                        failure_pattern TEXT NOT NULL,
                        successful_alternative TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        context_pattern TEXT NOT NULL,
                        tool_names_involved TEXT NOT NULL,
                        created_timestamp REAL NOT NULL,
                        last_confirmed_timestamp REAL NOT NULL,
                        confirmation_count INTEGER NOT NULL,
                        llm_analysis TEXT NOT NULL
                    )
                """)
                
                # Create indexes for efficient queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_name ON tool_executions(tool_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation ON tool_executions(conversation_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcome ON tool_executions(outcome)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON tool_executions(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_involved ON tool_learnings(tool_names_involved)")
                
                conn.commit()
                logger.info("Tool learning database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize tool learning database: {e}", exc_info=True)
    
    def start_conversation(self, conversation_id: str):
        """Start tracking a new conversation"""
        with self.lock:
            self.current_conversation_id = conversation_id
            self.current_execution_sequence = []
            logger.info(f"Started tool learning tracking for conversation: {conversation_id}")
    
    def record_tool_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str,
        outcome: ToolOutcome,
        execution_id: str,
        error_details: Optional[str] = None
    ) -> Tuple[str, List[ToolLearning]]:
        """
        Record a tool execution immediately when it happens.
        Returns the execution record ID and suggested alternatives if it's a failure.
        """
        if not self.current_conversation_id:
            # Auto-generate conversation ID if not set
            self.current_conversation_id = f"conv_{int(time.time())}"
        
        record = ToolExecutionRecord(
            tool_name=tool_name,
            arguments=arguments,
            outcome=outcome,
            result=result,
            timestamp=time.time(),
            conversation_id=self.current_conversation_id,
            execution_id=execution_id,
            error_details=error_details
        )
        
        with self.lock:
            self.current_execution_sequence.append(record)
        
        # Store to database immediately
        self._store_execution_record(record)
        
        # If this is a failure, immediately analyze for quick learning AND corrective action
        suggested_alternatives = []
        if outcome == ToolOutcome.FAILURE:
            suggested_alternatives = self._trigger_immediate_failure_analysis(record)
        
        return execution_id, suggested_alternatives
    
    def _store_execution_record(self, record: ToolExecutionRecord):
        """Store execution record to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO tool_executions (
                        tool_name, arguments, outcome, result, timestamp,
                        conversation_id, execution_id, error_details, success_indicators
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.tool_name,
                    json.dumps(record.arguments),
                    record.outcome.value,
                    record.result,
                    record.timestamp,
                    record.conversation_id,
                    record.execution_id,
                    record.error_details,
                    json.dumps(record.success_indicators) if record.success_indicators else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store execution record: {e}", exc_info=True)
    
    def _trigger_immediate_failure_analysis(self, failure_record: ToolExecutionRecord):
        """Trigger immediate analysis when a tool fails and suggest corrective actions"""
        logger.info(f"Tool failure detected: {failure_record.tool_name} - triggering immediate analysis")
        
        # Check if we have any similar past successes that could be alternatives
        similar_successes = self._find_similar_successful_executions(failure_record)
        
        if similar_successes:
            logger.info(f"Found {len(similar_successes)} similar successful executions for analysis")
        
        # More importantly, check for existing high-confidence learnings for immediate corrective action
        existing_learnings = self.query_relevant_learnings(
            failure_record.tool_name, 
            failure_record.arguments,
            context=f"Tool {failure_record.tool_name} just failed: {failure_record.error_details}"
        )
        
        # Return suggested alternatives for immediate action
        high_confidence_alternatives = [
            learning for learning in existing_learnings 
            if learning.confidence > 0.8
        ]
        
        if high_confidence_alternatives:
            logger.info(f"Found {len(high_confidence_alternatives)} high-confidence alternatives for immediate corrective action")
            return high_confidence_alternatives
        
        return []
    
    def analyze_failure_success_pattern(
        self,
        failure_execution_id: str,
        success_execution_id: str,
        llm_analysis_callback
    ) -> Optional[ToolLearning]:
        """
        Analyze a failure-success pattern using LLM to extract learning.
        
        Args:
            failure_execution_id: ID of the failed execution
            success_execution_id: ID of the successful execution
            llm_analysis_callback: Function to call LLM for analysis
        
        Returns:
            ToolLearning object if pattern is worth learning, None otherwise
        """
        failure_record = self._get_execution_record(failure_execution_id)
        success_record = self._get_execution_record(success_execution_id)
        
        if not failure_record or not success_record:
            logger.warning("Could not find both failure and success records for analysis")
            return None
        
        # Prepare context for LLM analysis
        analysis_context = {
            "failure": failure_record.to_dict(),
            "success": success_record.to_dict(),
            "conversation_context": self._get_conversation_context(failure_record.conversation_id),
            "similar_past_patterns": self._find_similar_patterns(failure_record.tool_name)
        }
        
        # Generate LLM analysis prompt
        analysis_prompt = self._build_learning_analysis_prompt(analysis_context)
        
        try:
            # Call LLM for analysis
            llm_response = llm_analysis_callback(analysis_prompt)
            
            # Parse LLM response to extract learning
            learning = self._parse_llm_learning_response(llm_response, failure_record, success_record)
            
            if learning:
                # Check if a similar learning already exists before storing
                if not self._learning_already_exists(learning):
                    self._store_learning(learning)
                    logger.info(f"New tool learning created: {learning.learning_id}")
                    return learning
                else:
                    logger.info(f"Similar learning already exists, skipping duplicate storage")
                    return None  # Return None to indicate no new learning was created
            
        except Exception as e:
            logger.error(f"Failed to analyze failure-success pattern: {e}", exc_info=True)
        
        return None
    
    def query_relevant_learnings(self, tool_name: str, arguments: Dict[str, Any], context: str = "") -> List[ToolLearning]:
        """
        Query for relevant past learnings before attempting a tool execution.
        
        Args:
            tool_name: Name of the tool about to be executed
            arguments: Arguments for the tool
            context: Current context/conversation context
        
        Returns:
            List of relevant ToolLearning objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Query learnings involving this tool
                cursor.execute("""
                    SELECT * FROM tool_learnings 
                    WHERE tool_names_involved LIKE ? 
                    ORDER BY confidence DESC, last_confirmed_timestamp DESC
                """, (f"%{tool_name}%",))
                
                learnings = []
                for row in cursor.fetchall():
                    learning_data = {
                        "learning_id": row[0],
                        "failure_pattern": row[1],
                        "successful_alternative": row[2],
                        "confidence": row[3],
                        "context_pattern": row[4],
                        "tool_names_involved": json.loads(row[5]),
                        "created_timestamp": row[6],
                        "last_confirmed_timestamp": row[7],
                        "confirmation_count": row[8],
                        "llm_analysis": row[9]
                    }
                    learnings.append(ToolLearning.from_dict(learning_data))
                
                # Filter for most relevant based on argument similarity and context
                relevant_learnings = self._filter_relevant_learnings(learnings, tool_name, arguments, context)
                
                return relevant_learnings[:5]  # Return top 5 most relevant
                
        except Exception as e:
            logger.error(f"Failed to query relevant learnings: {e}", exc_info=True)
            return []
    
    def _build_learning_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build a prompt for LLM to analyze failure-success patterns"""
        return f"""
You are analyzing a tool execution failure followed by a successful alternative to extract learning patterns.

FAILURE EXECUTION:
Tool: {context['failure']['tool_name']}
Arguments: {json.dumps(context['failure']['arguments'], indent=2)}
Result: {context['failure']['result']}
Error: {context['failure'].get('error_details', 'N/A')}

SUCCESS EXECUTION:
Tool: {context['success']['tool_name']}
Arguments: {json.dumps(context['success']['arguments'], indent=2)}
Result: {context['success']['result']}

CONVERSATION CONTEXT:
{context.get('conversation_context', 'No additional context')}

SIMILAR PAST PATTERNS:
{json.dumps(context.get('similar_past_patterns', []), indent=2)}

Please analyze this failure-success pattern and determine if there's a valuable learning. Consider:
1. Is this a systematic failure that could happen again?
2. Is the successful alternative consistently better?
3. What are the key differences between the approaches?
4. Under what conditions should the alternative be preferred?

Respond with a JSON object containing:
{{
    "is_learnable": true/false,
    "failure_pattern": "Concise description of what failed and why",
    "successful_alternative": "Description of the better approach",
    "confidence": 0.0-1.0,
    "context_pattern": "When this learning applies",
    "key_insights": "Important insights about this pattern"
}}

If is_learnable is false, briefly explain why this pattern isn't worth learning.
"""
    
    def _parse_llm_learning_response(
        self, 
        llm_response: str, 
        failure_record: ToolExecutionRecord, 
        success_record: ToolExecutionRecord
    ) -> Optional[ToolLearning]:
        """Parse LLM response to create a ToolLearning object"""
        try:
            # Extract JSON from LLM response
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM learning analysis response")
                return None
            
            analysis = json.loads(json_match.group())
            
            if not analysis.get("is_learnable", False):
                logger.info(f"LLM determined pattern is not learnable: {analysis.get('reason', 'No reason provided')}")
                return None
            
            learning_id = f"learning_{int(time.time())}_{failure_record.tool_name}"
            
            learning = ToolLearning(
                learning_id=learning_id,
                failure_pattern=analysis["failure_pattern"],
                successful_alternative=analysis["successful_alternative"],
                confidence=float(analysis["confidence"]),
                context_pattern=analysis["context_pattern"],
                tool_names_involved=[failure_record.tool_name, success_record.tool_name],
                created_timestamp=time.time(),
                last_confirmed_timestamp=time.time(),
                confirmation_count=1,
                llm_analysis=json.dumps(analysis)
            )
            
            return learning
            
        except Exception as e:
            logger.error(f"Failed to parse LLM learning response: {e}", exc_info=True)
            return None
    
    def _store_learning(self, learning: ToolLearning):
        """Store a learning to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO tool_learnings (
                        learning_id, failure_pattern, successful_alternative, confidence,
                        context_pattern, tool_names_involved, created_timestamp,
                        last_confirmed_timestamp, confirmation_count, llm_analysis
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    learning.learning_id,
                    learning.failure_pattern,
                    learning.successful_alternative,
                    learning.confidence,
                    learning.context_pattern,
                    json.dumps(learning.tool_names_involved),
                    learning.created_timestamp,
                    learning.last_confirmed_timestamp,
                    learning.confirmation_count,
                    learning.llm_analysis
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store learning: {e}", exc_info=True)
    
    def _get_execution_record(self, execution_id: str) -> Optional[ToolExecutionRecord]:
        """Retrieve an execution record by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM tool_executions WHERE execution_id = ?
                """, (execution_id,))
                
                row = cursor.fetchone()
                if row:
                    return ToolExecutionRecord(
                        tool_name=row[1],
                        arguments=json.loads(row[2]),
                        outcome=ToolOutcome(row[3]),
                        result=row[4],
                        timestamp=row[5],
                        conversation_id=row[6],
                        execution_id=row[7],
                        error_details=row[8],
                        success_indicators=json.loads(row[9]) if row[9] else None
                    )
        except Exception as e:
            logger.error(f"Failed to get execution record: {e}", exc_info=True)
        return None
    
    def _find_similar_successful_executions(self, failure_record: ToolExecutionRecord) -> List[ToolExecutionRecord]:
        """Find similar successful executions that could be alternatives"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM tool_executions 
                    WHERE outcome = 'success' 
                    AND timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT 20
                """, (failure_record.timestamp - 3600,))  # Within last hour
                
                successes = []
                for row in cursor.fetchall():
                    record = ToolExecutionRecord(
                        tool_name=row[1],
                        arguments=json.loads(row[2]),
                        outcome=ToolOutcome(row[3]),
                        result=row[4],
                        timestamp=row[5],
                        conversation_id=row[6],
                        execution_id=row[7],
                        error_details=row[8],
                        success_indicators=json.loads(row[9]) if row[9] else None
                    )
                    successes.append(record)
                
                return successes
        except Exception as e:
            logger.error(f"Failed to find similar successful executions: {e}", exc_info=True)
            return []
    
    def _get_conversation_context(self, conversation_id: str) -> str:
        """Get relevant context from the conversation"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT tool_name, arguments, outcome, result FROM tool_executions 
                    WHERE conversation_id = ? 
                    ORDER BY timestamp
                """, (conversation_id,))
                
                context_parts = []
                for row in cursor.fetchall():
                    context_parts.append(f"{row[0]}({json.dumps(row[1])}) -> {row[2]}: {row[3][:100]}...")
                
                return "\n".join(context_parts[-10:])  # Last 10 executions
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}", exc_info=True)
            return ""
    
    def _find_similar_patterns(self, tool_name: str) -> List[Dict[str, Any]]:
        """Find similar patterns for the given tool"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM tool_learnings 
                    WHERE tool_names_involved LIKE ?
                """, (f"%{tool_name}%",))
                
                patterns = []
                for row in cursor.fetchall():
                    patterns.append({
                        "pattern": row[1],
                        "alternative": row[2],
                        "confidence": row[3],
                        "confirmations": row[8]
                    })
                
                return patterns
        except Exception as e:
            logger.error(f"Failed to find similar patterns: {e}", exc_info=True)
            return []
    
    def _filter_relevant_learnings(
        self, 
        learnings: List[ToolLearning], 
        tool_name: str, 
        arguments: Dict[str, Any], 
        context: str
    ) -> List[ToolLearning]:
        """Filter learnings for relevance to current execution context and deduplicate"""
        relevant = []
        seen_patterns = set()  # Track patterns we've already seen
        
        for learning in learnings:
            if tool_name in learning.tool_names_involved and learning.confidence > 0.5:
                # Create a unique key based on failure pattern and alternative
                pattern_key = f"{learning.failure_pattern}|{learning.successful_alternative}"
                
                # Only add if we haven't seen this exact pattern before
                if pattern_key not in seen_patterns:
                    relevant.append(learning)
                    seen_patterns.add(pattern_key)
        
        return sorted(relevant, key=lambda x: x.confidence, reverse=True)

    def _learning_already_exists(self, new_learning: ToolLearning) -> bool:
        """Check if a similar learning already exists in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Query for learnings with similar patterns involving the same tools
                tools_json = json.dumps(new_learning.tool_names_involved)
                cursor.execute("""
                    SELECT failure_pattern, successful_alternative FROM tool_learnings 
                    WHERE tool_names_involved = ?
                """, (tools_json,))
                
                for row in cursor.fetchall():
                    existing_failure = row[0]
                    existing_alternative = row[1]
                    
                    # Check for semantic similarity (simple string comparison for now)
                    if (self._patterns_are_similar(new_learning.failure_pattern, existing_failure) and
                        self._patterns_are_similar(new_learning.successful_alternative, existing_alternative)):
                        return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to check for existing learning: {e}", exc_info=True)
            return False
    
    def _learnings_are_similar(self, learning1: ToolLearning, learning2: ToolLearning) -> bool:
        """Check if two learnings are similar enough to be considered duplicates"""
        return (
            self._patterns_are_similar(learning1.failure_pattern, learning2.failure_pattern) and
            self._patterns_are_similar(learning1.successful_alternative, learning2.successful_alternative) and
            set(learning1.tool_names_involved) == set(learning2.tool_names_involved)
        )
    
    def _patterns_are_similar(self, pattern1: str, pattern2: str, threshold: float = 0.8) -> bool:
        """
        Check if two patterns are similar using simple string similarity.
        This could be enhanced with more sophisticated NLP techniques.
        """
        # Simple approach: check if the patterns have significant word overlap
        words1 = set(pattern1.lower().split())
        words2 = set(pattern2.lower().split())
        
        if not words1 or not words2:
            return pattern1.strip() == pattern2.strip()
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold

# Global instance
tool_learning_store = ToolLearningStore()
