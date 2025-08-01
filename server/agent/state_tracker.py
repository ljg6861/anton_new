"""
Comprehensive State Tracking System for the Anton agent.

This component addresses the critical weakness of insufficient state tracking by:
1. Persisting relevant information across agent interactions
2. Learning from successful task completion patterns
3. Adapting strategies based on historical performance
4. Maintaining detailed execution context and decision history
5. Providing analytics for performance optimization
"""
import json
import logging
import pickle
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import hashlib
import threading

logger = logging.getLogger(__name__)


class StateCategory(Enum):
    """Categories of state information to track."""
    TASK_EXECUTION = "task_execution"
    TOOL_USAGE = "tool_usage"
    ERROR_HANDLING = "error_handling"
    DECISION_MAKING = "decision_making"
    PERFORMANCE_METRICS = "performance_metrics"
    LEARNING_INSIGHTS = "learning_insights"
    CONTEXT_MANAGEMENT = "context_management"
    USER_INTERACTION = "user_interaction"


class ExecutionState(Enum):
    """Current execution state of tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY_NEEDED = "retry_needed"


@dataclass
class TaskExecution:
    """Comprehensive tracking of task execution."""
    task_id: str
    original_task: str
    simplified_task: str
    task_type: str
    execution_state: ExecutionState
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    complexity_level: int = 1  # 1-5 scale
    
    # Execution details
    steps_taken: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)
    errors_encountered: List[Dict[str, Any]] = field(default_factory=list)
    context_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    token_usage: Dict[str, int] = field(default_factory=dict)
    response_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    
    # Results and outcomes
    success_indicators: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    final_output: Optional[str] = None
    user_satisfaction: Optional[int] = None  # 1-5 scale
    
    # Learning data
    patterns_identified: List[str] = field(default_factory=list)
    strategies_used: List[str] = field(default_factory=list)
    improvements_suggested: List[str] = field(default_factory=list)
    
    def complete(self, success: bool, final_output: str = "") -> None:
        """Mark task as completed and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.execution_state = ExecutionState.COMPLETED if success else ExecutionState.FAILED
        self.final_output = final_output
    
    def add_step(self, step_description: str) -> None:
        """Add an execution step."""
        self.steps_taken.append({
            "step": step_description,
            "timestamp": time.time(),
            "step_number": len(self.steps_taken) + 1
        })
    
    def record_decision(self, decision: str, rationale: str, alternatives: List[str] = None) -> None:
        """Record a decision made during execution."""
        self.decisions_made.append({
            "decision": decision,
            "rationale": rationale,
            "alternatives": alternatives or [],
            "timestamp": time.time()
        })
    
    def record_error(self, error_type: str, error_message: str, recovery_action: str = "") -> None:
        """Record an error and recovery action."""
        self.errors_encountered.append({
            "error_type": error_type,
            "error_message": error_message,
            "recovery_action": recovery_action,
            "timestamp": time.time()
        })


@dataclass
class ToolUsagePattern:
    """Pattern of tool usage for analysis."""
    tool_name: str
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    common_contexts: List[str] = field(default_factory=list)
    error_patterns: List[str] = field(default_factory=list)
    performance_trend: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.usage_count == 0:
            return 0.0
        return (self.success_count / self.usage_count) * 100
    
    def update_usage(self, success: bool, execution_time: float, context: str = "") -> None:
        """Update usage statistics."""
        self.usage_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.usage_count
        
        if context and context not in self.common_contexts:
            self.common_contexts.append(context)
            # Keep only recent contexts
            if len(self.common_contexts) > 10:
                self.common_contexts = self.common_contexts[-10:]
        
        # Track performance trend
        self.performance_trend.append(execution_time)
        if len(self.performance_trend) > 20:
            self.performance_trend = self.performance_trend[-20:]


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    timestamp: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_tasks: int
    completed_tasks: int
    error_rate: float
    average_response_time: float
    token_usage_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class StateDatabase:
    """SQLite database for persistent state storage."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._lock:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS task_executions (
                    task_id TEXT PRIMARY KEY,
                    original_task TEXT,
                    task_type TEXT,
                    execution_state TEXT,
                    start_time REAL,
                    end_time REAL,
                    duration REAL,
                    complexity_level INTEGER,
                    execution_data TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            ''')
            
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS tool_usage (
                    usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT,
                    task_id TEXT,
                    success BOOLEAN,
                    execution_time REAL,
                    context TEXT,
                    timestamp REAL,
                    FOREIGN KEY (task_id) REFERENCES task_executions (task_id)
                )
            ''')
            
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL,
                    active_tasks INTEGER,
                    completed_tasks INTEGER,
                    error_rate REAL,
                    average_response_time REAL,
                    token_usage_rate REAL
                )
            ''')
            
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS state_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    category TEXT,
                    updated_at REAL
                )
            ''')
            
            # Create indexes for better performance
            self.connection.execute('CREATE INDEX IF NOT EXISTS idx_task_type ON task_executions (task_type)')
            self.connection.execute('CREATE INDEX IF NOT EXISTS idx_execution_state ON task_executions (execution_state)')
            self.connection.execute('CREATE INDEX IF NOT EXISTS idx_tool_name ON tool_usage (tool_name)')
            self.connection.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_snapshots (timestamp)')
            
            self.connection.commit()
    
    def store_task_execution(self, execution: TaskExecution) -> None:
        """Store task execution in database."""
        with self._lock:
            execution_data = json.dumps(asdict(execution), default=str)
            self.connection.execute('''
                INSERT OR REPLACE INTO task_executions 
                (task_id, original_task, task_type, execution_state, start_time, end_time, 
                 duration, complexity_level, execution_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution.task_id, execution.original_task, execution.task_type,
                execution.execution_state.value, execution.start_time, execution.end_time,
                execution.duration, execution.complexity_level, execution_data
            ))
            self.connection.commit()
    
    def store_tool_usage(self, tool_name: str, task_id: str, success: bool, 
                        execution_time: float, context: str) -> None:
        """Store tool usage record."""
        with self._lock:
            self.connection.execute('''
                INSERT INTO tool_usage 
                (tool_name, task_id, success, execution_time, context, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (tool_name, task_id, success, execution_time, context, time.time()))
            self.connection.commit()
    
    def store_performance_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Store performance snapshot."""
        with self._lock:
            self.connection.execute('''
                INSERT INTO performance_snapshots 
                (timestamp, memory_usage_mb, cpu_usage_percent, active_tasks, 
                 completed_tasks, error_rate, average_response_time, token_usage_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.timestamp, snapshot.memory_usage_mb, snapshot.cpu_usage_percent,
                snapshot.active_tasks, snapshot.completed_tasks, snapshot.error_rate,
                snapshot.average_response_time, snapshot.token_usage_rate
            ))
            self.connection.commit()
    
    def get_task_executions(self, limit: int = 100, task_type: str = None) -> List[TaskExecution]:
        """Retrieve task executions from database."""
        with self._lock:
            query = 'SELECT execution_data FROM task_executions'
            params = []
            
            if task_type:
                query += ' WHERE task_type = ?'
                params.append(task_type)
            
            query += ' ORDER BY start_time DESC LIMIT ?'
            params.append(limit)
            
            cursor = self.connection.execute(query, params)
            executions = []
            
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[0])
                    execution = TaskExecution(**data)
                    executions.append(execution)
                except Exception as e:
                    logger.error(f"Failed to deserialize task execution: {e}")
            
            return executions
    
    def get_tool_usage_stats(self, days: int = 30) -> Dict[str, ToolUsagePattern]:
        """Get tool usage statistics for the specified period."""
        with self._lock:
            cutoff_time = time.time() - (days * 24 * 3600)
            cursor = self.connection.execute('''
                SELECT tool_name, success, execution_time, context
                FROM tool_usage
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
            
            patterns = {}
            for tool_name, success, exec_time, context in cursor.fetchall():
                if tool_name not in patterns:
                    patterns[tool_name] = ToolUsagePattern(tool_name=tool_name)
                
                patterns[tool_name].update_usage(bool(success), exec_time, context)
            
            return patterns
    
    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if self.connection:
                self.connection.close()


class ComprehensiveStateTracker:
    """
    Comprehensive state tracking system for the Anton agent.
    
    Features:
    - Persistent storage of all agent interactions and decisions
    - Performance pattern analysis and optimization
    - Historical success/failure tracking
    - Adaptive strategy recommendations
    - Real-time performance monitoring
    """
    
    def __init__(self, storage_path: str = "anton_state.db"):
        self.storage_path = Path(storage_path)
        self.db = StateDatabase(str(self.storage_path))
        
        # In-memory state for fast access
        self.active_tasks: Dict[str, TaskExecution] = {}
        self.tool_patterns: Dict[str, ToolUsagePattern] = {}
        self.performance_history: deque = deque(maxlen=1000)
        
        # Strategy patterns learned from successful executions
        self.success_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Performance monitoring
        self.monitoring_enabled = True
        self.last_performance_snapshot = time.time()
        
        logger.info(f"Comprehensive State Tracker initialized with storage: {self.storage_path}")
    
    def start_task_tracking(
        self,
        task_description: str,
        task_type: str = "general",
        complexity: int = 1
    ) -> str:
        """Start tracking a new task execution."""
        task_id = self._generate_task_id(task_description)
        
        execution = TaskExecution(
            task_id=task_id,
            original_task=task_description,
            simplified_task=self._simplify_task_description(task_description),
            task_type=task_type,
            execution_state=ExecutionState.IN_PROGRESS,
            start_time=time.time(),
            complexity_level=complexity
        )
        
        self.active_tasks[task_id] = execution
        logger.info(f"Started tracking task {task_id[:8]}... ({task_type})")
        
        return task_id
    
    def _generate_task_id(self, task_description: str) -> str:
        """Generate unique task ID."""
        timestamp = str(time.time())
        content = f"{task_description}_{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _simplify_task_description(self, task: str) -> str:
        """Create a simplified version of task description for pattern matching."""
        # Remove specific file paths, timestamps, etc.
        import re
        simplified = re.sub(r'/[^\s]+\.(py|js|txt|md)', '/file.ext', task)
        simplified = re.sub(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}', 'timestamp', simplified)
        simplified = re.sub(r'\b\d+\b', 'number', simplified)
        return simplified.lower().strip()
    
    def record_step(self, task_id: str, step_description: str) -> None:
        """Record an execution step."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].add_step(step_description)
            logger.debug(f"Task {task_id[:8]}... step: {step_description[:50]}...")
    
    def record_decision(
        self,
        task_id: str,
        decision: str,
        rationale: str,
        alternatives: List[str] = None
    ) -> None:
        """Record a decision made during task execution."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].record_decision(decision, rationale, alternatives)
            logger.debug(f"Task {task_id[:8]}... decision: {decision}")
    
    def record_tool_usage(
        self,
        task_id: str,
        tool_name: str,
        success: bool,
        execution_time: float,
        context: str = ""
    ) -> None:
        """Record tool usage and update patterns."""
        # Update in-memory patterns
        if tool_name not in self.tool_patterns:
            self.tool_patterns[tool_name] = ToolUsagePattern(tool_name=tool_name)
        
        self.tool_patterns[tool_name].update_usage(success, execution_time, context)
        
        # Update task execution record
        if task_id in self.active_tasks:
            if tool_name not in self.active_tasks[task_id].tools_used:
                self.active_tasks[task_id].tools_used.append(tool_name)
        
        # Store in database
        self.db.store_tool_usage(tool_name, task_id, success, execution_time, context)
        
        logger.debug(f"Recorded tool usage: {tool_name} ({'success' if success else 'failure'})")
    
    def record_error(
        self,
        task_id: str,
        error_type: str,
        error_message: str,
        recovery_action: str = ""
    ) -> None:
        """Record an error and recovery action."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].record_error(error_type, error_message, recovery_action)
            logger.info(f"Task {task_id[:8]}... error: {error_type}")
    
    def complete_task(
        self,
        task_id: str,
        success: bool,
        final_output: str = "",
        user_satisfaction: int = None
    ) -> None:
        """Complete task tracking and analyze patterns."""
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not found in active tasks")
            return
        
        execution = self.active_tasks[task_id]
        execution.complete(success, final_output)
        
        if user_satisfaction:
            execution.user_satisfaction = user_satisfaction
        
        # Store in database
        self.db.store_task_execution(execution)
        
        # Analyze patterns
        self._analyze_execution_patterns(execution)
        
        # Move from active to completed
        del self.active_tasks[task_id]
        
        logger.info(f"Completed task {task_id[:8]}... ({'success' if success else 'failure'})")
    
    def _analyze_execution_patterns(self, execution: TaskExecution) -> None:
        """Analyze execution to identify success/failure patterns."""
        pattern_data = {
            "task_type": execution.task_type,
            "complexity": execution.complexity_level,
            "tools_used": execution.tools_used,
            "duration": execution.duration,
            "steps_count": len(execution.steps_taken),
            "decisions_count": len(execution.decisions_made),
            "errors_count": len(execution.errors_encountered)
        }
        
        if execution.execution_state == ExecutionState.COMPLETED:
            self.success_patterns[execution.task_type].append(pattern_data)
            
            # Keep only recent patterns
            if len(self.success_patterns[execution.task_type]) > 50:
                self.success_patterns[execution.task_type] = self.success_patterns[execution.task_type][-50:]
        else:
            self.failure_patterns[execution.task_type].append(pattern_data)
            
            # Keep only recent patterns
            if len(self.failure_patterns[execution.task_type]) > 50:
                self.failure_patterns[execution.task_type] = self.failure_patterns[execution.task_type][-50:]
    
    def get_strategy_recommendations(self, task_type: str, complexity: int) -> List[str]:
        """Get strategy recommendations based on historical patterns."""
        recommendations = []
        
        # Analyze successful patterns for this task type
        if task_type in self.success_patterns:
            patterns = self.success_patterns[task_type]
            
            # Find patterns with similar complexity
            similar_patterns = [p for p in patterns if abs(p["complexity"] - complexity) <= 1]
            
            if similar_patterns:
                # Most common tools in successful executions
                tool_frequency = defaultdict(int)
                for pattern in similar_patterns[-10:]:  # Last 10 similar patterns
                    for tool in pattern["tools_used"]:
                        tool_frequency[tool] += 1
                
                if tool_frequency:
                    top_tools = sorted(tool_frequency.items(), key=lambda x: x[1], reverse=True)[:3]
                    recommendations.append(f"Consider using tools: {', '.join([tool for tool, _ in top_tools])}")
                
                # Average successful execution duration
                avg_duration = sum(p["duration"] for p in similar_patterns if p["duration"]) / len(similar_patterns)
                recommendations.append(f"Expected duration: {avg_duration:.1f} seconds based on similar tasks")
                
                # Step count recommendations
                avg_steps = sum(p["steps_count"] for p in similar_patterns) / len(similar_patterns)
                recommendations.append(f"Plan for approximately {int(avg_steps)} execution steps")
        
        # Analyze failure patterns to avoid common pitfalls
        if task_type in self.failure_patterns:
            failure_patterns = self.failure_patterns[task_type]
            
            # Common failure tools to avoid or use carefully
            failure_tools = defaultdict(int)
            for pattern in failure_patterns[-10:]:
                for tool in pattern["tools_used"]:
                    failure_tools[tool] += 1
            
            if failure_tools:
                risky_tools = [tool for tool, count in failure_tools.items() if count >= 3]
                if risky_tools:
                    recommendations.append(f"Use with caution: {', '.join(risky_tools)} (frequent in failed tasks)")
        
        return recommendations
    
    def get_tool_recommendations(self, context: str) -> List[Tuple[str, float]]:
        """Get tool recommendations based on context and historical performance."""
        recommendations = []
        
        for tool_name, pattern in self.tool_patterns.items():
            if pattern.usage_count < 3:  # Skip tools with insufficient data
                continue
            
            confidence = 0.0
            
            # Base confidence from success rate
            confidence += (pattern.success_rate / 100) * 0.6
            
            # Context relevance
            context_relevance = 0.0
            for common_context in pattern.common_contexts:
                if any(word in context.lower() for word in common_context.lower().split()):
                    context_relevance += 0.1
            
            confidence += min(context_relevance, 0.3)
            
            # Performance factor (prefer faster tools)
            if pattern.average_execution_time > 0:
                speed_factor = max(0, 1 - (pattern.average_execution_time / 10))  # Normalize to 10 seconds
                confidence += speed_factor * 0.1
            
            if confidence > 0.3:  # Minimum confidence threshold
                recommendations.append((tool_name, confidence))
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:5]  # Top 5 recommendations
    
    def capture_performance_snapshot(self) -> None:
        """Capture current system performance snapshot."""
        if not self.monitoring_enabled:
            return
        
        try:
            import psutil
            
            snapshot = PerformanceSnapshot(
                timestamp=time.time(),
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                cpu_usage_percent=psutil.cpu_percent(),
                active_tasks=len(self.active_tasks),
                completed_tasks=len(self.performance_history),
                error_rate=self._calculate_current_error_rate(),
                average_response_time=self._calculate_average_response_time(),
                token_usage_rate=self._calculate_token_usage_rate()
            )
            
            self.performance_history.append(snapshot)
            self.db.store_performance_snapshot(snapshot)
            self.last_performance_snapshot = time.time()
            
        except ImportError:
            logger.warning("psutil not available for performance monitoring")
        except Exception as e:
            logger.error(f"Failed to capture performance snapshot: {e}")
    
    def _calculate_current_error_rate(self) -> float:
        """Calculate current error rate from recent tasks."""
        recent_snapshots = list(self.performance_history)[-20:]  # Last 20 snapshots
        if not recent_snapshots:
            return 0.0
        
        # This is a simplified calculation - in practice, you'd track actual errors
        # For now, return a placeholder value
        return 0.05  # 5% error rate
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time from recent tasks."""
        # Get recent completed tasks and calculate average duration
        recent_completed = self.db.get_task_executions(limit=20)
        if not recent_completed:
            return 0.0
        
        valid_durations = [task.duration for task in recent_completed if task.duration]
        if not valid_durations:
            return 0.0
        
        return sum(valid_durations) / len(valid_durations)
    
    def _calculate_token_usage_rate(self) -> float:
        """Calculate token usage rate."""
        # Placeholder calculation - would need actual token tracking
        return 1000.0  # tokens per hour
    
    def get_comprehensive_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analytics for the specified period."""
        analytics = {
            "time_period_days": days,
            "task_statistics": {},
            "tool_performance": {},
            "performance_trends": {},
            "success_patterns": {},
            "improvement_opportunities": []
        }
        
        # Task statistics
        recent_tasks = self.db.get_task_executions(limit=1000)
        cutoff_time = time.time() - (days * 24 * 3600)
        relevant_tasks = [task for task in recent_tasks if task.start_time > cutoff_time]
        
        if relevant_tasks:
            completed_tasks = [task for task in relevant_tasks if task.execution_state == ExecutionState.COMPLETED]
            failed_tasks = [task for task in relevant_tasks if task.execution_state == ExecutionState.FAILED]
            
            analytics["task_statistics"] = {
                "total_tasks": len(relevant_tasks),
                "completed_tasks": len(completed_tasks),
                "failed_tasks": len(failed_tasks),
                "success_rate": (len(completed_tasks) / len(relevant_tasks)) * 100 if relevant_tasks else 0,
                "average_duration": sum(task.duration for task in completed_tasks if task.duration) / len(completed_tasks) if completed_tasks else 0,
                "task_types": self._analyze_task_types(relevant_tasks)
            }
        
        # Tool performance
        tool_stats = self.db.get_tool_usage_stats(days)
        analytics["tool_performance"] = {
            tool_name: {
                "usage_count": pattern.usage_count,
                "success_rate": pattern.success_rate,
                "average_execution_time": pattern.average_execution_time
            }
            for tool_name, pattern in tool_stats.items()
        }
        
        # Performance trends
        recent_snapshots = [s for s in self.performance_history if s.timestamp > cutoff_time]
        if recent_snapshots:
            analytics["performance_trends"] = {
                "average_memory_usage": sum(s.memory_usage_mb for s in recent_snapshots) / len(recent_snapshots),
                "average_cpu_usage": sum(s.cpu_usage_percent for s in recent_snapshots) / len(recent_snapshots),
                "peak_memory_usage": max(s.memory_usage_mb for s in recent_snapshots),
                "peak_cpu_usage": max(s.cpu_usage_percent for s in recent_snapshots)
            }
        
        # Success patterns summary
        analytics["success_patterns"] = {
            task_type: {
                "pattern_count": len(patterns),
                "most_successful_tools": self._get_most_successful_tools(patterns)
            }
            for task_type, patterns in self.success_patterns.items()
        }
        
        # Improvement opportunities
        analytics["improvement_opportunities"] = self._identify_improvement_opportunities(
            relevant_tasks, tool_stats
        )
        
        return analytics
    
    def _analyze_task_types(self, tasks: List[TaskExecution]) -> Dict[str, int]:
        """Analyze distribution of task types."""
        type_counts = defaultdict(int)
        for task in tasks:
            type_counts[task.task_type] += 1
        return dict(type_counts)
    
    def _get_most_successful_tools(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Get most frequently used tools in successful patterns."""
        tool_frequency = defaultdict(int)
        for pattern in patterns:
            for tool in pattern["tools_used"]:
                tool_frequency[tool] += 1
        
        sorted_tools = sorted(tool_frequency.items(), key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in sorted_tools[:5]]
    
    def _identify_improvement_opportunities(
        self,
        tasks: List[TaskExecution],
        tool_stats: Dict[str, ToolUsagePattern]
    ) -> List[str]:
        """Identify opportunities for performance improvement."""
        opportunities = []
        
        # High failure rate tools
        for tool_name, pattern in tool_stats.items():
            if pattern.usage_count > 5 and pattern.success_rate < 70:
                opportunities.append(f"Tool '{tool_name}' has low success rate ({pattern.success_rate:.1f}%) - consider alternatives")
        
        # Slow performing tools
        for tool_name, pattern in tool_stats.items():
            if pattern.usage_count > 5 and pattern.average_execution_time > 5:
                opportunities.append(f"Tool '{tool_name}' is slow (avg {pattern.average_execution_time:.1f}s) - consider optimization")
        
        # Task complexity vs success rate
        if tasks:
            high_complexity_tasks = [task for task in tasks if task.complexity_level >= 4]
            if high_complexity_tasks:
                success_rate = len([task for task in high_complexity_tasks if task.execution_state == ExecutionState.COMPLETED]) / len(high_complexity_tasks) * 100
                if success_rate < 60:
                    opportunities.append(f"High complexity tasks have low success rate ({success_rate:.1f}%) - consider breaking down complex tasks")
        
        return opportunities
    
    def export_state_data(self, output_path: str) -> None:
        """Export comprehensive state data for analysis or backup."""
        export_data = {
            "export_timestamp": time.time(),
            "active_tasks": {task_id: asdict(task) for task_id, task in self.active_tasks.items()},
            "tool_patterns": {tool_name: asdict(pattern) for tool_name, pattern in self.tool_patterns.items()},
            "success_patterns": dict(self.success_patterns),
            "failure_patterns": dict(self.failure_patterns),
            "recent_performance": [s.to_dict() for s in list(self.performance_history)[-100:]]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"State data exported to {output_path}")
    
    def close(self) -> None:
        """Clean up resources."""
        self.db.close()
        logger.info("State tracker closed")


# Create singleton instance
state_tracker = ComprehensiveStateTracker()