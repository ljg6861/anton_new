"""
Structured state management for ReAct Agent.
Follows the design pattern of centralizing all agent state in structured objects.
"""
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class AgentStatus(Enum):
    """Agent execution status"""
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    CRITIQUING = "CRITIQUING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class ContextBlob:
    """A piece of context with metadata"""
    content: str
    source: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    context_type: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallTrace:
    """Trace of a tool call execution"""
    name: str
    arguments: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    result: Optional[Any] = None
    success: bool = False
    error_message: Optional[str] = None
    cost: float = 0.0
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate call duration in seconds"""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None
    
    def mark_complete(self, result: Any, success: bool = True, cost: float = 0.0, error: Optional[str] = None):
        """Mark the tool call as complete"""
        self.end_time = time.time()
        self.result = result
        self.success = success
        self.cost = cost
        self.error_message = error


@dataclass
class Evidence:
    """Evidence collected during agent execution"""
    type: str  # "file_content", "tool_output", "observation", etc.
    content: str
    source: str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Budgets:
    """Resource budgets for agent execution"""
    # Token budgets
    total_tokens: int = 8192
    system_tools_pct: float = 0.15
    domain_bundle_pct: float = 0.30
    session_summary_pct: float = 0.15
    working_memory_pct: float = 0.35
    scratchpad_pct: float = 0.05
    
    # Time and iteration budgets
    max_iterations: int = 10
    per_tool_timeout: float = 30.0
    total_timeout: float = 300.0
    
    # Cost budgets
    max_cost: float = 1.0
    
    @property
    def system_tools_budget(self) -> int:
        return int(self.total_tokens * self.system_tools_pct)
    
    @property
    def domain_bundle_budget(self) -> int:
        return int(self.total_tokens * self.domain_bundle_pct)
    
    @property
    def session_summary_budget(self) -> int:
        return int(self.total_tokens * self.session_summary_pct)
    
    @property
    def working_memory_budget(self) -> int:
        return int(self.total_tokens * self.working_memory_pct)
    
    @property
    def scratchpad_budget(self) -> int:
        return int(self.total_tokens * self.scratchpad_pct)


@dataclass
class State:
    """Central state object for ReAct Agent execution"""
    goal: str
    context: List[ContextBlob] = field(default_factory=list)
    tool_calls: List[ToolCallTrace] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    turns: int = 0
    cost: float = 0.0
    status: AgentStatus = AgentStatus.PLANNING
    budgets: Budgets = field(default_factory=Budgets)
    
    # Additional state tracking
    start_time: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    scratchpad: str = ""
    
    # Memory components
    working_memory: List[Dict[str, str]] = field(default_factory=list)
    session_decisions: List[str] = field(default_factory=list)
    session_todos: List[str] = field(default_factory=list)
    session_context: str = ""
    
    # Orchestration fields
    subgoals: List[str] = field(default_factory=list)
    route: Optional[str] = None
    
    def add_context(self, content: str, source: str, importance: float = 1.0, 
                   context_type: str = "general", metadata: Optional[Dict[str, Any]] = None):
        """Add context to state"""
        blob = ContextBlob(
            content=content,
            source=source,
            importance=importance,
            context_type=context_type,
            metadata=metadata or {}
        )
        self.context.append(blob)
    
    def add_evidence(self, evidence_type: str, content: str, source: str, 
                    confidence: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        """Add evidence to state"""
        evidence = Evidence(
            type=evidence_type,
            content=content,
            source=source,
            confidence=confidence,
            metadata=metadata or {}
        )
        self.evidence.append(evidence)
    
    def start_tool_call(self, name: str, arguments: Dict[str, Any]) -> ToolCallTrace:
        """Start tracking a tool call"""
        trace = ToolCallTrace(
            name=name,
            arguments=arguments,
            start_time=time.time()
        )
        self.tool_calls.append(trace)
        return trace
    
    def complete_tool_call(self, trace: ToolCallTrace, result: Any, success: bool = True, 
                          cost: float = 0.0, error: Optional[str] = None):
        """Complete a tool call and update state"""
        trace.mark_complete(result, success, cost, error)
        self.cost += cost
        
        # Add tool result as evidence
        self.add_evidence(
            evidence_type="tool_output",
            content=str(result)[:1000] if result else "",
            source=f"tool_{trace.name}",
            confidence=1.0 if success else 0.5,
            metadata={"tool_call_trace": trace, "arguments": trace.arguments}
        )
    
    def increment_turn(self):
        """Increment turn counter"""
        self.turns += 1
    
    def set_status(self, status: AgentStatus):
        """Update agent status"""
        self.status = status
    
    def add_to_scratchpad(self, content: str):
        """Add content to scratchpad"""
        self.scratchpad += content + "\n"
    
    def add_decision(self, decision: str):
        """Add a decision to session memory"""
        self.session_decisions.append(decision)
        if len(self.session_decisions) > 10:
            self.session_decisions = self.session_decisions[-10:]
    
    def add_todo(self, todo: str):
        """Add a TODO to session memory"""
        if todo not in self.session_todos:
            self.session_todos.append(todo)
    
    def complete_todo(self, todo: str):
        """Mark a TODO as complete"""
        if todo in self.session_todos:
            self.session_todos.remove(todo)
    
    def set_session_context(self, context: str):
        """Set session context"""
        self.session_context = context
    
    def get_duration(self) -> float:
        """Get execution duration in seconds"""
        return time.time() - self.start_time
    
    def is_budget_exceeded(self) -> bool:
        """Check if any budget limits are exceeded"""
        if self.turns >= self.budgets.max_iterations:
            return True
        if self.cost >= self.budgets.max_cost:
            return True
        if self.get_duration() >= self.budgets.total_timeout:
            return True
        return False
    
    def get_recent_context(self, max_items: int = 5) -> List[ContextBlob]:
        """Get most recent context items"""
        return sorted(self.context, key=lambda x: x.timestamp, reverse=True)[:max_items]
    
    def get_high_importance_context(self, min_importance: float = 2.0) -> List[ContextBlob]:
        """Get high importance context items"""
        return [ctx for ctx in self.context if ctx.importance >= min_importance]
    
    def get_successful_tool_calls(self) -> List[ToolCallTrace]:
        """Get successful tool calls"""
        return [call for call in self.tool_calls if call.success]
    
    def get_failed_tool_calls(self) -> List[ToolCallTrace]:
        """Get failed tool calls"""
        return [call for call in self.tool_calls if not call.success and call.end_time is not None]


def make_state(goal: str, budgets: Optional[Budgets] = None, user_id: Optional[str] = None) -> State:
    """Factory function to create a new State object"""
    return State(
        goal=goal,
        budgets=budgets or Budgets(),
        user_id=user_id,
        session_id=f"{user_id or 'anon'}_{int(time.time())}"
    )


def enforce_budgets(state: State) -> bool:
    """Check if budgets allow continued execution"""
    if state.is_budget_exceeded():
        if state.turns >= state.budgets.max_iterations:
            state.set_status(AgentStatus.FAILED)
            state.add_context("Budget exceeded: max iterations", "budget_enforcer", 3.0)
        elif state.cost >= state.budgets.max_cost:
            state.set_status(AgentStatus.FAILED)
            state.add_context("Budget exceeded: max cost", "budget_enforcer", 3.0)
        elif state.get_duration() >= state.budgets.total_timeout:
            state.set_status(AgentStatus.FAILED)
            state.add_context("Budget exceeded: total timeout", "budget_enforcer", 3.0)
        return False
    return True
