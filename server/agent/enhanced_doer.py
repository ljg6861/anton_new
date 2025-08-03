"""
Enhanced Doer Component

Enforces structured tool usage, prevents conversational responses,
and provides standardized output with comprehensive validation.
"""

import re
import json
import time
import asyncio

# Make httpx optional for testing environments
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

from typing import AsyncGenerator, Any, List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from server.agent import config
from server.agent.config import ASSISTANT_ROLE, USER_ROLE
from server.agent.tool_executor import execute_tool
from server.agent.tools.tool_manager import tool_manager


class ResponseType(Enum):
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer" 
    INVALID = "invalid"


class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_FORMAT = "invalid_format"
    TOOL_NOT_FOUND = "tool_not_found"


@dataclass
class ToolCallResult:
    """Standardized tool call result."""
    tool_name: str
    arguments: Dict
    result: str
    status: ExecutionStatus
    duration: float
    error_message: Optional[str] = None


@dataclass
class DoerResponse:
    """Standardized Doer response format."""
    response_type: ResponseType
    content: str
    tool_calls: List[ToolCallResult]
    execution_status: ExecutionStatus
    duration: float
    raw_response: str
    validation_errors: List[str]


class EnhancedDoer:
    """
    Enhanced Doer component with strict structure enforcement.
    
    Features:
    - Mandatory structured tool usage
    - Conversational response prevention
    - Comprehensive validation
    - Timeout enforcement
    - Tool existence validation
    - Execution history tracking
    """
    
    def __init__(self, 
                 max_response_time: float = 15.0,
                 require_structured_output: bool = True,
                 validate_tools_before_execution: bool = True):
        self.max_response_time = max_response_time
        self.require_structured_output = require_structured_output
        self.validate_tools_before_execution = validate_tools_before_execution
        
        # Execution tracking
        self.execution_history: List[DoerResponse] = []
        self.tool_usage_stats: Dict[str, int] = {}
        
        # Response patterns
        self.tool_call_pattern = re.compile(r'<tool_code>\s*(\{.*?\})\s*</tool_code>', re.DOTALL)
        self.final_answer_pattern = re.compile(r'FINAL ANSWER:\s*(.*)', re.DOTALL | re.IGNORECASE)
        
        # Invalid patterns (conversational responses to reject)
        self.invalid_patterns = [
            re.compile(r'^(hello|hi|thank you|thanks|please|sorry)', re.IGNORECASE),
            re.compile(r'(let me|i will|i can|i should|i need to)', re.IGNORECASE),
            re.compile(r'(how can i|what would you like|is there anything)', re.IGNORECASE)
        ]
    
    async def execute_structured_turn(self,
                                    api_base_url: str,
                                    messages: List[Dict],
                                    logger: Any,
                                    tools: List,
                                    context_store: Dict = None) -> DoerResponse:
        """
        Execute a structured Doer turn with comprehensive validation.
        
        Returns a standardized DoerResponse with detailed execution information.
        """
        start_time = time.time()
        
        try:
            # Generate response with timeout
            raw_response = await asyncio.wait_for(
                self._generate_response(api_base_url, messages, logger, tools),
                timeout=self.max_response_time
            )
            
            # Parse and validate response
            doer_response = self._parse_and_validate_response(raw_response, logger)
            
            # Execute tool calls if present
            if doer_response.response_type == ResponseType.TOOL_CALL:
                await self._execute_tool_calls(doer_response, logger, context_store)
            
            # Update execution history
            doer_response.duration = time.time() - start_time
            self.execution_history.append(doer_response)
            
            # Update tool usage statistics
            for tool_call in doer_response.tool_calls:
                self.tool_usage_stats[tool_call.tool_name] = self.tool_usage_stats.get(tool_call.tool_name, 0) + 1
            
            return doer_response
            
        except asyncio.TimeoutError:
            logger.error(f"Doer response timeout after {self.max_response_time}s")
            return DoerResponse(
                response_type=ResponseType.INVALID,
                content="",
                tool_calls=[],
                execution_status=ExecutionStatus.TIMEOUT,
                duration=time.time() - start_time,
                raw_response="",
                validation_errors=[f"Response timeout after {self.max_response_time}s"]
            )
        
        except Exception as e:
            logger.error(f"Error in structured Doer turn: {e}", exc_info=True)
            return DoerResponse(
                response_type=ResponseType.INVALID,
                content="",
                tool_calls=[],
                execution_status=ExecutionStatus.FAILED,
                duration=time.time() - start_time,
                raw_response="",
                validation_errors=[f"Execution error: {str(e)}"]
            )
    
    async def _generate_response(self,
                               api_base_url: str, 
                               messages: List[Dict],
                               logger: Any,
                               tools: List) -> str:
        """Generate raw response from the model."""
        # If httpx is not available, return a mock response for testing
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, returning mock response for testing")
            return "FINAL ANSWER: Mock response for testing - httpx not available"
        
        from server.agent.doer import execute_turn
        
        response_buffer = ""
        async for token in execute_turn(api_base_url, messages, logger, tools, 0.6, True):
            response_buffer += token
        
        return response_buffer
    
    def _parse_and_validate_response(self, raw_response: str, logger: Any) -> DoerResponse:
        """Parse and validate the raw response from the model."""
        validation_errors = []
        
        # Remove thinking tags for content analysis
        content = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
        
        # Check for invalid conversational patterns
        if self.require_structured_output:
            for pattern in self.invalid_patterns:
                if pattern.search(content):
                    validation_errors.append(f"Conversational response detected: {pattern.pattern}")
        
        # Determine response type
        response_type = self._determine_response_type(content)
        
        # Validate required structure
        if response_type == ResponseType.INVALID:
            validation_errors.append("Response must contain either tool call or FINAL ANSWER")
        
        # Extract tool calls
        tool_calls = self._extract_tool_calls(content, logger)
        
        # Determine execution status
        execution_status = ExecutionStatus.SUCCESS
        if validation_errors:
            execution_status = ExecutionStatus.INVALID_FORMAT
        elif response_type == ResponseType.INVALID:
            execution_status = ExecutionStatus.INVALID_FORMAT
        
        return DoerResponse(
            response_type=response_type,
            content=content,
            tool_calls=tool_calls,
            execution_status=execution_status,
            duration=0.0,  # Will be set by caller
            raw_response=raw_response,
            validation_errors=validation_errors
        )
    
    def _determine_response_type(self, content: str) -> ResponseType:
        """Determine the type of response based on content."""
        # Check for final answer
        if self.final_answer_pattern.search(content):
            return ResponseType.FINAL_ANSWER
        
        # Check for tool calls
        if self.tool_call_pattern.search(content):
            return ResponseType.TOOL_CALL
        
        return ResponseType.INVALID
    
    def _extract_tool_calls(self, content: str, logger: Any) -> List[ToolCallResult]:
        """Extract and validate tool calls from content."""
        tool_calls = []
        
        for match in self.tool_call_pattern.finditer(content):
            tool_call_content = match.group(1).strip()
            
            try:
                tool_data = json.loads(tool_call_content)
                tool_name = tool_data.get("name")
                
                if not tool_name:
                    logger.error(f"Tool call missing 'name' field: {tool_call_content}")
                    continue
                
                # Validate tool existence if enabled
                if self.validate_tools_before_execution:
                    if not self._validate_tool_exists(tool_name):
                        tool_calls.append(ToolCallResult(
                            tool_name=tool_name,
                            arguments=tool_data.get("arguments", {}),
                            result="",
                            status=ExecutionStatus.TOOL_NOT_FOUND,
                            duration=0.0,
                            error_message=f"Tool '{tool_name}' not found in registry"
                        ))
                        continue
                
                # Create placeholder result (will be executed later)
                tool_calls.append(ToolCallResult(
                    tool_name=tool_name,
                    arguments=tool_data.get("arguments", {}),
                    result="",
                    status=ExecutionStatus.SUCCESS,
                    duration=0.0
                ))
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in tool call: {e}. Content: {tool_call_content}")
                continue
        
        return tool_calls
    
    def _validate_tool_exists(self, tool_name: str) -> bool:
        """Validate that a tool exists in the registry."""
        try:
            # Check if tool exists in tool manager
            return tool_manager.get_tool(tool_name) is not None
        except Exception:
            return False
    
    async def _execute_tool_calls(self, 
                                doer_response: DoerResponse,
                                logger: Any,
                                context_store: Dict = None):
        """Execute all tool calls in the response."""
        for i, tool_call in enumerate(doer_response.tool_calls):
            if tool_call.status != ExecutionStatus.SUCCESS:
                continue  # Skip invalid tool calls
            
            start_time = time.time()
            
            try:
                # Execute the tool
                result = execute_tool(tool_call.tool_name, tool_call.arguments, logger)
                
                # Update tool call result
                tool_call.result = result
                tool_call.duration = time.time() - start_time
                tool_call.status = ExecutionStatus.SUCCESS
                
                # Update context store
                if context_store is not None:
                    self._update_context_store(context_store, tool_call, logger)
                
                logger.info(f"Tool '{tool_call.tool_name}' executed successfully in {tool_call.duration:.2f}s")
                
            except Exception as e:
                tool_call.status = ExecutionStatus.FAILED
                tool_call.error_message = str(e)
                tool_call.duration = time.time() - start_time
                
                logger.error(f"Tool '{tool_call.tool_name}' execution failed: {e}")
                
                # Update overall response status if any tool fails
                doer_response.execution_status = ExecutionStatus.FAILED
    
    def _update_context_store(self, context_store: Dict, tool_call: ToolCallResult, logger: Any):
        """Update context store with tool execution information."""
        # Track all tool outputs for learning analysis
        if "tool_outputs" not in context_store:
            context_store["tool_outputs"] = []
        
        # Store tool output with metadata (limit to last 20 outputs)
        tool_output_entry = (
            f"Tool: {tool_call.tool_name} | "
            f"Args: {tool_call.arguments} | "
            f"Result: {tool_call.result[:500]}{'...' if len(tool_call.result) > 500 else ''} | "
            f"Status: {tool_call.status.value} | "
            f"Duration: {tool_call.duration:.2f}s"
        )
        context_store["tool_outputs"].append(tool_output_entry)
        
        # Keep only recent outputs
        if len(context_store["tool_outputs"]) > 20:
            context_store["tool_outputs"] = context_store["tool_outputs"][-20:]
        
        # Specific tool handling
        if tool_call.tool_name == "read_file":
            file_path = tool_call.arguments.get("file_path")
            if file_path and tool_call.status == ExecutionStatus.SUCCESS:
                context_store.setdefault("explored_files", set()).add(file_path)
                
                # Store content with size management
                content = tool_call.result
                if len(content) > 10000:
                    content = content[:10000] + "... [truncated]"
                context_store.setdefault("code_content", {})[file_path] = content
        
        elif tool_call.tool_name == "list_directory":
            path = tool_call.arguments.get("path", ".")
            if tool_call.status == ExecutionStatus.SUCCESS:
                context_store.setdefault("explored_files", set()).add(path)
                context_store.setdefault("task_progress", []).append(f"Listed directory {path}")
    
    def get_execution_statistics(self) -> Dict:
        """Get comprehensive execution statistics."""
        if not self.execution_history:
            return {}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for r in self.execution_history if r.execution_status == ExecutionStatus.SUCCESS)
        
        avg_duration = sum(r.duration for r in self.execution_history) / total_executions
        total_tool_calls = sum(len(r.tool_calls) for r in self.execution_history)
        
        # Response type distribution
        response_types = {}
        for response_type in ResponseType:
            count = sum(1 for r in self.execution_history if r.response_type == response_type)
            response_types[response_type.value] = count
        
        # Execution status distribution
        execution_statuses = {}
        for status in ExecutionStatus:
            count = sum(1 for r in self.execution_history if r.execution_status == status)
            execution_statuses[status.value] = count
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'average_duration': avg_duration,
            'total_tool_calls': total_tool_calls,
            'tool_usage_stats': self.tool_usage_stats.copy(),
            'response_type_distribution': response_types,
            'execution_status_distribution': execution_statuses
        }
    
    def reset_statistics(self):
        """Reset all execution statistics."""
        self.execution_history.clear()
        self.tool_usage_stats.clear()