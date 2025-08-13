"""
ToolsRouter: Centralized tool execution with allowlists, timeouts, and retries.
Provides a safe and observable layer over the underlying tool system.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of tool execution"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    RETRY_EXHAUSTED = "retry_exhausted"


@dataclass
class ExecutionResult:
    """Result of tool execution with metadata"""
    status: ExecutionStatus
    result: Any = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    attempts: int = 1
    tool_name: str = ""
    arguments: Dict[str, Any] = None
    
    @property
    def ok(self) -> bool:
        """Check if execution was successful"""
        return self.status == ExecutionStatus.SUCCESS
    
    def __post_init__(self):
        if self.arguments is None:
            self.arguments = {}


class ToolsRouter:
    """
    Centralized tool execution router with safety features.
    Implements allowlist, timeouts, retries, and observability.
    """
    
    def __init__(self, 
                 allowlist: Optional[Set[str]] = None,
                 max_retries: int = 2,
                 default_timeout_ms: float = 30000,
                 auto_discover_tools: bool = True):
        """
        Initialize the ToolsRouter.
        
        Args:
            allowlist: Set of allowed tool names. If None and auto_discover_tools=True, 
                      uses all tools from the tools/ directory.
            max_retries: Maximum number of retry attempts per tool call
            default_timeout_ms: Default timeout in milliseconds
            auto_discover_tools: If True, automatically discover tools from tool_manager
        """
        self.max_retries = max_retries
        self.default_timeout_ms = default_timeout_ms
        self.auto_discover_tools = auto_discover_tools
        
        # Initialize allowlist
        if allowlist is not None:
            self.allowlist = allowlist.copy()
        elif auto_discover_tools:
            self.allowlist = self._discover_available_tools()
        else:
            # Fallback to basic safe tools if no discovery
            self.allowlist = {
                "search", "http", "bash", "python", "browser_click",
                "read_file", "write_file", "list_directory", "create_file",
                "edit_file", "delete_file", "run_shell_command", 
                "run_git_command", "search_web", "search_code",
                "get_code_stats", "expand_content", "rebuild_index"
            }
        
        # Statistics tracking
        self.execution_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "blocked_calls": 0,
            "timeout_calls": 0,
            "retry_calls": 0
        }
        
        logger.info(f"ToolsRouter initialized with {len(self.allowlist)} tools from dynamic discovery")
        logger.debug(f"Allowed tools: {sorted(self.allowlist)}")
    
    def _discover_available_tools(self) -> Set[str]:
        """
        Dynamically discover available tools from the tool_manager.
        
        Returns:
            Set of available tool names
        """
        try:
            from server.agent.tools.tool_manager import tool_manager
            
            # Get all registered tool names
            available_tools = set(tool_manager.get_tool_names())
            
            logger.info(f"Discovered {len(available_tools)} tools from tools/ directory")
            logger.debug(f"Available tools: {sorted(available_tools)}")
            
            return available_tools
            
        except ImportError as e:
            logger.error(f"Failed to import tool_manager for tool discovery: {e}")
            # Fallback to empty set - will need manual allowlist
            return set()
        except Exception as e:
            logger.error(f"Error during tool discovery: {e}")
            # Fallback to empty set - will need manual allowlist  
            return set()
    
    def refresh_allowlist(self) -> None:
        """
        Refresh the allowlist by re-discovering available tools.
        Useful when tools are added/removed at runtime.
        """
        if self.auto_discover_tools:
            old_count = len(self.allowlist)
            self.allowlist = self._discover_available_tools()
            new_count = len(self.allowlist)
            logger.info(f"Refreshed allowlist: {old_count} -> {new_count} tools")
        else:
            logger.warning("Auto-discovery disabled, cannot refresh allowlist automatically")
    
    async def call(self, name: str, args: Dict[str, Any], 
                   timeout_ms: Optional[float] = None) -> ExecutionResult:
        """
        Execute a tool call with safety features.
        
        Args:
            name: Tool name to execute
            args: Arguments to pass to the tool
            timeout_ms: Timeout in milliseconds (uses default if None)
            
        Returns:
            ExecutionResult with status and result/error information
        """
        self.execution_stats["total_calls"] += 1
        start_time = time.time()
        
        # Check allowlist
        if name not in self.allowlist:
            self.execution_stats["blocked_calls"] += 1
            logger.warning(f"Tool '{name}' blocked by allowlist")
            return ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                error_message=f"blocked:{name}",
                tool_name=name,
                arguments=args,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        timeout_seconds = (timeout_ms or self.default_timeout_ms) / 1000.0
        attempts = 0
        last_result = None
        
        while attempts < self.max_retries:
            attempts += 1
            
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._low_level_dispatch(name, args),
                    timeout=timeout_seconds
                )
                
                execution_time = (time.time() - start_time) * 1000
                
                if isinstance(result, Exception):
                    # Tool returned an exception
                    last_result = ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        error_message=str(result),
                        tool_name=name,
                        arguments=args,
                        attempts=attempts,
                        execution_time_ms=execution_time
                    )
                    
                    if attempts < self.max_retries:
                        self.execution_stats["retry_calls"] += 1
                        logger.warning(f"Tool '{name}' failed (attempt {attempts}), retrying: {result}")
                        await asyncio.sleep(0.1 * attempts)  # Progressive backoff
                        continue
                else:
                    # Success
                    self.execution_stats["successful_calls"] += 1
                    return ExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        result=result,
                        tool_name=name,
                        arguments=args,
                        attempts=attempts,
                        execution_time_ms=execution_time
                    )
                    
            except asyncio.TimeoutError:
                execution_time = (time.time() - start_time) * 1000
                self.execution_stats["timeout_calls"] += 1
                logger.error(f"Tool '{name}' timed out after {timeout_seconds}s")
                
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    error_message=f"Tool execution timed out after {timeout_seconds}s",
                    tool_name=name,
                    arguments=args,
                    attempts=attempts,
                    execution_time_ms=execution_time
                )
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(f"Unexpected error executing tool '{name}': {e}", exc_info=True)
                
                last_result = ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error_message=f"Unexpected error: {str(e)}",
                    tool_name=name,
                    arguments=args,
                    attempts=attempts,
                    execution_time_ms=execution_time
                )
                
                if attempts < self.max_retries:
                    self.execution_stats["retry_calls"] += 1
                    logger.warning(f"Tool '{name}' failed with exception (attempt {attempts}), retrying")
                    await asyncio.sleep(0.1 * attempts)  # Progressive backoff
                    continue
        
        # All retries exhausted
        self.execution_stats["failed_calls"] += 1
        if last_result:
            last_result.status = ExecutionStatus.RETRY_EXHAUSTED
            return last_result
        else:
            return ExecutionResult(
                status=ExecutionStatus.RETRY_EXHAUSTED,
                error_message="All retry attempts exhausted",
                tool_name=name,
                arguments=args,
                attempts=attempts,
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _low_level_dispatch(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Low-level tool dispatch to the existing tool system.
        This is where we integrate with the current tool_manager.
        """
        from server.agent.tools.tool_manager import tool_manager
        
        # Use asyncio.to_thread to run the potentially blocking tool execution
        # in a thread pool to avoid blocking the event loop
        try:
            result = await asyncio.to_thread(tool_manager.run_tool, name, args)
            return result
        except Exception as e:
            return e
    
    def add_to_allowlist(self, tool_names: List[str]) -> None:
        """Add tools to the allowlist"""
        self.allowlist.update(tool_names)
        logger.info(f"Added to allowlist: {tool_names}")
    
    def remove_from_allowlist(self, tool_names: List[str]) -> None:
        """Remove tools from the allowlist"""
        for tool_name in tool_names:
            self.allowlist.discard(tool_name)
        logger.info(f"Removed from allowlist: {tool_names}")
    
    def is_allowed(self, tool_name: str) -> bool:
        """Check if a tool is in the allowlist"""
        return tool_name in self.allowlist
    
    def get_available_tools(self) -> Set[str]:
        """Get all currently available tools from tool_manager"""
        return self._discover_available_tools()
    
    def get_allowlist(self) -> Set[str]:
        """Get the current allowlist"""
        return self.allowlist.copy()
    
    def sync_with_available_tools(self) -> Dict[str, List[str]]:
        """
        Synchronize allowlist with currently available tools.
        
        Returns:
            Dictionary with 'added' and 'removed' tool lists
        """
        if not self.auto_discover_tools:
            logger.warning("Auto-discovery disabled, cannot sync with available tools")
            return {"added": [], "removed": []}
        
        current_allowlist = self.allowlist.copy()
        available_tools = self._discover_available_tools()
        
        # Find tools that should be added (available but not in allowlist)
        added_tools = list(available_tools - current_allowlist)
        
        # Find tools that should be removed (in allowlist but not available)
        removed_tools = list(current_allowlist - available_tools)
        
        # Update allowlist
        self.allowlist = available_tools
        
        if added_tools:
            logger.info(f"Added newly discovered tools: {added_tools}")
        if removed_tools:
            logger.info(f"Removed unavailable tools: {removed_tools}")
        
        return {"added": added_tools, "removed": removed_tools}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total = self.execution_stats["total_calls"]
        if total == 0:
            return self.execution_stats.copy()
        
        stats = self.execution_stats.copy()
        stats["success_rate"] = self.execution_stats["successful_calls"] / total
        stats["failure_rate"] = self.execution_stats["failed_calls"] / total
        stats["block_rate"] = self.execution_stats["blocked_calls"] / total
        stats["timeout_rate"] = self.execution_stats["timeout_calls"] / total
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset execution statistics"""
        for key in self.execution_stats:
            self.execution_stats[key] = 0
        logger.info("Tool execution statistics reset")


# Global instance for the application with dynamic tool discovery
tools_router = ToolsRouter(auto_discover_tools=True)
