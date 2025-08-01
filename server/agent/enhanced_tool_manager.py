"""
Enhanced Tool Management System for the Anton agent.

This component addresses the critical weakness of inefficient tool management by:
1. Categorizing tools by capability and function
2. Implementing tool selection strategies based on past performance
3. Supporting tool composition for complex tasks
4. Tracking tool performance and usage patterns
"""
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories for organizing tools by their primary function."""
    FILE_OPERATIONS = "file_operations"
    CODE_ANALYSIS = "code_analysis"
    WEB_INTERACTION = "web_interaction"
    DATA_PROCESSING = "data_processing"
    SYSTEM_OPERATIONS = "system_operations"
    COMMUNICATION = "communication"
    UTILITY = "utility"


class ToolCapability(Enum):
    """Specific capabilities that tools can provide."""
    READ = "read"
    WRITE = "write"
    SEARCH = "search"
    ANALYZE = "analyze"
    EXECUTE = "execute"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    CREATE = "create"
    DELETE = "delete"
    MONITOR = "monitor"


@dataclass
class ToolPerformanceMetrics:
    """Performance metrics for individual tools."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    last_used: float = 0.0
    error_patterns: List[str] = field(default_factory=list)
    success_contexts: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100
    
    @property 
    def reliability_score(self) -> float:
        """Calculate overall reliability score (0-1)."""
        if self.total_executions == 0:
            return 0.5  # Neutral score for untested tools
        
        success_factor = self.success_rate / 100
        speed_factor = max(0, 1 - (self.average_execution_time / 10))  # Penalize slow tools
        usage_factor = min(1, self.total_executions / 10)  # Reward well-tested tools
        
        return (success_factor * 0.6 + speed_factor * 0.2 + usage_factor * 0.2)
    
    def update_execution(self, execution_time: float, success: bool, context: str = "") -> None:
        """Update metrics after a tool execution."""
        self.total_executions += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.total_executions
        self.last_used = time.time()
        
        if success:
            self.successful_executions += 1
            if context and len(self.success_contexts) < 10:
                self.success_contexts.append(context)
        else:
            self.failed_executions += 1


@dataclass
class ToolMetadata:
    """Enhanced metadata for tools including categorization and capabilities."""
    name: str
    category: ToolCategory
    capabilities: Set[ToolCapability]
    description: str
    complexity: int = 1  # 1-5 scale, 1 = simple, 5 = complex
    dependencies: List[str] = field(default_factory=list)
    alternative_tools: List[str] = field(default_factory=list)
    performance: ToolPerformanceMetrics = field(default_factory=ToolPerformanceMetrics)
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def matches_capabilities(self, required_capabilities: Set[ToolCapability]) -> bool:
        """Check if tool matches required capabilities."""
        return required_capabilities.issubset(self.capabilities)
    
    def is_suitable_for_context(self, context: Dict[str, Any]) -> bool:
        """Check if tool is suitable for given context."""
        for req_key, req_value in self.context_requirements.items():
            if req_key not in context:
                return False
            if isinstance(req_value, list):
                if context[req_key] not in req_value:
                    return False
            elif context[req_key] != req_value:
                return False
        return True


@dataclass
class ToolComposition:
    """Represents a composition of tools for complex tasks."""
    name: str
    description: str
    tool_sequence: List[Tuple[str, Dict[str, Any]]]  # (tool_name, parameters)
    required_capabilities: Set[ToolCapability]
    success_criteria: Dict[str, Any]
    fallback_compositions: List[str] = field(default_factory=list)


class EnhancedToolManager:
    """
    Enhanced tool manager with categorization, performance tracking, and composition.
    
    Features:
    - Tool categorization by function and capability
    - Performance-based tool selection
    - Tool composition for complex tasks
    - Adaptive tool recommendations based on context
    - Comprehensive error handling and fallback strategies
    """
    
    def __init__(self):
        # Core tool registry
        self.tools: Dict[str, Any] = {}
        self.tool_metadata: Dict[str, ToolMetadata] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, ToolPerformanceMetrics] = {}
        
        # Tool compositions for complex tasks
        self.compositions: Dict[str, ToolComposition] = {}
        
        # Usage patterns and recommendations
        self.usage_patterns = defaultdict(list)
        self.recommendation_cache = {}
        
        # Tool availability and health monitoring
        self.tool_health_status: Dict[str, bool] = {}
        
        logger.info("Enhanced Tool Manager initialized")
    
    def register_tool(self, tool_instance: Any, metadata: Optional[ToolMetadata] = None) -> None:
        """Register a tool with enhanced metadata and categorization."""
        try:
            function_name = tool_instance.function["function"]["name"]
            self.tools[function_name] = tool_instance
            
            # Auto-generate metadata if not provided
            if metadata is None:
                metadata = self._auto_generate_metadata(tool_instance)
            
            self.tool_metadata[function_name] = metadata
            self.performance_metrics[function_name] = ToolPerformanceMetrics()
            self.tool_health_status[function_name] = True
            
            logger.info(f"Tool '{function_name}' registered with category {metadata.category.value}")
            
        except Exception as e:
            logger.error(f"Failed to register tool {tool_instance}: {e}")
            raise
    
    def _auto_generate_metadata(self, tool_instance: Any) -> ToolMetadata:
        """Auto-generate tool metadata from tool definition."""
        try:
            function_def = tool_instance.function["function"]
            name = function_def["name"]
            description = function_def.get("description", "")
            
            # Infer category from name and description
            category = self._infer_category(name, description)
            
            # Infer capabilities from name and description
            capabilities = self._infer_capabilities(name, description)
            
            return ToolMetadata(
                name=name,
                category=category,
                capabilities=capabilities,
                description=description
            )
            
        except Exception as e:
            logger.error(f"Failed to auto-generate metadata: {e}")
            # Return default metadata
            return ToolMetadata(
                name="unknown",
                category=ToolCategory.UTILITY,
                capabilities={ToolCapability.EXECUTE},
                description="Auto-generated metadata"
            )
    
    def _infer_category(self, name: str, description: str) -> ToolCategory:
        """Infer tool category from name and description."""
        name_lower = name.lower()
        desc_lower = description.lower()
        
        if any(keyword in name_lower for keyword in ["file", "read", "write", "directory"]):
            return ToolCategory.FILE_OPERATIONS
        elif any(keyword in name_lower for keyword in ["code", "analyze", "parse", "lint"]):
            return ToolCategory.CODE_ANALYSIS
        elif any(keyword in name_lower for keyword in ["web", "http", "url", "browser"]):
            return ToolCategory.WEB_INTERACTION
        elif any(keyword in name_lower for keyword in ["data", "json", "csv", "process"]):
            return ToolCategory.DATA_PROCESSING
        elif any(keyword in name_lower for keyword in ["system", "shell", "command", "execute"]):
            return ToolCategory.SYSTEM_OPERATIONS
        elif any(keyword in name_lower for keyword in ["send", "email", "message", "notify"]):
            return ToolCategory.COMMUNICATION
        else:
            return ToolCategory.UTILITY
    
    def _infer_capabilities(self, name: str, description: str) -> Set[ToolCapability]:
        """Infer tool capabilities from name and description."""
        capabilities = set()
        text = f"{name} {description}".lower()
        
        capability_keywords = {
            ToolCapability.READ: ["read", "get", "fetch", "retrieve", "view"],
            ToolCapability.WRITE: ["write", "save", "create", "update", "modify"],
            ToolCapability.SEARCH: ["search", "find", "query", "lookup", "filter"],
            ToolCapability.ANALYZE: ["analyze", "inspect", "examine", "check", "validate"],
            ToolCapability.EXECUTE: ["execute", "run", "perform", "process", "handle"],
            ToolCapability.TRANSFORM: ["transform", "convert", "format", "parse", "encode"],
            ToolCapability.CREATE: ["create", "make", "generate", "build", "new"],
            ToolCapability.DELETE: ["delete", "remove", "clear", "clean", "purge"],
        }
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in text for keyword in keywords):
                capabilities.add(capability)
        
        # Ensure at least one capability
        if not capabilities:
            capabilities.add(ToolCapability.EXECUTE)
        
        return capabilities
    
    def get_tools_by_category(self, category: ToolCategory) -> Dict[str, Any]:
        """Get all tools in a specific category."""
        return {
            name: tool for name, tool in self.tools.items()
            if self.tool_metadata[name].category == category
        }
    
    def get_tools_by_capabilities(self, capabilities: Set[ToolCapability]) -> Dict[str, Any]:
        """Get tools that match the required capabilities."""
        matching_tools = {}
        for name, tool in self.tools.items():
            if self.tool_metadata[name].matches_capabilities(capabilities):
                matching_tools[name] = tool
        return matching_tools
    
    def recommend_tools(
        self,
        task_description: str,
        context: Dict[str, Any] = None,
        max_recommendations: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Recommend tools based on task description and context.
        
        Returns:
            List of (tool_name, confidence_score) tuples, sorted by confidence.
        """
        if context is None:
            context = {}
        
        # Check cache first
        cache_key = f"{task_description}_{hash(str(sorted(context.items())))}"
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]
        
        recommendations = []
        task_lower = task_description.lower()
        
        for tool_name, tool in self.tools.items():
            metadata = self.tool_metadata[tool_name]
            performance = self.performance_metrics[tool_name]
            
            # Calculate confidence score
            confidence = self._calculate_tool_confidence(
                tool_name, metadata, performance, task_lower, context
            )
            
            if confidence > 0.1:  # Only include tools with reasonable confidence
                recommendations.append((tool_name, confidence))
        
        # Sort by confidence and limit results
        recommendations.sort(key=lambda x: x[1], reverse=True)
        recommendations = recommendations[:max_recommendations]
        
        # Cache the result
        self.recommendation_cache[cache_key] = recommendations
        
        logger.info(f"Recommended {len(recommendations)} tools for task: {task_description[:50]}...")
        return recommendations
    
    def _calculate_tool_confidence(
        self,
        tool_name: str,
        metadata: ToolMetadata,
        performance: ToolPerformanceMetrics,
        task_description: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for tool recommendation."""
        confidence = 0.0
        
        # Base confidence from name and description matching
        if tool_name.lower() in task_description:
            confidence += 0.4
        
        if any(word in task_description for word in metadata.description.lower().split()):
            confidence += 0.2
        
        # Performance-based confidence
        reliability = performance.reliability_score
        confidence += reliability * 0.3
        
        # Context suitability
        if metadata.is_suitable_for_context(context):
            confidence += 0.2
        
        # Penalize complex tools for simple tasks
        if "simple" in task_description or "quick" in task_description:
            confidence -= (metadata.complexity - 1) * 0.1
        
        # Health status
        if not self.tool_health_status.get(tool_name, True):
            confidence *= 0.5  # Reduce confidence for unhealthy tools
        
        return max(0.0, min(1.0, confidence))
    
    def execute_tool_with_tracking(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: str = ""
    ) -> Tuple[str, bool, float]:
        """
        Execute a tool with comprehensive performance tracking.
        
        Returns:
            Tuple of (result, success, execution_time)
        """
        if tool_name not in self.tools:
            logger.error(f"Tool '{tool_name}' not found")
            return f"Error: Tool '{tool_name}' not found", False, 0.0
        
        start_time = time.time()
        success = False
        result = ""
        
        try:
            logger.info(f"Executing tool '{tool_name}' with tracking")
            tool_instance = self.tools[tool_name]
            result = tool_instance.run(tool_args)
            success = True
            
        except Exception as e:
            logger.error(f"Tool execution failed for '{tool_name}': {e}", exc_info=True)
            result = f"Error: {str(e)}"
            success = False
            
            # Update error patterns
            if tool_name in self.performance_metrics:
                error_msg = str(e)[:100]  # Truncate long error messages
                if error_msg not in self.performance_metrics[tool_name].error_patterns:
                    self.performance_metrics[tool_name].error_patterns.append(error_msg)
        
        finally:
            execution_time = time.time() - start_time
            
            # Update performance metrics
            if tool_name in self.performance_metrics:
                self.performance_metrics[tool_name].update_execution(
                    execution_time, success, context
                )
            
            # Update tool health status
            self._update_tool_health(tool_name, success)
            
            # Record usage pattern
            self.usage_patterns[tool_name].append({
                "timestamp": time.time(),
                "success": success,
                "execution_time": execution_time,
                "context": context[:50]  # Truncate context
            })
        
        logger.info(f"Tool '{tool_name}' executed in {execution_time:.2f}s, success: {success}")
        return result, success, execution_time
    
    def _update_tool_health(self, tool_name: str, success: bool) -> None:
        """Update tool health status based on recent executions."""
        if tool_name not in self.tool_health_status:
            self.tool_health_status[tool_name] = True
        
        # Consider tool unhealthy if last 3 executions failed
        recent_patterns = self.usage_patterns[tool_name][-3:]
        if len(recent_patterns) >= 3:
            recent_failures = sum(1 for pattern in recent_patterns if not pattern["success"])
            self.tool_health_status[tool_name] = recent_failures < 3
    
    def create_tool_composition(
        self,
        name: str,
        description: str,
        tool_sequence: List[Tuple[str, Dict[str, Any]]],
        success_criteria: Dict[str, Any] = None
    ) -> bool:
        """Create a new tool composition for complex tasks."""
        try:
            # Validate that all tools in sequence exist
            for tool_name, _ in tool_sequence:
                if tool_name not in self.tools:
                    logger.error(f"Tool '{tool_name}' not found for composition '{name}'")
                    return False
            
            # Infer required capabilities from tools in sequence
            required_capabilities = set()
            for tool_name, _ in tool_sequence:
                required_capabilities.update(self.tool_metadata[tool_name].capabilities)
            
            composition = ToolComposition(
                name=name,
                description=description,
                tool_sequence=tool_sequence,
                required_capabilities=required_capabilities,
                success_criteria=success_criteria or {}
            )
            
            self.compositions[name] = composition
            logger.info(f"Tool composition '{name}' created with {len(tool_sequence)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create tool composition '{name}': {e}")
            return False
    
    def execute_composition(
        self,
        composition_name: str,
        initial_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a tool composition."""
        if composition_name not in self.compositions:
            return {"error": f"Composition '{composition_name}' not found"}
        
        composition = self.compositions[composition_name]
        context = initial_context or {}
        results = []
        
        logger.info(f"Executing composition '{composition_name}' with {len(composition.tool_sequence)} tools")
        
        for i, (tool_name, tool_args) in enumerate(composition.tool_sequence):
            try:
                # Execute tool with tracking
                result, success, exec_time = self.execute_tool_with_tracking(
                    tool_name, tool_args, f"composition_{composition_name}_step_{i}"
                )
                
                step_result = {
                    "step": i,
                    "tool": tool_name,
                    "result": result,
                    "success": success,
                    "execution_time": exec_time
                }
                results.append(step_result)
                
                if not success:
                    logger.warning(f"Composition '{composition_name}' failed at step {i}")
                    return {
                        "composition": composition_name,
                        "success": False,
                        "failed_at_step": i,
                        "results": results
                    }
                
                # Update context with result for next step
                context[f"step_{i}_result"] = result
                
            except Exception as e:
                logger.error(f"Composition execution failed at step {i}: {e}")
                return {
                    "composition": composition_name,
                    "success": False,
                    "error": str(e),
                    "failed_at_step": i,
                    "results": results
                }
        
        logger.info(f"Composition '{composition_name}' completed successfully")
        return {
            "composition": composition_name,
            "success": True,
            "results": results,
            "final_context": context
        }
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get enhanced tool schemas with metadata."""
        schemas = []
        for tool_name, tool in self.tools.items():
            schema = tool.function.copy()
            
            # Add metadata to schema
            if tool_name in self.tool_metadata:
                metadata = self.tool_metadata[tool_name]
                schema["metadata"] = {
                    "category": metadata.category.value,
                    "capabilities": [cap.value for cap in metadata.capabilities],
                    "complexity": metadata.complexity,
                    "reliability": self.performance_metrics[tool_name].reliability_score
                }
            
            schemas.append(schema)
        
        return schemas
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "total_tools": len(self.tools),
            "total_executions": sum(m.total_executions for m in self.performance_metrics.values()),
            "overall_success_rate": 0.0,
            "tool_performance": {},
            "category_performance": defaultdict(list),
            "top_performers": [],
            "problematic_tools": []
        }
        
        # Calculate overall success rate
        total_executions = sum(m.total_executions for m in self.performance_metrics.values())
        total_successes = sum(m.successful_executions for m in self.performance_metrics.values())
        if total_executions > 0:
            report["overall_success_rate"] = (total_successes / total_executions) * 100
        
        # Individual tool performance
        for tool_name, metrics in self.performance_metrics.items():
            if metrics.total_executions > 0:
                tool_report = {
                    "executions": metrics.total_executions,
                    "success_rate": metrics.success_rate,
                    "avg_execution_time": metrics.average_execution_time,
                    "reliability_score": metrics.reliability_score,
                    "category": self.tool_metadata[tool_name].category.value
                }
                report["tool_performance"][tool_name] = tool_report
                
                # Group by category
                report["category_performance"][tool_report["category"]].append(
                    (tool_name, metrics.reliability_score)
                )
                
                # Identify top performers and problematic tools
                if metrics.reliability_score > 0.8 and metrics.total_executions > 5:
                    report["top_performers"].append((tool_name, metrics.reliability_score))
                elif metrics.reliability_score < 0.3 and metrics.total_executions > 3:
                    report["problematic_tools"].append((tool_name, metrics.reliability_score))
        
        # Sort lists
        report["top_performers"].sort(key=lambda x: x[1], reverse=True)
        report["problematic_tools"].sort(key=lambda x: x[1])
        
        return report
    
    def optimize_tool_selection(self) -> None:
        """Optimize tool selection based on performance history."""
        logger.info("Optimizing tool selection based on performance data...")
        
        # Clear recommendation cache to force recalculation
        self.recommendation_cache.clear()
        
        # Update alternative tools based on performance
        for tool_name, metadata in self.tool_metadata.items():
            performance = self.performance_metrics[tool_name]
            
            if performance.reliability_score < 0.5:
                # Find better alternatives in the same category
                category_tools = self.get_tools_by_category(metadata.category)
                alternatives = []
                
                for alt_name, alt_tool in category_tools.items():
                    if alt_name != tool_name:
                        alt_performance = self.performance_metrics[alt_name]
                        if alt_performance.reliability_score > performance.reliability_score:
                            alternatives.append(alt_name)
                
                metadata.alternative_tools = alternatives[:3]  # Keep top 3 alternatives
        
        logger.info("Tool selection optimization completed")


# Create enhanced tool manager instance
enhanced_tool_manager = EnhancedToolManager()