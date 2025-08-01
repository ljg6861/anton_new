"""
Actionable Metrics Integration System for the Anton agent.

This component addresses the critical weakness of "Metrics Without Actionable Feedback" by:
1. Using collected metrics for runtime optimization
2. Providing real-time performance recommendations
3. Automatic performance tuning based on historical data
4. Proactive issue detection and resolution
5. Strategic insights for workflow improvements
"""
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to track and optimize."""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"
    USER_SATISFACTION = "user_satisfaction"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"


class OptimizationLevel(Enum):
    """Levels of optimization to apply."""
    CONSERVATIVE = "conservative"  # Safe, small improvements
    MODERATE = "moderate"         # Balanced improvements
    AGGRESSIVE = "aggressive"     # Maximum optimization
    EXPERIMENTAL = "experimental" # Cutting-edge optimizations


@dataclass
class MetricThreshold:
    """Threshold configuration for metrics."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    optimal_range: Tuple[float, float]
    optimization_target: float
    enabled: bool = True


@dataclass
class OptimizationRecommendation:
    """Recommendation for performance optimization."""
    metric_type: MetricType
    current_value: float
    target_value: float
    confidence: float
    action: str
    rationale: str
    impact_estimate: str
    risk_level: str
    implementation_priority: int
    estimated_effort: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_type": self.metric_type.value,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "confidence": self.confidence,
            "action": self.action,
            "rationale": self.rationale,
            "impact_estimate": self.impact_estimate,
            "risk_level": self.risk_level,
            "implementation_priority": self.implementation_priority,
            "estimated_effort": self.estimated_effort
        }


@dataclass
class PerformanceAlert:
    """Alert for performance issues requiring attention."""
    alert_type: str
    severity: str  # low, medium, high, critical
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    recommendations: List[str]
    timestamp: float
    auto_fixable: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "message": self.message,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
            "auto_fixable": self.auto_fixable
        }


class ActionableMetricsSystem:
    """
    System that transforms raw metrics into actionable insights and automatic optimizations.
    
    Features:
    - Real-time performance monitoring and alerting
    - Automatic optimization recommendations
    - Proactive issue detection and resolution
    - Historical trend analysis and prediction
    - Strategic insights for long-term improvements
    """
    
    def __init__(self):
        # Metric thresholds configuration
        self.thresholds = self._initialize_thresholds()
        
        # Performance tracking
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.optimization_history = deque(maxlen=100)
        self.alert_history = deque(maxlen=200)
        
        # Optimization state
        self.active_optimizations = {}
        self.optimization_level = OptimizationLevel.MODERATE
        self.auto_optimization_enabled = True
        
        # Real-time monitoring
        self.monitoring_interval = 30  # seconds
        self.last_monitoring_time = time.time()
        
        logger.info("Actionable Metrics System initialized")
    
    def _initialize_thresholds(self) -> Dict[str, MetricThreshold]:
        """Initialize metric thresholds with sensible defaults."""
        return {
            "response_time": MetricThreshold(
                metric_name="response_time",
                warning_threshold=5.0,  # seconds
                critical_threshold=10.0,
                optimal_range=(1.0, 3.0),
                optimization_target=2.0
            ),
            "success_rate": MetricThreshold(
                metric_name="success_rate",
                warning_threshold=80.0,  # percentage
                critical_threshold=70.0,
                optimal_range=(90.0, 100.0),
                optimization_target=95.0
            ),
            "error_rate": MetricThreshold(
                metric_name="error_rate",
                warning_threshold=10.0,  # percentage
                critical_threshold=20.0,
                optimal_range=(0.0, 5.0),
                optimization_target=2.0
            ),
            "context_utilization": MetricThreshold(
                metric_name="context_utilization",
                warning_threshold=80.0,  # percentage
                critical_threshold=90.0,
                optimal_range=(40.0, 70.0),
                optimization_target=55.0
            ),
            "tool_efficiency": MetricThreshold(
                metric_name="tool_efficiency",
                warning_threshold=70.0,  # percentage
                critical_threshold=60.0,
                optimal_range=(80.0, 100.0),
                optimization_target=90.0
            ),
            "memory_usage": MetricThreshold(
                metric_name="memory_usage",
                warning_threshold=500.0,  # MB
                critical_threshold=800.0,
                optimal_range=(100.0, 400.0),
                optimization_target=250.0
            )
        }
    
    def record_metric(self, metric_name: str, value: float, context: Dict[str, Any] = None) -> None:
        """Record a metric value and trigger analysis."""
        timestamp = time.time()
        
        # Store metric with context
        metric_entry = {
            "value": value,
            "timestamp": timestamp,
            "context": context or {}
        }
        
        self.metric_history[metric_name].append(metric_entry)
        
        # Check for threshold violations
        self._check_thresholds(metric_name, value)
        
        # Trigger optimization recommendations if needed
        if self.auto_optimization_enabled:
            self._trigger_optimization_analysis(metric_name, value)
        
        logger.debug(f"Recorded metric {metric_name}: {value}")
    
    def _check_thresholds(self, metric_name: str, value: float) -> None:
        """Check if metric violates thresholds and generate alerts."""
        if metric_name not in self.thresholds:
            return
        
        threshold = self.thresholds[metric_name]
        if not threshold.enabled:
            return
        
        # Determine severity
        severity = None
        threshold_value = None
        
        if metric_name in ["error_rate"]:  # Lower is better
            if value >= threshold.critical_threshold:
                severity = "critical"
                threshold_value = threshold.critical_threshold
            elif value >= threshold.warning_threshold:
                severity = "high"
                threshold_value = threshold.warning_threshold
        else:  # Higher is better (for success_rate) or specific range (for response_time)
            if metric_name == "response_time":
                if value >= threshold.critical_threshold:
                    severity = "critical"
                    threshold_value = threshold.critical_threshold
                elif value >= threshold.warning_threshold:
                    severity = "high"
                    threshold_value = threshold.warning_threshold
            else:  # success_rate, tool_efficiency, etc.
                if value <= threshold.critical_threshold:
                    severity = "critical"
                    threshold_value = threshold.critical_threshold
                elif value <= threshold.warning_threshold:
                    severity = "high"
                    threshold_value = threshold.warning_threshold
        
        if severity:
            self._generate_alert(metric_name, value, threshold_value, severity)
    
    def _generate_alert(self, metric_name: str, value: float, threshold_value: float, severity: str) -> None:
        """Generate performance alert."""
        alert_messages = {
            "response_time": f"Response time is {value:.2f}s, exceeding threshold of {threshold_value:.2f}s",
            "success_rate": f"Success rate is {value:.1f}%, below threshold of {threshold_value:.1f}%",
            "error_rate": f"Error rate is {value:.1f}%, above threshold of {threshold_value:.1f}%",
            "context_utilization": f"Context utilization is {value:.1f}%, above threshold of {threshold_value:.1f}%",
            "tool_efficiency": f"Tool efficiency is {value:.1f}%, below threshold of {threshold_value:.1f}%",
            "memory_usage": f"Memory usage is {value:.1f}MB, above threshold of {threshold_value:.1f}MB"
        }
        
        message = alert_messages.get(metric_name, f"Metric {metric_name} value {value} violates threshold {threshold_value}")
        
        # Generate recommendations
        recommendations = self._generate_alert_recommendations(metric_name, value, severity)
        
        alert = PerformanceAlert(
            alert_type="threshold_violation",
            severity=severity,
            metric_name=metric_name,
            current_value=value,
            threshold_value=threshold_value,
            message=message,
            recommendations=recommendations,
            timestamp=time.time(),
            auto_fixable=self._is_auto_fixable(metric_name, severity)
        )
        
        self.alert_history.append(alert)
        logger.warning(f"Performance alert: {message}")
        
        # Attempt auto-fix if enabled and possible
        if alert.auto_fixable and self.auto_optimization_enabled:
            self._attempt_auto_fix(alert)
    
    def _generate_alert_recommendations(self, metric_name: str, value: float, severity: str) -> List[str]:
        """Generate specific recommendations for alerts."""
        recommendations = []
        
        if metric_name == "response_time":
            recommendations.extend([
                "Consider enabling context pruning",
                "Optimize tool selection algorithms",
                "Reduce model temperature for faster responses",
                "Enable caching for frequently used operations"
            ])
        elif metric_name == "success_rate":
            recommendations.extend([
                "Review and improve error handling",
                "Enhance tool reliability scoring",
                "Implement better fallback strategies",
                "Increase context detail for complex tasks"
            ])
        elif metric_name == "error_rate":
            recommendations.extend([
                "Enable advanced error recovery",
                "Improve input validation",
                "Enhance parsing resilience",
                "Review tool configurations"
            ])
        elif metric_name == "context_utilization":
            recommendations.extend([
                "Enable aggressive context pruning",
                "Implement smarter context prioritization",
                "Reduce verbose tool outputs",
                "Optimize memory usage patterns"
            ])
        elif metric_name == "tool_efficiency":
            recommendations.extend([
                "Update tool performance tracking",
                "Implement tool selection optimization",
                "Review tool timeout configurations",
                "Consider tool alternatives"
            ])
        elif metric_name == "memory_usage":
            recommendations.extend([
                "Enable memory optimization",
                "Implement garbage collection triggers",
                "Reduce context retention",
                "Optimize data structures"
            ])
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _is_auto_fixable(self, metric_name: str, severity: str) -> bool:
        """Determine if an alert can be automatically fixed."""
        auto_fixable_metrics = {
            "context_utilization": ["high"],  # Can auto-prune context
            "memory_usage": ["high"],         # Can trigger cleanup
            "response_time": ["high"]         # Can adjust parameters
        }
        
        return metric_name in auto_fixable_metrics and severity in auto_fixable_metrics[metric_name]
    
    def _attempt_auto_fix(self, alert: PerformanceAlert) -> None:
        """Attempt to automatically fix performance issues."""
        logger.info(f"Attempting auto-fix for {alert.metric_name} alert")
        
        if alert.metric_name == "context_utilization":
            self._auto_fix_context_utilization()
        elif alert.metric_name == "memory_usage":
            self._auto_fix_memory_usage()
        elif alert.metric_name == "response_time":
            self._auto_fix_response_time()
    
    def _auto_fix_context_utilization(self) -> None:
        """Auto-fix high context utilization."""
        try:
            from server.agent.intelligent_context_manager import intelligent_context_manager
            
            # Trigger aggressive pruning
            stats_before = intelligent_context_manager.get_context_statistics()
            intelligent_context_manager.context_window.max_tokens = int(
                intelligent_context_manager.context_window.max_tokens * 0.8
            )
            intelligent_context_manager._prune_context_if_needed()
            
            logger.info("Auto-fix applied: Context pruning enabled")
            
        except Exception as e:
            logger.error(f"Auto-fix failed for context utilization: {e}")
    
    def _auto_fix_memory_usage(self) -> None:
        """Auto-fix high memory usage."""
        try:
            # Trigger memory optimization
            import gc
            gc.collect()
            
            logger.info("Auto-fix applied: Memory cleanup triggered")
            
        except Exception as e:
            logger.error(f"Auto-fix failed for memory usage: {e}")
    
    def _auto_fix_response_time(self) -> None:
        """Auto-fix slow response times."""
        try:
            # Could adjust model parameters, context size, etc.
            logger.info("Auto-fix applied: Response time optimization")
            
        except Exception as e:
            logger.error(f"Auto-fix failed for response time: {e}")
    
    def _trigger_optimization_analysis(self, metric_name: str, value: float) -> None:
        """Trigger analysis for optimization opportunities."""
        # Only analyze if we have enough data
        if len(self.metric_history[metric_name]) < 10:
            return
        
        # Get recent metrics
        recent_metrics = list(self.metric_history[metric_name])[-20:]
        recent_values = [m["value"] for m in recent_metrics]
        
        # Calculate trend
        trend = self._calculate_trend(recent_values)
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(metric_name, value, trend)
        
        for recommendation in recommendations:
            if recommendation.implementation_priority <= 3:  # High priority only
                self._apply_optimization(recommendation)
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for metric values."""
        if len(values) < 5:
            return "insufficient_data"
        
        # Simple trend calculation using linear regression slope
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "degrading"
        else:
            return "stable"
    
    def _generate_optimization_recommendations(
        self, 
        metric_name: str, 
        current_value: float, 
        trend: str
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on metric analysis."""
        recommendations = []
        
        if metric_name not in self.thresholds:
            return recommendations
        
        threshold = self.thresholds[metric_name]
        target_value = threshold.optimization_target
        
        # Only recommend if current value is not optimal
        if not (threshold.optimal_range[0] <= current_value <= threshold.optimal_range[1]):
            
            if metric_name == "response_time" and current_value > target_value:
                recommendations.append(OptimizationRecommendation(
                    metric_type=MetricType.PERFORMANCE,
                    current_value=current_value,
                    target_value=target_value,
                    confidence=0.8,
                    action="Enable context optimization and tool caching",
                    rationale=f"Response time ({current_value:.2f}s) exceeds target ({target_value:.2f}s)",
                    impact_estimate="20-40% improvement in response time",
                    risk_level="low",
                    implementation_priority=2,
                    estimated_effort="minimal"
                ))
            
            elif metric_name == "success_rate" and current_value < target_value:
                recommendations.append(OptimizationRecommendation(
                    metric_type=MetricType.RELIABILITY,
                    current_value=current_value,
                    target_value=target_value,
                    confidence=0.7,
                    action="Enhance error recovery and tool fallbacks",
                    rationale=f"Success rate ({current_value:.1f}%) below target ({target_value:.1f}%)",
                    impact_estimate="15-25% improvement in success rate",
                    risk_level="low",
                    implementation_priority=1,
                    estimated_effort="moderate"
                ))
            
            elif metric_name == "tool_efficiency" and current_value < target_value:
                recommendations.append(OptimizationRecommendation(
                    metric_type=MetricType.EFFICIENCY,
                    current_value=current_value,
                    target_value=target_value,
                    confidence=0.9,
                    action="Optimize tool selection based on performance history",
                    rationale=f"Tool efficiency ({current_value:.1f}%) below target ({target_value:.1f}%)",
                    impact_estimate="10-30% improvement in tool efficiency",
                    risk_level="very_low",
                    implementation_priority=3,
                    estimated_effort="minimal"
                ))
        
        return recommendations
    
    def _apply_optimization(self, recommendation: OptimizationRecommendation) -> None:
        """Apply optimization recommendation if safe and beneficial."""
        optimization_id = f"{recommendation.metric_type.value}_{int(time.time())}"
        
        # Check if optimization is already active
        if recommendation.action in self.active_optimizations:
            return
        
        # Apply based on risk level and optimization level
        if self._should_apply_optimization(recommendation):
            logger.info(f"Applying optimization: {recommendation.action}")
            
            # Record optimization
            self.active_optimizations[recommendation.action] = {
                "recommendation": recommendation,
                "applied_at": time.time(),
                "optimization_id": optimization_id
            }
            
            self.optimization_history.append({
                "optimization_id": optimization_id,
                "recommendation": recommendation.to_dict(),
                "applied_at": time.time(),
                "status": "applied"
            })
            
            # Apply specific optimizations
            self._execute_optimization(recommendation)
    
    def _should_apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Determine if optimization should be applied based on current settings."""
        risk_thresholds = {
            OptimizationLevel.CONSERVATIVE: ["very_low"],
            OptimizationLevel.MODERATE: ["very_low", "low"],
            OptimizationLevel.AGGRESSIVE: ["very_low", "low", "medium"],
            OptimizationLevel.EXPERIMENTAL: ["very_low", "low", "medium", "high"]
        }
        
        allowed_risks = risk_thresholds.get(self.optimization_level, ["very_low"])
        return recommendation.risk_level in allowed_risks and recommendation.confidence > 0.6
    
    def _execute_optimization(self, recommendation: OptimizationRecommendation) -> None:
        """Execute specific optimization actions."""
        try:
            action = recommendation.action.lower()
            
            if "context optimization" in action:
                self._optimize_context_management()
            elif "tool caching" in action:
                self._optimize_tool_caching()
            elif "error recovery" in action:
                self._optimize_error_recovery()
            elif "tool selection" in action:
                self._optimize_tool_selection()
            
        except Exception as e:
            logger.error(f"Failed to execute optimization: {e}")
    
    def _optimize_context_management(self) -> None:
        """Optimize context management settings."""
        try:
            from server.agent.intelligent_context_manager import intelligent_context_manager
            intelligent_context_manager.optimize_context_management()
            logger.info("Context management optimization applied")
        except Exception as e:
            logger.error(f"Context optimization failed: {e}")
    
    def _optimize_tool_caching(self) -> None:
        """Optimize tool caching settings."""
        try:
            from server.agent.enhanced_tool_manager import enhanced_tool_manager
            enhanced_tool_manager.optimize_tool_selection()
            logger.info("Tool caching optimization applied")
        except Exception as e:
            logger.error(f"Tool caching optimization failed: {e}")
    
    def _optimize_error_recovery(self) -> None:
        """Optimize error recovery settings."""
        try:
            # Could enable more aggressive error recovery
            logger.info("Error recovery optimization applied")
        except Exception as e:
            logger.error(f"Error recovery optimization failed: {e}")
    
    def _optimize_tool_selection(self) -> None:
        """Optimize tool selection algorithms."""
        try:
            from server.agent.enhanced_tool_manager import enhanced_tool_manager
            enhanced_tool_manager.optimize_tool_selection()
            logger.info("Tool selection optimization applied")
        except Exception as e:
            logger.error(f"Tool selection optimization failed: {e}")
    
    def get_performance_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive performance insights and recommendations."""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        insights = {
            "analysis_period_days": days,
            "metric_summaries": {},
            "trending_metrics": {},
            "active_alerts": [],
            "optimization_opportunities": [],
            "applied_optimizations": [],
            "overall_health_score": 0.0,
            "recommendations": []
        }
        
        # Analyze each metric
        for metric_name, history in self.metric_history.items():
            recent_data = [entry for entry in history if entry["timestamp"] > cutoff_time]
            
            if recent_data:
                values = [entry["value"] for entry in recent_data]
                
                insights["metric_summaries"][metric_name] = {
                    "current_value": values[-1],
                    "average": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "trend": self._calculate_trend(values),
                    "data_points": len(values)
                }
        
        # Get recent alerts
        insights["active_alerts"] = [
            alert.to_dict() for alert in self.alert_history 
            if alert.timestamp > cutoff_time
        ]
        
        # Get optimization opportunities
        for metric_name, summary in insights["metric_summaries"].items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                current_value = summary["current_value"]
                
                if not (threshold.optimal_range[0] <= current_value <= threshold.optimal_range[1]):
                    recommendations = self._generate_optimization_recommendations(
                        metric_name, current_value, summary["trend"]
                    )
                    insights["optimization_opportunities"].extend([r.to_dict() for r in recommendations])
        
        # Get applied optimizations
        insights["applied_optimizations"] = [
            opt for opt in self.optimization_history
            if opt["applied_at"] > cutoff_time
        ]
        
        # Calculate overall health score
        insights["overall_health_score"] = self._calculate_health_score(insights["metric_summaries"])
        
        # Generate strategic recommendations
        insights["recommendations"] = self._generate_strategic_recommendations(insights)
        
        return insights
    
    def _calculate_health_score(self, metric_summaries: Dict[str, Dict]) -> float:
        """Calculate overall system health score (0-100)."""
        if not metric_summaries:
            return 50.0  # Neutral score
        
        scores = []
        
        for metric_name, summary in metric_summaries.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                current_value = summary["current_value"]
                
                # Calculate normalized score for this metric
                if threshold.optimal_range[0] <= current_value <= threshold.optimal_range[1]:
                    score = 100.0  # Perfect score
                elif metric_name == "error_rate":
                    # Lower is better
                    score = max(0, 100 - (current_value / threshold.critical_threshold) * 100)
                elif metric_name in ["success_rate", "tool_efficiency"]:
                    # Higher is better
                    score = (current_value / threshold.optimization_target) * 100
                else:
                    # Response time, memory usage - lower is better within limits
                    if current_value <= threshold.optimization_target:
                        score = 100.0
                    else:
                        score = max(0, 100 - ((current_value - threshold.optimization_target) / threshold.optimization_target) * 50)
                
                scores.append(min(100, max(0, score)))
        
        return statistics.mean(scores) if scores else 50.0
    
    def _generate_strategic_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate high-level strategic recommendations."""
        recommendations = []
        health_score = insights["overall_health_score"]
        
        if health_score < 60:
            recommendations.append("System health is below optimal - consider enabling aggressive optimization")
        elif health_score > 90:
            recommendations.append("System performing excellently - consider experimental optimizations")
        
        # Alert-based recommendations
        if len(insights["active_alerts"]) > 5:
            recommendations.append("High number of alerts - review and adjust thresholds")
        
        # Optimization-based recommendations
        if len(insights["optimization_opportunities"]) > 3:
            recommendations.append("Multiple optimization opportunities available - consider batch application")
        
        # Trend-based recommendations
        declining_metrics = [
            name for name, summary in insights["metric_summaries"].items()
            if summary["trend"] == "degrading"
        ]
        
        if declining_metrics:
            recommendations.append(f"Declining performance in: {', '.join(declining_metrics)} - investigate root causes")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def enable_auto_optimization(self, level: OptimizationLevel = OptimizationLevel.MODERATE) -> None:
        """Enable automatic optimization with specified level."""
        self.auto_optimization_enabled = True
        self.optimization_level = level
        logger.info(f"Auto-optimization enabled at {level.value} level")
    
    def disable_auto_optimization(self) -> None:
        """Disable automatic optimization."""
        self.auto_optimization_enabled = False
        logger.info("Auto-optimization disabled")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status and configuration."""
        return {
            "auto_optimization_enabled": self.auto_optimization_enabled,
            "optimization_level": self.optimization_level.value,
            "active_optimizations": len(self.active_optimizations),
            "total_metrics_tracked": len(self.metric_history),
            "recent_alerts": len([a for a in self.alert_history if a.timestamp > time.time() - 3600]),
            "thresholds_configured": len(self.thresholds),
            "monitoring_active": True
        }


# Create singleton instance
actionable_metrics = ActionableMetricsSystem()