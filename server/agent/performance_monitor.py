"""
Performance Monitoring System

Tracks operation durations, identifies bottlenecks, provides optimization suggestions,
and monitors resource usage with actionable insights.
"""

import time
import threading

# Make psutil optional for environments where it's not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict


class PerformanceLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    operation_type: str  # 'agent_turn', 'tool_call', 'evaluation', etc.
    metadata: Dict = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        return self.duration * 1000


@dataclass
class ResourceSnapshot:
    """System resource snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0


@dataclass
class PerformanceAlert:
    """Performance alert for threshold violations."""
    alert_type: str
    severity: PerformanceLevel
    message: str
    timestamp: float
    operation: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Features:
    - Operation duration tracking with categorization
    - Resource usage monitoring (CPU, memory, disk I/O)
    - Bottleneck identification and analysis
    - Optimization suggestion generation
    - Threshold-based alerting
    - Performance trend analysis
    """
    
    def __init__(self,
                 max_operation_duration: float = 30.0,
                 max_memory_percent: float = 80.0,
                 max_cpu_percent: float = 90.0,
                 monitoring_interval: float = 5.0,
                 max_history_size: int = 1000):
        
        # Configuration
        self.max_operation_duration = max_operation_duration
        self.max_memory_percent = max_memory_percent
        self.max_cpu_percent = max_cpu_percent
        self.monitoring_interval = monitoring_interval
        self.max_history_size = max_history_size
        
        # Data storage
        self.operation_metrics: deque = deque(maxlen=max_history_size)
        self.resource_snapshots: deque = deque(maxlen=max_history_size)
        self.performance_alerts: deque = deque(maxlen=100)  # Keep last 100 alerts
        
        # Operation tracking
        self.active_operations: Dict[str, float] = {}  # operation_id -> start_time
        self.operation_categories: Dict[str, List[OperationMetrics]] = defaultdict(list)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.baseline_metrics: Optional[Dict] = None
        
        # Performance thresholds
        self.thresholds = {
            'doer_response_time': 15.0,  # seconds
            'tool_call_time': 10.0,      # seconds
            'evaluation_time': 5.0,       # seconds
            'memory_usage': 80.0,         # percent
            'cpu_usage': 90.0,            # percent
        }
        
    def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Capture baseline metrics
        self._capture_baseline()
    
    def stop_monitoring(self):
        """Stop continuous resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
    
    def start_operation(self, operation_name: str, operation_type: str, metadata: Dict = None) -> str:
        """Start tracking an operation and return operation ID."""
        operation_id = f"{operation_type}_{operation_name}_{int(time.time() * 1000)}"
        self.active_operations[operation_id] = time.time()
        
        # Store metadata for later use
        if not hasattr(self, '_operation_metadata'):
            self._operation_metadata = {}
        self._operation_metadata[operation_id] = {
            'name': operation_name,
            'type': operation_type,
            'metadata': metadata or {}
        }
        
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True) -> OperationMetrics:
        """End tracking an operation and record metrics."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found in active operations")
        
        start_time = self.active_operations.pop(operation_id)
        end_time = time.time()
        duration = end_time - start_time
        
        # Get stored metadata
        metadata_info = getattr(self, '_operation_metadata', {}).get(operation_id, {})
        operation_name = metadata_info.get('name', 'unknown')
        operation_type = metadata_info.get('type', 'unknown')
        metadata = metadata_info.get('metadata', {})
        
        # Create metrics record
        metrics = OperationMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            success=success,
            operation_type=operation_type,
            metadata=metadata
        )
        
        # Store metrics
        self.operation_metrics.append(metrics)
        self.operation_categories[operation_type].append(metrics)
        
        # Check for performance alerts
        self._check_operation_thresholds(metrics)
        
        # Clean up metadata
        if hasattr(self, '_operation_metadata'):
            self._operation_metadata.pop(operation_id, None)
        
        return metrics
    
    def record_resource_snapshot(self) -> ResourceSnapshot:
        """Record current resource usage."""
        try:
            # Default values for when psutil is not available
            cpu_percent = 0.0
            memory_percent = 0.0
            memory_mb = 0.0
            disk_io_read_mb = 0.0
            disk_io_write_mb = 0.0
            
            if PSUTIL_AVAILABLE:
                # Get system resource information
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_mb = memory.used / (1024 * 1024)
                
                # Get disk I/O if available
                try:
                    disk_io = psutil.disk_io_counters()
                    if disk_io:
                        disk_io_read_mb = disk_io.read_bytes / (1024 * 1024)
                        disk_io_write_mb = disk_io.write_bytes / (1024 * 1024)
                except Exception:
                    pass  # Disk I/O not available on all systems
            
            snapshot = ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_mb,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb
            )
            
            self.resource_snapshots.append(snapshot)
            
            # Check resource thresholds only if psutil is available
            if PSUTIL_AVAILABLE:
                self._check_resource_thresholds(snapshot)
            
            return snapshot
            
        except Exception as e:
            # Return minimal snapshot on error
            return ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_mb=0.0
            )
    
    def identify_bottlenecks(self) -> List[Dict]:
        """Identify performance bottlenecks based on collected metrics."""
        bottlenecks = []
        
        if not self.operation_metrics:
            return bottlenecks
        
        # Analyze operation types
        for operation_type, metrics_list in self.operation_categories.items():
            if len(metrics_list) < 2:
                continue
            
            # Calculate statistics
            durations = [m.duration for m in metrics_list]
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            success_rate = sum(1 for m in metrics_list if m.success) / len(metrics_list)
            
            # Check for bottlenecks
            if avg_duration > self.thresholds.get(f"{operation_type}_time", 10.0):
                bottlenecks.append({
                    'type': 'slow_operation',
                    'operation_type': operation_type,
                    'average_duration': avg_duration,
                    'max_duration': max_duration,
                    'sample_count': len(metrics_list),
                    'severity': self._calculate_severity(avg_duration, 10.0)
                })
            
            if success_rate < 0.8:  # Less than 80% success rate
                bottlenecks.append({
                    'type': 'low_success_rate',
                    'operation_type': operation_type,
                    'success_rate': success_rate,
                    'total_operations': len(metrics_list),
                    'severity': self._calculate_severity(0.8 - success_rate, 0.2)
                })
        
        # Analyze resource usage
        if self.resource_snapshots:
            recent_snapshots = list(self.resource_snapshots)[-10:]  # Last 10 snapshots
            
            avg_cpu = sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots)
            avg_memory = sum(s.memory_percent for s in recent_snapshots) / len(recent_snapshots)
            
            if avg_cpu > self.max_cpu_percent:
                bottlenecks.append({
                    'type': 'high_cpu_usage',
                    'average_cpu_percent': avg_cpu,
                    'threshold': self.max_cpu_percent,
                    'severity': self._calculate_severity(avg_cpu, self.max_cpu_percent)
                })
            
            if avg_memory > self.max_memory_percent:
                bottlenecks.append({
                    'type': 'high_memory_usage',
                    'average_memory_percent': avg_memory,
                    'threshold': self.max_memory_percent,
                    'severity': self._calculate_severity(avg_memory, self.max_memory_percent)
                })
        
        return sorted(bottlenecks, key=lambda x: x.get('severity', 0), reverse=True)
    
    def generate_optimization_suggestions(self) -> List[Dict]:
        """Generate actionable optimization suggestions based on performance data."""
        suggestions = []
        bottlenecks = self.identify_bottlenecks()
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'slow_operation':
                op_type = bottleneck['operation_type']
                avg_duration = bottleneck['average_duration']
                
                if op_type == 'doer' and avg_duration > 15.0:
                    suggestions.append({
                        'category': 'response_optimization',
                        'priority': 'high',
                        'description': f"Doer responses averaging {avg_duration:.1f}s (target: <15s)",
                        'actions': [
                            "Implement response timeouts",
                            "Optimize model parameters for faster generation",
                            "Consider breaking complex tasks into smaller steps"
                        ],
                        'estimated_impact': 'high'
                    })
                
                elif op_type == 'tool_call' and avg_duration > 10.0:
                    suggestions.append({
                        'category': 'tool_optimization',
                        'priority': 'medium',
                        'description': f"Tool calls averaging {avg_duration:.1f}s (target: <10s)",
                        'actions': [
                            "Add tool call timeouts",
                            "Optimize tool implementations",
                            "Cache frequently accessed data"
                        ],
                        'estimated_impact': 'medium'
                    })
            
            elif bottleneck['type'] == 'low_success_rate':
                op_type = bottleneck['operation_type']
                success_rate = bottleneck['success_rate']
                
                suggestions.append({
                    'category': 'reliability_improvement',
                    'priority': 'high',
                    'description': f"{op_type} operations have {success_rate:.1%} success rate",
                    'actions': [
                        "Improve error handling and recovery",
                        "Add input validation",
                        "Implement retry mechanisms"
                    ],
                    'estimated_impact': 'high'
                })
            
            elif bottleneck['type'] == 'high_memory_usage':
                suggestions.append({
                    'category': 'memory_optimization',
                    'priority': 'high',
                    'description': f"Memory usage averaging {bottleneck['average_memory_percent']:.1f}%",
                    'actions': [
                        "Implement context pruning",
                        "Clear unused data structures",
                        "Optimize data storage formats"
                    ],
                    'estimated_impact': 'high'
                })
            
            elif bottleneck['type'] == 'high_cpu_usage':
                suggestions.append({
                    'category': 'cpu_optimization',
                    'priority': 'medium',
                    'description': f"CPU usage averaging {bottleneck['average_cpu_percent']:.1f}%",
                    'actions': [
                        "Optimize computation-heavy operations",
                        "Implement operation queuing",
                        "Consider parallel processing"
                    ],
                    'estimated_impact': 'medium'
                })
        
        # Add general suggestions based on trends
        if len(self.operation_metrics) > 10:
            recent_ops = list(self.operation_metrics)[-10:]
            long_ops = [op for op in recent_ops if op.duration > 20.0]
            
            if len(long_ops) > 3:
                suggestions.append({
                    'category': 'general_optimization',
                    'priority': 'medium',
                    'description': "Multiple operations taking longer than 20 seconds",
                    'actions': [
                        "Review task complexity and break down if needed",
                        "Implement progressive timeouts",
                        "Add operation cancellation capabilities"
                    ],
                    'estimated_impact': 'medium'
                })
        
        return suggestions
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        if not self.operation_metrics and not self.resource_snapshots:
            return {'error': 'No performance data available'}
        
        # Operation statistics
        operation_stats = {}
        total_operations = len(self.operation_metrics)
        
        if total_operations > 0:
            all_durations = [op.duration for op in self.operation_metrics]
            all_successes = sum(1 for op in self.operation_metrics if op.success)
            
            operation_stats = {
                'total_operations': total_operations,
                'average_duration': sum(all_durations) / len(all_durations),
                'max_duration': max(all_durations),
                'min_duration': min(all_durations),
                'success_rate': all_successes / total_operations,
                'total_time': sum(all_durations)
            }
            
            # Per-operation-type statistics
            type_stats = {}
            for op_type, ops in self.operation_categories.items():
                durations = [op.duration for op in ops]
                successes = sum(1 for op in ops if op.success)
                
                type_stats[op_type] = {
                    'count': len(ops),
                    'average_duration': sum(durations) / len(durations) if durations else 0,
                    'success_rate': successes / len(ops) if ops else 0,
                    'total_time': sum(durations)
                }
            
            operation_stats['by_type'] = type_stats
        
        # Resource statistics
        resource_stats = {}
        if self.resource_snapshots:
            recent_snapshots = list(self.resource_snapshots)[-20:]  # Last 20 snapshots
            
            resource_stats = {
                'average_cpu_percent': sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots),
                'max_cpu_percent': max(s.cpu_percent for s in recent_snapshots),
                'average_memory_percent': sum(s.memory_percent for s in recent_snapshots) / len(recent_snapshots),
                'max_memory_percent': max(s.memory_percent for s in recent_snapshots),
                'average_memory_mb': sum(s.memory_mb for s in recent_snapshots) / len(recent_snapshots),
                'snapshot_count': len(self.resource_snapshots)
            }
        
        # Performance analysis
        bottlenecks = self.identify_bottlenecks()
        suggestions = self.generate_optimization_suggestions()
        alerts = [
            {
                'type': alert.alert_type,
                'severity': alert.severity.value,
                'message': alert.message,
                'timestamp': alert.timestamp
            }
            for alert in list(self.performance_alerts)[-10:]  # Last 10 alerts
        ]
        
        # Overall performance level
        performance_level = self._calculate_overall_performance_level()
        
        return {
            'performance_level': performance_level.value,
            'operation_statistics': operation_stats,
            'resource_statistics': resource_stats,
            'bottlenecks': bottlenecks,
            'optimization_suggestions': suggestions,
            'recent_alerts': alerts,
            'monitoring_active': self.monitoring_active,
            'baseline_comparison': self._compare_to_baseline() if self.baseline_metrics else None
        }
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.monitoring_active:
            try:
                self.record_resource_snapshot()
                time.sleep(self.monitoring_interval)
            except Exception:
                continue  # Continue monitoring even if snapshot fails
    
    def _capture_baseline(self):
        """Capture baseline performance metrics."""
        # Wait a moment to get initial readings
        time.sleep(1.0)
        
        snapshot = self.record_resource_snapshot()
        self.baseline_metrics = {
            'cpu_percent': snapshot.cpu_percent,
            'memory_percent': snapshot.memory_percent,
            'memory_mb': snapshot.memory_mb,
            'timestamp': snapshot.timestamp
        }
    
    def _check_operation_thresholds(self, metrics: OperationMetrics):
        """Check if operation metrics exceed thresholds."""
        threshold_key = f"{metrics.operation_type}_time"
        threshold = self.thresholds.get(threshold_key, self.max_operation_duration)
        
        if metrics.duration > threshold:
            severity = self._calculate_severity_level(metrics.duration, threshold)
            alert = PerformanceAlert(
                alert_type='slow_operation',
                severity=severity,
                message=f"{metrics.operation_type} '{metrics.operation_name}' took {metrics.duration:.1f}s (threshold: {threshold:.1f}s)",
                timestamp=time.time(),
                operation=metrics.operation_name,
                value=metrics.duration,
                threshold=threshold
            )
            self.performance_alerts.append(alert)
    
    def _check_resource_thresholds(self, snapshot: ResourceSnapshot):
        """Check if resource usage exceeds thresholds."""
        if snapshot.memory_percent > self.max_memory_percent:
            severity = self._calculate_severity_level(snapshot.memory_percent, self.max_memory_percent)
            alert = PerformanceAlert(
                alert_type='high_memory',
                severity=severity,
                message=f"Memory usage at {snapshot.memory_percent:.1f}% (threshold: {self.max_memory_percent:.1f}%)",
                timestamp=snapshot.timestamp,
                value=snapshot.memory_percent,
                threshold=self.max_memory_percent
            )
            self.performance_alerts.append(alert)
        
        if snapshot.cpu_percent > self.max_cpu_percent:
            severity = self._calculate_severity_level(snapshot.cpu_percent, self.max_cpu_percent)
            alert = PerformanceAlert(
                alert_type='high_cpu',
                severity=severity,
                message=f"CPU usage at {snapshot.cpu_percent:.1f}% (threshold: {self.max_cpu_percent:.1f}%)",
                timestamp=snapshot.timestamp,
                value=snapshot.cpu_percent,
                threshold=self.max_cpu_percent
            )
            self.performance_alerts.append(alert)
    
    def _calculate_severity(self, value: float, threshold: float) -> float:
        """Calculate severity score (0-1) based on how much a value exceeds threshold."""
        if value <= threshold:
            return 0.0
        
        excess_ratio = (value - threshold) / threshold
        return min(1.0, excess_ratio)
    
    def _calculate_severity_level(self, value: float, threshold: float) -> PerformanceLevel:
        """Calculate severity level based on threshold exceedance."""
        if value <= threshold:
            return PerformanceLevel.GOOD
        
        excess_ratio = (value - threshold) / threshold
        
        if excess_ratio >= 1.0:  # 100% over threshold
            return PerformanceLevel.CRITICAL
        elif excess_ratio >= 0.5:  # 50% over threshold
            return PerformanceLevel.POOR
        elif excess_ratio >= 0.2:  # 20% over threshold
            return PerformanceLevel.FAIR
        else:
            return PerformanceLevel.GOOD
    
    def _calculate_overall_performance_level(self) -> PerformanceLevel:
        """Calculate overall performance level based on all metrics."""
        if not self.operation_metrics and not self.resource_snapshots:
            return PerformanceLevel.GOOD
        
        # Check recent alerts for critical issues
        recent_alerts = [a for a in self.performance_alerts if time.time() - a.timestamp < 300]  # Last 5 minutes
        critical_alerts = [a for a in recent_alerts if a.severity == PerformanceLevel.CRITICAL]
        poor_alerts = [a for a in recent_alerts if a.severity == PerformanceLevel.POOR]
        
        if critical_alerts:
            return PerformanceLevel.CRITICAL
        elif len(poor_alerts) >= 3:
            return PerformanceLevel.POOR
        elif poor_alerts:
            return PerformanceLevel.FAIR
        
        # Check operation success rates
        if self.operation_metrics:
            recent_ops = list(self.operation_metrics)[-20:]  # Last 20 operations
            success_rate = sum(1 for op in recent_ops if op.success) / len(recent_ops)
            
            if success_rate < 0.7:
                return PerformanceLevel.POOR
            elif success_rate < 0.9:
                return PerformanceLevel.FAIR
        
        return PerformanceLevel.GOOD
    
    def _compare_to_baseline(self) -> Dict:
        """Compare current performance to baseline."""
        if not self.baseline_metrics or not self.resource_snapshots:
            return {}
        
        recent_snapshot = self.resource_snapshots[-1]
        
        return {
            'cpu_change': recent_snapshot.cpu_percent - self.baseline_metrics['cpu_percent'],
            'memory_change': recent_snapshot.memory_percent - self.baseline_metrics['memory_percent'],
            'memory_mb_change': recent_snapshot.memory_mb - self.baseline_metrics['memory_mb'],
            'baseline_timestamp': self.baseline_metrics['timestamp'],
            'current_timestamp': recent_snapshot.timestamp
        }
    
    def reset_statistics(self):
        """Reset all performance statistics."""
        self.operation_metrics.clear()
        self.resource_snapshots.clear()
        self.performance_alerts.clear()
        self.active_operations.clear()
        self.operation_categories.clear()
        self.baseline_metrics = None