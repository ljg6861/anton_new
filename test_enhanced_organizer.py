"""
Enhanced Organizer Test Suite

Comprehensive test of the enhanced organizer system demonstrating:
- Loop detection and prevention
- Structured tool usage enforcement
- Three-level evaluation system
- Performance monitoring
- Coordination with state tracking
"""

import asyncio
import json
import time
from typing import Dict, Any

from server.agent.enhanced_organizer import enhanced_organizer
from server.model_server import AgentChatRequest


class MockLogger:
    """Mock logger for testing."""
    
    def __init__(self):
        self.logs = []
    
    def info(self, msg):
        print(f"[INFO] {msg}")
        self.logs.append(("INFO", msg))
    
    def warning(self, msg):
        print(f"[WARNING] {msg}")
        self.logs.append(("WARNING", msg))
    
    def error(self, msg, exc_info=None):
        print(f"[ERROR] {msg}")
        self.logs.append(("ERROR", msg))
    
    def debug(self, msg):
        print(f"[DEBUG] {msg}")
        self.logs.append(("DEBUG", msg))


async def test_enhanced_organizer():
    """Test the enhanced organizer with various scenarios."""
    
    print("=" * 80)
    print("ENHANCED ORGANIZER COMPREHENSIVE TEST")
    print("=" * 80)
    
    logger = MockLogger()
    api_base_url = "http://localhost:8000"  # Mock URL for testing
    
    # Test cases
    test_cases = [
        {
            "name": "Code Analysis Task",
            "message": "Analyze the structure of the Anton agent system and identify the main components",
            "expected_features": ["loop_detection", "structured_doer", "three_level_evaluation"]
        },
        {
            "name": "File Exploration Task", 
            "message": "Find and examine the main organizer file to understand the workflow",
            "expected_features": ["context_building", "tool_validation", "performance_monitoring"]
        },
        {
            "name": "Complex Investigation",
            "message": "Investigate how tool execution works and identify any potential improvement areas",
            "expected_features": ["comprehensive_tracking", "optimization_suggestions"]
        }
    ]
    
    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*20} TEST {i}: {test_case['name']} {'='*20}")
        
        # Create test request
        request = AgentChatRequest(
            messages=[
                {"role": "user", "content": test_case["message"]}
            ],
            tools=[
                {"name": "read_file", "description": "Read file contents"},
                {"name": "list_directory", "description": "List directory contents"},
                {"name": "search_files", "description": "Search for files by pattern"}
            ]
        )
        
        print(f"Task: {test_case['message']}")
        print(f"Expected features: {', '.join(test_case['expected_features'])}")
        print("\nStarting enhanced organizer execution...")
        
        start_time = time.time()
        
        try:
            # Run enhanced organizer (with timeout for testing)
            response_parts = []
            async with asyncio.timeout(60):  # 1 minute timeout for test
                async for response_part in enhanced_organizer.run_enhanced_organizer_loop(
                    request, logger, api_base_url
                ):
                    response_parts.append(response_part)
                    print(f"Response part: {response_part[:100]}...")
            
            execution_time = time.time() - start_time
            full_response = "".join(response_parts)
            
            print(f"\nExecution completed in {execution_time:.2f} seconds")
            print(f"Response length: {len(full_response)} characters")
            
            # Get system status
            status = enhanced_organizer.get_current_status()
            print(f"\nSystem Status: {status.get('status', 'unknown')}")
            
            # Analyze results
            print("\n--- ANALYSIS ---")
            _analyze_test_results(test_case, status, logger.logs, execution_time)
            
        except asyncio.TimeoutError:
            print("Test timed out - this may indicate infinite loops or performance issues")
        except Exception as e:
            print(f"Test failed with error: {e}")
        
        print(f"\n{'='*60}")
    
    # Run component-specific tests
    print(f"\n{'='*30} COMPONENT TESTS {'='*30}")
    await test_loop_detection()
    await test_doer_enhancement()
    await test_evaluator_improvement()
    await test_performance_monitoring()
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print("="*80)


def _analyze_test_results(test_case: Dict, status: Dict, logs: list, execution_time: float):
    """Analyze test results and check for expected features."""
    
    expected_features = test_case.get("expected_features", [])
    detected_features = []
    
    # Check coordination features
    coordination = status.get("coordination", {})
    if coordination:
        detected_features.append("comprehensive_tracking")
        
        task_summary = coordination.get("task_summary", {})
        if task_summary.get("interaction_count", 0) > 0:
            detected_features.append("agent_coordination")
    
    # Check performance monitoring
    performance = status.get("performance", {})
    if performance.get("monitoring_active"):
        detected_features.append("performance_monitoring")
        
        if performance.get("optimization_suggestions"):
            detected_features.append("optimization_suggestions")
    
    # Check doer statistics
    doer_stats = status.get("doer_statistics", {})
    if doer_stats.get("total_executions", 0) > 0:
        detected_features.append("structured_doer")
        
        if doer_stats.get("response_type_distribution", {}).get("tool_call", 0) > 0:
            detected_features.append("tool_validation")
    
    # Check evaluator statistics  
    evaluator_stats = status.get("evaluator_statistics", {})
    if evaluator_stats.get("total_evaluations", 0) > 0:
        level_dist = evaluator_stats.get("level_distribution", {})
        if "PARTIAL" in level_dist:
            detected_features.append("three_level_evaluation")
    
    # Check logs for specific patterns
    log_messages = [msg for level, msg in logs]
    log_text = " ".join(log_messages).lower()
    
    if "loop detected" in log_text:
        detected_features.append("loop_detection")
    
    if "context building" in log_text:
        detected_features.append("context_building")
    
    # Performance analysis
    print(f"Execution time: {execution_time:.2f}s")
    if execution_time > 30:
        print("⚠️  Long execution time - may need optimization")
    else:
        print("✅ Good execution time")
    
    # Feature analysis
    print(f"Expected features: {len(expected_features)}")
    print(f"Detected features: {len(detected_features)}")
    
    for feature in expected_features:
        if feature in detected_features:
            print(f"✅ {feature}")
        else:
            print(f"❌ {feature} (not detected)")
    
    # Additional detected features
    additional = set(detected_features) - set(expected_features)
    if additional:
        print(f"➕ Additional features: {', '.join(additional)}")


async def test_loop_detection():
    """Test loop detection specifically."""
    print("\n--- LOOP DETECTION TEST ---")
    
    from server.agent.loop_detector import LoopDetector
    
    detector = LoopDetector(similarity_threshold=0.85)
    
    # Simulate similar instructions
    detector.add_instruction("Read the main config file", 1, time.time(), "planner")
    detector.add_instruction("Read the configuration file", 2, time.time(), "planner")  # Similar
    detector.add_instruction("Read the main config file again", 3, time.time(), "planner")  # Very similar
    
    is_loop, pattern_breaking, loop_info = detector.detect_loop()
    
    print(f"Loop detected: {is_loop}")
    if is_loop:
        print(f"Pattern breaking instruction: {pattern_breaking}")
        print(f"Loop info: {loop_info}")
        print("✅ Loop detection working")
    else:
        print("❌ Loop detection not triggered")
    
    print(f"Detector status: {detector.get_status()}")


async def test_doer_enhancement():
    """Test enhanced Doer component."""
    print("\n--- ENHANCED DOER TEST ---")
    
    from server.agent.enhanced_doer import EnhancedDoer, ResponseType, ExecutionStatus
    
    doer = EnhancedDoer(max_response_time=5.0)
    
    # Test structured response parsing
    test_responses = [
        {
            "content": '<tool_code>{"name": "read_file", "arguments": {"file_path": "test.py"}}</tool_code>',
            "expected_type": ResponseType.TOOL_CALL
        },
        {
            "content": "FINAL ANSWER: The task has been completed successfully.",
            "expected_type": ResponseType.FINAL_ANSWER
        },
        {
            "content": "Hello, I will help you with this task...",
            "expected_type": ResponseType.INVALID  # Conversational
        }
    ]
    
    for i, test in enumerate(test_responses):
        result = doer._parse_and_validate_response(test["content"], MockLogger())
        
        print(f"Test {i+1}: Expected {test['expected_type'].value}, Got {result.response_type.value}")
        if result.response_type == test["expected_type"]:
            print("✅ Correct classification")
        else:
            print("❌ Incorrect classification")
        
        if result.validation_errors:
            print(f"   Validation errors: {result.validation_errors}")
    
    print(f"Doer statistics: {doer.get_execution_statistics()}")


async def test_evaluator_improvement():
    """Test enhanced Evaluator component."""
    print("\n--- ENHANCED EVALUATOR TEST ---")
    
    from server.agent.enhanced_evaluator import EnhancedEvaluator, EvaluationLevel
    from server.agent.enhanced_doer import DoerResponse, ResponseType, ExecutionStatus
    
    evaluator = EnhancedEvaluator()
    
    # Test evaluation scenarios
    test_scenarios = [
        {
            "name": "Successful tool execution",
            "doer_response": DoerResponse(
                response_type=ResponseType.TOOL_CALL,
                content="Successfully read the file and found the configuration",
                tool_calls=[],
                execution_status=ExecutionStatus.SUCCESS,
                duration=2.0,
                raw_response="test",
                validation_errors=[]
            ),
            "expected_level": EvaluationLevel.SUCCESS
        },
        {
            "name": "Partial progress",
            "doer_response": DoerResponse(
                response_type=ResponseType.TOOL_CALL,
                content="Found some files but need to read them",
                tool_calls=[],
                execution_status=ExecutionStatus.SUCCESS,
                duration=1.0,
                raw_response="test",
                validation_errors=[]
            ),
            "expected_level": EvaluationLevel.PARTIAL
        },
        {
            "name": "Failed execution",
            "doer_response": DoerResponse(
                response_type=ResponseType.INVALID,
                content="I'm not sure what to do",
                tool_calls=[],
                execution_status=ExecutionStatus.FAILED,
                duration=0.5,
                raw_response="test",
                validation_errors=["No tool calls made"]
            ),
            "expected_level": EvaluationLevel.FAILURE
        }
    ]
    
    for scenario in test_scenarios:
        result = await evaluator.evaluate_doer_result(
            original_task="Test task",
            delegated_instruction="Test instruction", 
            doer_response=scenario["doer_response"],
            context_store={},
            logger=MockLogger()
        )
        
        print(f"{scenario['name']}: Expected {scenario['expected_level'].value}, Got {result.level.value}")
        print(f"   Reason: {result.reason}")
        print(f"   Progress score: {result.progress_score:.2f}")
        
        if result.level == scenario["expected_level"]:
            print("✅ Correct evaluation")
        else:
            print("❌ Incorrect evaluation")
    
    print(f"Evaluator statistics: {evaluator.get_evaluation_statistics()}")


async def test_performance_monitoring():
    """Test performance monitoring component."""
    print("\n--- PERFORMANCE MONITORING TEST ---")
    
    from server.agent.performance_monitor import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Simulate operations
    op1 = monitor.start_operation("test_operation", "doer", {"test": True})
    await asyncio.sleep(0.1)  # Simulate work
    monitor.end_operation(op1, success=True)
    
    op2 = monitor.start_operation("slow_operation", "evaluator", {"test": True})
    await asyncio.sleep(0.2)  # Simulate slower work
    monitor.end_operation(op2, success=False)
    
    # Get performance report
    report = monitor.get_performance_report()
    
    print(f"Performance level: {report.get('performance_level', 'unknown')}")
    print(f"Total operations: {report.get('operation_statistics', {}).get('total_operations', 0)}")
    print(f"Success rate: {report.get('operation_statistics', {}).get('success_rate', 0):.1%}")
    
    bottlenecks = monitor.identify_bottlenecks()
    suggestions = monitor.generate_optimization_suggestions()
    
    print(f"Bottlenecks identified: {len(bottlenecks)}")
    print(f"Optimization suggestions: {len(suggestions)}")
    
    if bottlenecks or suggestions:
        print("✅ Performance analysis working")
    else:
        print("ℹ️  No performance issues detected (expected for short test)")
    
    monitor.stop_monitoring()


if __name__ == "__main__":
    print("Running Enhanced Organizer Test Suite...")
    asyncio.run(test_enhanced_organizer())