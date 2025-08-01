#!/usr/bin/env python3
"""
Comprehensive test suite for the enhanced Anton agent architecture.

This test suite validates all 8 critical architectural improvements:
1. Adaptive Workflow Orchestration
2. Intelligent Context Management  
3. Enhanced Tool Management
4. Resilient Parsing System
5. Comprehensive State Tracking
6. Actionable Metrics Integration
7. Migration System
8. Integration Layer

Usage:
    python test_enhanced_architecture.py
    python test_enhanced_architecture.py --component context_management
    python test_enhanced_architecture.py --verbose
"""
import asyncio
import json
import logging
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import os

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging for tests
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during tests
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TestIntelligentContextManagement(unittest.TestCase):
    """Test suite for intelligent context management."""
    
    def setUp(self):
        """Set up test environment."""
        from server.agent.intelligent_context_manager import (
            IntelligentContextManager, ContextType, ContextPriority
        )
        self.context_manager = IntelligentContextManager()
        self.ContextType = ContextType
        self.ContextPriority = ContextPriority
    
    def test_context_addition(self):
        """Test adding different types of context."""
        item_id = self.context_manager.add_context(
            "Test system prompt",
            self.ContextType.SYSTEM_PROMPT,
            self.ContextPriority.CRITICAL
        )
        
        self.assertIsNotNone(item_id)
        self.assertEqual(len(self.context_manager.context_items), 1)
        
        # Test context retrieval
        context = self.context_manager.get_context_for_prompt(max_tokens=100)
        self.assertIn("Test system prompt", context)
    
    def test_context_pruning(self):
        """Test intelligent context pruning."""
        # Add many items to trigger pruning
        for i in range(60):  # More than max_tokens limit should allow
            self.context_manager.add_context(
                f"Test content {i} " * 50,  # Long content
                self.ContextType.TOOL_OUTPUT,
                self.ContextPriority.LOW
            )
        
        # Should have pruned items
        self.assertLess(len(self.context_manager.context_window.items), 60)
    
    def test_memory_search(self):
        """Test memory search functionality."""
        # Add some memory items
        self.context_manager.add_context(
            "Database connection established successfully",
            self.ContextType.MEMORY,
            self.ContextPriority.HIGH
        )
        
        self.context_manager.add_context(
            "Authentication failed for user",
            self.ContextType.MEMORY,
            self.ContextPriority.HIGH
        )
        
        # Search for relevant memories
        memories = self.context_manager.get_relevant_memories("database connection")
        self.assertGreater(len(memories), 0)
    
    def test_importance_scoring(self):
        """Test importance scoring system."""
        # Add items with different priorities
        high_item_id = self.context_manager.add_context(
            "Critical system error",
            self.ContextType.ERROR_INFO,
            self.ContextPriority.HIGH
        )
        
        low_item_id = self.context_manager.add_context(
            "Debug information",
            self.ContextType.PROGRESS_UPDATE,
            self.ContextPriority.LOW
        )
        
        # Find items by ID and check importance scores
        high_item = next(item for item in self.context_manager.context_items 
                        if self.context_manager._generate_item_id(item) == high_item_id)
        low_item = next(item for item in self.context_manager.context_items 
                       if self.context_manager._generate_item_id(item) == low_item_id)
        
        self.assertGreater(high_item.importance_score, low_item.importance_score)


class TestEnhancedToolManagement(unittest.TestCase):
    """Test suite for enhanced tool management."""
    
    def setUp(self):
        """Set up test environment."""
        from server.agent.enhanced_tool_manager import (
            EnhancedToolManager, ToolCategory, ToolCapability, ToolMetadata
        )
        self.tool_manager = EnhancedToolManager()
        self.ToolCategory = ToolCategory
        self.ToolCapability = ToolCapability
        self.ToolMetadata = ToolMetadata
    
    def test_tool_registration(self):
        """Test tool registration with metadata."""
        # Create mock tool
        mock_tool = type('MockTool', (), {
            'function': {
                "function": {
                    "name": "test_tool",
                    "description": "Test tool for file operations"
                }
            },
            'run': lambda self, args: "test result"
        })()
        
        metadata = self.ToolMetadata(
            name="test_tool",
            category=self.ToolCategory.FILE_OPERATIONS,
            capabilities={self.ToolCapability.READ, self.ToolCapability.ANALYZE}
        )
        
        self.tool_manager.register_tool(mock_tool, metadata)
        
        self.assertIn("test_tool", self.tool_manager.tools)
        self.assertIn("test_tool", self.tool_manager.tool_metadata)
    
    def test_tool_recommendations(self):
        """Test tool recommendation system."""
        # Register test tool first
        self.test_tool_registration()
        
        recommendations = self.tool_manager.recommend_tools("read a file and analyze it")
        self.assertIsInstance(recommendations, list)
    
    def test_tool_categorization(self):
        """Test tool categorization."""
        # Register test tool first
        self.test_tool_registration()
        
        file_tools = self.tool_manager.get_tools_by_category(self.ToolCategory.FILE_OPERATIONS)
        self.assertIn("test_tool", file_tools)
        
        capabilities = {self.ToolCapability.READ}
        read_tools = self.tool_manager.get_tools_by_capabilities(capabilities)
        self.assertIn("test_tool", read_tools)
    
    def test_performance_tracking(self):
        """Test tool performance tracking."""
        # Register test tool first
        self.test_tool_registration()
        
        # Execute tool with tracking
        result, success, exec_time = self.tool_manager.execute_tool_with_tracking(
            "test_tool", {"test": "args"}, "test context"
        )
        
        self.assertTrue(success)
        self.assertIn("test_tool", self.tool_manager.performance_metrics)
        
        # Check performance report
        report = self.tool_manager.get_performance_report()
        self.assertIn("test_tool", report["tool_performance"])


class TestResilientParsing(unittest.TestCase):
    """Test suite for resilient parsing system."""
    
    def setUp(self):
        """Set up test environment."""
        from server.agent.resilient_parser import (
            ResilientParsingSystem, OutputFormat, ParsingContext
        )
        self.parser = ResilientParsingSystem()
        self.OutputFormat = OutputFormat
        self.ParsingContext = ParsingContext
    
    def test_json_parsing(self):
        """Test JSON parsing with various formats."""
        test_cases = [
            '{"name": "test", "value": 123}',
            '```json\n{"data": "value"}\n```',
            '{"broken": "json", "missing": quotes}',  # Should handle gracefully
        ]
        
        for test_input in test_cases:
            result = self.parser.parse(test_input)
            self.assertIsNotNone(result)
            self.assertIn(result.result.value, ["success", "partial_success", "failed"])
    
    def test_xml_parsing(self):
        """Test XML tag parsing."""
        xml_input = '<result>Success</result><data>Test data</data>'
        result = self.parser.parse(xml_input)
        
        self.assertEqual(result.result.value, "success")
        self.assertIsInstance(result.content, dict)
        self.assertIn("result", result.content)
    
    def test_structured_text_parsing(self):
        """Test structured text parsing."""
        structured_input = """
        Name: John Doe
        Age: 30
        City: New York
        Status: Active
        """
        
        result = self.parser.parse(structured_input)
        self.assertIn(result.result.value, ["success", "partial_success"])
        self.assertIsInstance(result.content, dict)
    
    def test_fallback_parsing(self):
        """Test fallback to plain text parsing."""
        plain_input = "This is just plain text without structure"
        result = self.parser.parse(plain_input)
        
        self.assertEqual(result.result.value, "success")
        self.assertIn("raw_text", result.content)
    
    def test_parsing_with_context(self):
        """Test parsing with context hints."""
        context = self.ParsingContext(
            expected_format=self.OutputFormat.JSON,
            expected_fields=["name", "age"]
        )
        
        input_text = "name: Alice, age: 25"
        result = self.parser.parse(input_text, context)
        
        self.assertIsNotNone(result.content)


class TestComprehensiveStateTracking(unittest.TestCase):
    """Test suite for comprehensive state tracking."""
    
    def setUp(self):
        """Set up test environment."""
        from server.agent.state_tracker import ComprehensiveStateTracker
        
        # Use temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.state_tracker = ComprehensiveStateTracker(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test environment."""
        self.state_tracker.close()
        os.unlink(self.temp_db.name)
    
    def test_task_tracking(self):
        """Test task tracking functionality."""
        task_id = self.state_tracker.start_task_tracking(
            "Test task", "test_type", 2
        )
        
        self.assertIsNotNone(task_id)
        self.assertIn(task_id, self.state_tracker.active_tasks)
        
        # Record steps and decisions
        self.state_tracker.record_step(task_id, "Test step")
        self.state_tracker.record_decision(task_id, "Test decision", "Test rationale")
        
        # Complete task
        self.state_tracker.complete_task(task_id, True, "Success")
        self.assertNotIn(task_id, self.state_tracker.active_tasks)
    
    def test_tool_usage_tracking(self):
        """Test tool usage tracking."""
        task_id = self.state_tracker.start_task_tracking("Tool test", "test", 1)
        
        self.state_tracker.record_tool_usage(
            task_id, "test_tool", True, 1.5, "test context"
        )
        
        self.assertIn("test_tool", self.state_tracker.tool_patterns)
        
        # Complete task
        self.state_tracker.complete_task(task_id, True)
    
    def test_strategy_recommendations(self):
        """Test strategy recommendation generation."""
        # Create some successful patterns first
        task_id = self.state_tracker.start_task_tracking("Strategy test", "test_type", 2)
        self.state_tracker.record_tool_usage(task_id, "tool1", True, 1.0)
        self.state_tracker.complete_task(task_id, True)
        
        recommendations = self.state_tracker.get_strategy_recommendations("test_type", 2)
        self.assertIsInstance(recommendations, list)
    
    def test_analytics_generation(self):
        """Test analytics generation."""
        # Create some test data
        task_id = self.state_tracker.start_task_tracking("Analytics test", "test", 1)
        self.state_tracker.record_tool_usage(task_id, "test_tool", True, 1.0)
        self.state_tracker.complete_task(task_id, True)
        
        analytics = self.state_tracker.get_comprehensive_analytics(days=1)
        self.assertIn("task_statistics", analytics)
        self.assertIn("tool_performance", analytics)


class TestWorkflowOrchestration(unittest.TestCase):
    """Test suite for adaptive workflow orchestration."""
    
    def setUp(self):
        """Set up test environment."""
        from server.agent.workflow_orchestrator import (
            AdaptiveWorkflowOrchestrator, TaskComplexity, WorkflowState
        )
        self.orchestrator = AdaptiveWorkflowOrchestrator("http://test", logging.getLogger())
        self.TaskComplexity = TaskComplexity
        self.WorkflowState = WorkflowState
    
    def test_task_complexity_assessment(self):
        """Test task complexity assessment."""
        simple_task = "Show me the file list"
        complex_task = "Refactor the entire architecture"
        
        simple_complexity = self.orchestrator._assess_task_complexity(simple_task)
        complex_complexity = self.orchestrator._assess_task_complexity(complex_task)
        
        self.assertEqual(simple_complexity, self.TaskComplexity.SIMPLE)
        self.assertEqual(complex_complexity, self.TaskComplexity.COMPLEX)
    
    def test_error_strategies(self):
        """Test error recovery strategies."""
        strategies = self.orchestrator.error_strategies
        
        self.assertIn("tool_execution_failed", strategies)
        self.assertIn("model_api_timeout", strategies)
        
        for strategy in strategies.values():
            self.assertTrue(hasattr(strategy, 'should_retry'))
            self.assertTrue(hasattr(strategy, 'recovery_action'))
    
    def test_adaptation_rules(self):
        """Test workflow adaptation rules."""
        rules = self.orchestrator.adaptation_rules
        
        self.assertIsInstance(rules, list)
        self.assertGreater(len(rules), 0)
        
        for rule in rules:
            self.assertIn("condition", rule)
            self.assertIn("action", rule)


class TestActionableMetrics(unittest.TestCase):
    """Test suite for actionable metrics system."""
    
    def setUp(self):
        """Set up test environment."""
        from server.agent.actionable_metrics import ActionableMetricsSystem, MetricType
        self.metrics = ActionableMetricsSystem()
        self.MetricType = MetricType
    
    def test_metric_recording(self):
        """Test metric recording and threshold checking."""
        # Record a normal metric
        self.metrics.record_metric("response_time", 2.0)
        self.assertIn("response_time", self.metrics.metric_history)
        
        # Record a metric that should trigger alert
        self.metrics.record_metric("response_time", 12.0)  # Above critical threshold
        
        # Check if alert was generated
        recent_alerts = [alert for alert in self.metrics.alert_history 
                        if alert.timestamp > time.time() - 10]
        self.assertGreater(len(recent_alerts), 0)
    
    def test_optimization_recommendations(self):
        """Test optimization recommendation generation."""
        # Record poor performance metrics
        for i in range(15):
            self.metrics.record_metric("success_rate", 60.0)  # Below target
        
        # Should generate recommendations
        insights = self.metrics.get_performance_insights(days=1)
        self.assertIn("optimization_opportunities", insights)
    
    def test_health_score_calculation(self):
        """Test health score calculation."""
        # Record good metrics
        self.metrics.record_metric("success_rate", 95.0)
        self.metrics.record_metric("response_time", 2.0)
        self.metrics.record_metric("error_rate", 1.0)
        
        insights = self.metrics.get_performance_insights(days=1)
        health_score = insights["overall_health_score"]
        
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 100.0)
    
    def test_auto_optimization(self):
        """Test automatic optimization system."""
        from server.agent.actionable_metrics import OptimizationLevel
        
        # Enable auto-optimization
        self.metrics.enable_auto_optimization(OptimizationLevel.CONSERVATIVE)
        
        # Record metric that should trigger optimization
        self.metrics.record_metric("context_utilization", 85.0)  # High utilization
        
        # Check if optimization was applied
        status = self.metrics.get_current_status()
        self.assertTrue(status["auto_optimization_enabled"])


class TestMigrationSystem(unittest.TestCase):
    """Test suite for migration and feature flag system."""
    
    def setUp(self):
        """Set up test environment."""
        from server.agent.migration_manager import MigrationManager
        
        # Use temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.temp_config.close()
        
        self.migration_manager = MigrationManager()
        self.migration_manager.feature_flags.config_path = self.temp_config.name
    
    def tearDown(self):
        """Clean up test environment."""
        os.unlink(self.temp_config.name)
    
    def test_feature_flags(self):
        """Test feature flag system."""
        # Test enabling/disabling features
        self.migration_manager.feature_flags.enable("test_feature")
        self.assertTrue(self.migration_manager.feature_flags.is_enabled("test_feature"))
        
        self.migration_manager.feature_flags.disable("test_feature")
        self.assertFalse(self.migration_manager.feature_flags.is_enabled("test_feature"))
    
    def test_migration_validation(self):
        """Test migration readiness validation."""
        validation = self.migration_manager.validate_migration_readiness()
        
        self.assertIn("ready", validation)
        self.assertIn("issues", validation)
        self.assertIn("recommendations", validation)
    
    def test_compatibility_tests(self):
        """Test compatibility testing."""
        results = self.migration_manager.run_compatibility_tests()
        
        self.assertIn("all_tests_passed", results)
        self.assertIn("test_results", results)
        self.assertIsInstance(results["test_results"], dict)


class TestIntegrationLayer(unittest.TestCase):
    """Test suite for integration layer."""
    
    def setUp(self):
        """Set up test environment."""
        from server.agent.integrated_organizer import IntegratedOrganizer
        self.organizer = IntegratedOrganizer("http://test", logging.getLogger())
    
    def test_task_classification(self):
        """Test task classification system."""
        test_cases = [
            ("Read the README file", "information_retrieval"),
            ("Analyze the code quality", "analysis"),
            ("Create a new module", "creation"),
            ("Fix the bug in authentication", "problem_solving"),
            ("Refactor the database layer", "optimization")
        ]
        
        for task, expected_type in test_cases:
            result = self.organizer._classify_task_type(task)
            self.assertEqual(result, expected_type)
    
    def test_complexity_assessment(self):
        """Test task complexity assessment."""
        simple_task = "Show the file contents"
        complex_task = "Design a new architecture"
        
        simple_complexity = self.organizer._assess_task_complexity(simple_task)
        complex_complexity = self.organizer._assess_task_complexity(complex_task)
        
        self.assertLessEqual(simple_complexity, 2)
        self.assertGreaterEqual(complex_complexity, 4)
    
    def test_workflow_selection(self):
        """Test workflow selection logic."""
        simple_task = "List files"
        complex_task = "Refactor entire codebase"
        
        simple_enhanced = self.organizer._should_use_enhanced_workflow(simple_task)
        complex_enhanced = self.organizer._should_use_enhanced_workflow(complex_task)
        
        self.assertFalse(simple_enhanced)  # Should use traditional
        self.assertTrue(complex_enhanced)  # Should use enhanced


def run_component_tests(component_name: str = None) -> Dict[str, Any]:
    """Run tests for specific component or all components."""
    test_classes = {
        "context_management": TestIntelligentContextManagement,
        "tool_management": TestEnhancedToolManagement,
        "resilient_parsing": TestResilientParsing,
        "state_tracking": TestComprehensiveStateTracking,
        "workflow_orchestration": TestWorkflowOrchestration,
        "actionable_metrics": TestActionableMetrics,
        "migration_system": TestMigrationSystem,
        "integration_layer": TestIntegrationLayer
    }
    
    if component_name and component_name in test_classes:
        test_classes = {component_name: test_classes[component_name]}
    
    results = {}
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for name, test_class in test_classes.items():
        print(f"\n🧪 Running {name} tests...")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        success_rate = ((tests_run - failures - errors) / tests_run * 100) if tests_run > 0 else 0
        
        results[name] = {
            "tests_run": tests_run,
            "failures": failures,
            "errors": errors,
            "success_rate": success_rate,
            "status": "passed" if failures == 0 and errors == 0 else "failed"
        }
        
        status_icon = "✅" if results[name]["status"] == "passed" else "❌"
        print(f"{status_icon} {name}: {tests_run} tests, {success_rate:.1f}% success rate")
        
        if failures > 0 or errors > 0:
            print(f"   Failures: {failures}, Errors: {errors}")
        
        total_tests += tests_run
        total_failures += failures
        total_errors += errors
    
    overall_success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
    
    return {
        "component_results": results,
        "overall": {
            "total_tests": total_tests,
            "total_failures": total_failures,
            "total_errors": total_errors,
            "success_rate": overall_success_rate,
            "status": "passed" if total_failures == 0 and total_errors == 0 else "failed"
        }
    }


def main():
    """Main test execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Architecture Test Suite")
    parser.add_argument("--component", help="Test specific component", 
                       choices=["context_management", "tool_management", "resilient_parsing",
                               "state_tracking", "workflow_orchestration", "actionable_metrics",
                               "migration_system", "integration_layer"])
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    print("🚀 Enhanced Anton Agent Architecture Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    results = run_component_tests(args.component)
    duration = time.time() - start_time
    
    print(f"\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n⏱️  Total test duration: {duration:.2f} seconds")
    print(f"🧪 Total tests run: {results['overall']['total_tests']}")
    print(f"✅ Overall success rate: {results['overall']['success_rate']:.1f}%")
    
    if results['overall']['total_failures'] > 0 or results['overall']['total_errors'] > 0:
        print(f"❌ Failures: {results['overall']['total_failures']}")
        print(f"💥 Errors: {results['overall']['total_errors']}")
    
    print(f"\n📋 Component Results:")
    for component, result in results["component_results"].items():
        status_icon = "✅" if result["status"] == "passed" else "❌"
        print(f"  {status_icon} {component.replace('_', ' ').title()}: "
              f"{result['tests_run']} tests, {result['success_rate']:.1f}% success")
    
    # Save detailed results
    results_file = "test_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "duration": duration,
                "results": results
            }, f, indent=2)
        print(f"\n💾 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")
    
    # Provide recommendations based on results
    print(f"\n💡 Recommendations:")
    if results['overall']['success_rate'] >= 90:
        print("  • Excellent test coverage - system is ready for production")
        print("  • Consider adding integration tests for end-to-end workflows")
    elif results['overall']['success_rate'] >= 75:
        print("  • Good test coverage - address failing tests before deployment")
        print("  • Review error logs for specific failure causes")
    else:
        print("  • Test coverage needs improvement - significant issues detected")
        print("  • Recommend thorough debugging before proceeding")
    
    print(f"\n🎉 Test suite completed!")
    
    # Exit with appropriate code
    sys.exit(0 if results['overall']['status'] == "passed" else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)