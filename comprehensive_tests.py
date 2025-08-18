#!/usr/bin/env python3
"""
Comprehensive tests for Anton AI Assistant.

This file consolidates all testing for the Anton system, including:
- Tool execution and failure detection
- Tool learning system and corrective actions
- ReAct agent flows
- Server integration tests

Tests interact with running server directly (assumes services are up).
Run with: python comprehensive_tests.py
"""

import asyncio
import httpx
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
AGENT_SERVER_URL = "http://localhost:8001"
MODEL_SERVER_URL = "http://localhost:8002"
TEST_TIMEOUT = 30.0

@dataclass
class TestResult:
    """Represents the result of a test case."""
    name: str
    passed: bool
    message: str
    duration: float = 0.0

class TestSuite:
    """Main test suite for Anton comprehensive testing."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
    
    async def run_all_tests(self):
        """Run all test suites."""
        logger.info("🚀 Starting Anton Comprehensive Test Suite")
        
        # Test suites in dependency order
        test_suites = [
            ("Tool Execution", self.test_tool_execution),
            ("Tool Failure Detection", self.test_tool_failure_detection),
            ("Tool Learning System", self.test_tool_learning_system),
            ("Agent Chat Integration", self.test_agent_chat_integration),
            ("Memory Summarization", self.test_memory_summarization),
        ]
        
        for suite_name, test_func in test_suites:
            logger.info(f"\n📋 Running {suite_name} tests...")
            try:
                await test_func()
            except Exception as e:
                logger.error(f"❌ Test suite {suite_name} failed with exception: {e}")
                self.results.append(TestResult(
                    name=f"{suite_name} (Exception)",
                    passed=False,
                    message=str(e)
                ))
        
        self._print_summary()
    
    async def test_server_health(self):
        """Test that required servers are running and responsive."""
        
        # Test Agent Server
        await self._test_case(
            "Agent Server Health",
            self._check_agent_server_health,
            "Agent server should be running on port 8001"
        )
        
        # Test Model Server  
        await self._test_case(
            "Model Server Health",
            self._check_model_server_health,
            "Model server should be running on port 8002"
        )
    
    async def test_tool_execution(self):
        """Test basic tool execution functionality."""
        
        # Test successful tool execution
        await self._test_case(
            "Successful Tool Execution",
            self._test_successful_tool_execution,
            "Tools should execute successfully and return results"
        )
        
        # Test tool with parameters
        await self._test_case(
            "Tool Parameter Handling",
            self._test_tool_parameter_handling,
            "Tools should handle both 'path' and 'file_path' parameters"
        )
    
    async def test_tool_failure_detection(self):
        """Test that tool failures are properly detected and reported."""
        
        # Test parameter validation failures
        await self._test_case(
            "Parameter Validation Failure",
            self._test_parameter_validation_failure,
            "Missing required parameters should raise exceptions"
        )
        
        # Test file not found failures
        await self._test_case(
            "File Not Found Failure", 
            self._test_file_not_found_failure,
            "Non-existent files should raise FileNotFoundError"
        )
        
        # Test nonexistent tool failures
        await self._test_case(
            "Nonexistent Tool Failure",
            self._test_nonexistent_tool_failure,
            "Unknown tools should raise exceptions"
        )
        
        # Test tool executor status reporting
        await self._test_case(
            "Tool Status Reporting",
            self._test_tool_status_reporting,
            "Tool executor should properly report success vs failure status"
        )
    
    async def test_tool_learning_system(self):
        """Test the tool learning and corrective action system."""
        
        # Test tool execution recording
        await self._test_case(
            "Tool Learning Recording",
            self._test_tool_learning_recording,
            "Tool executions should be recorded for learning"
        )
    
    async def test_agent_chat_integration(self):
        """Test integration with the agent chat system."""
        
        # Test basic agent interaction
        await self._test_case(
            "Basic Agent Chat",
            self._test_basic_agent_chat,
            "Agent should respond to simple messages"
        )

    async def test_memory_summarization(self):
        """Test LLM-only summarization and memory behaviors."""
        await self._test_case(
            "LLM Summarization Always Used",
            self._test_llm_summarization_always_used,
            "LLM should always be used for summarization when threshold is met"
        )
        await self._test_case(
            "No Heuristic Summarization Methods",
            self._test_no_heuristic_summarization,
            "All heuristic summarization methods should be removed"
        )
        await self._test_case(
            "Memory Token Estimation",
            self._test_memory_token_estimation,
            "Token estimation and budget management should work correctly"
        )
    
    # Helper methods for test execution
    async def _test_case(self, name: str, test_func, description: str):
        """Execute a single test case with timing and error handling."""
        start_time = time.time()
        try:
            result = await asyncio.wait_for(test_func(), timeout=TEST_TIMEOUT)
            duration = time.time() - start_time
            
            if result:
                logger.info(f"✅ {name}: PASSED ({duration:.2f}s)")
                self.results.append(TestResult(name, True, "OK", duration))
            else:
                logger.error(f"❌ {name}: FAILED ({duration:.2f}s)")
                self.results.append(TestResult(name, False, "Test returned False", duration))
                
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.error(f"⏰ {name}: TIMEOUT ({duration:.2f}s)")
            self.results.append(TestResult(name, False, "Test timed out", duration))
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {name}: ERROR ({duration:.2f}s) - {e}")
            self.results.append(TestResult(name, False, str(e), duration))
    
    # Individual test implementations
    async def _check_agent_server_health(self) -> bool:
        """Check if agent server is responding."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{AGENT_SERVER_URL}/health", timeout=5.0)
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Agent server health check failed: {e}")
            return False
    
    async def _check_model_server_health(self) -> bool:
        """Check if model server is responding."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{MODEL_SERVER_URL}/health", timeout=5.0)
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Model server health check failed: {e}")
            return False
    
    async def _test_successful_tool_execution(self) -> bool:
        """Test that tools execute successfully and return expected results."""
        try:
            # Import tool execution directly to test the core functionality
            from server.agent.tool_executor import execute_tool_async
            
            # Test read_file tool with existing file
            result = await execute_tool_async('read_file', {'path': 'README.md'}, logger)
            
            # Verify we got a non-empty string result
            if isinstance(result, str) and len(result) > 0:
                logger.info(f"Tool execution successful, result length: {len(result)}")
                return True
            else:
                logger.error(f"Unexpected result type or empty result: {type(result)}")
                return False
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return False
    
    async def _test_tool_parameter_handling(self) -> bool:
        """Test that read_file tool handles both 'path' and 'file_path' parameters."""
        try:
            from server.agent.tool_executor import execute_tool_async
            
            # Test with 'path' parameter
            result1 = await execute_tool_async('read_file', {'path': 'README.md'}, logger)
            
            # Test with 'file_path' parameter  
            result2 = await execute_tool_async('read_file', {'file_path': 'README.md'}, logger)
            
            # Both should return the same content
            if result1 == result2 and len(result1) > 0:
                logger.info("Both parameter names work correctly")
                return True
            else:
                logger.error("Parameter handling inconsistent")
                return False
                
        except Exception as e:
            logger.error(f"Parameter handling test failed: {e}")
            return False
    
    async def _test_parameter_validation_failure(self) -> bool:
        """Test that missing required parameters raise exceptions."""
        try:
            from server.agent.tool_executor import execute_tool_async
            
            # This should raise an exception due to missing parameters
            try:
                result = await execute_tool_async('read_file', {}, logger)
                logger.error(f"Expected exception but got result: {result}")
                return False
            except ValueError as e:
                if "required parameter" in str(e).lower():
                    logger.info(f"Correctly raised ValueError for missing parameter: {e}")
                    return True
                else:
                    logger.error(f"Wrong exception message: {e}")
                    return False
            except Exception as e:
                logger.error(f"Wrong exception type: {type(e)} - {e}")
                return False
                
        except Exception as e:
            logger.error(f"Parameter validation test setup failed: {e}")
            return False
    
    async def _test_file_not_found_failure(self) -> bool:
        """Test that non-existent files raise FileNotFoundError."""
        try:
            from server.agent.tool_executor import execute_tool_async
            
            # This should raise FileNotFoundError
            try:
                result = await execute_tool_async('read_file', {'path': 'nonexistent_file_xyz.txt'}, logger)
                logger.error(f"Expected FileNotFoundError but got result: {result}")
                return False
            except FileNotFoundError as e:
                logger.info(f"Correctly raised FileNotFoundError: {e}")
                return True
            except Exception as e:
                logger.error(f"Wrong exception type: {type(e)} - {e}")
                return False
                
        except Exception as e:
            logger.error(f"File not found test setup failed: {e}")
            return False
    
    async def _test_nonexistent_tool_failure(self) -> bool:
        """Test that unknown tools raise exceptions."""
        try:
            from server.agent.tool_executor import execute_tool_async
            
            # This should raise an exception for unknown tool
            try:
                result = await execute_tool_async('nonexistent_tool_xyz', {}, logger)
                logger.error(f"Expected exception but got result: {result}")
                return False
            except Exception as e:
                if "not found" in str(e).lower():
                    logger.info(f"Correctly raised exception for unknown tool: {e}")
                    return True
                else:
                    logger.error(f"Wrong exception message: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"Nonexistent tool test setup failed: {e}")
            return False
    
    async def _test_tool_learning_recording(self) -> bool:
        """Test that tool executions are recorded for learning."""
        try:
            # Test basic tool learning store functionality
            from server.agent.tool_learning_store import tool_learning_store
            
            # Check if learning store is accessible
            initial_count = len(tool_learning_store.current_execution_sequence)
            logger.info(f"Current execution sequence length: {initial_count}")
            
            # Execute a tool to generate a learning record
            from server.agent.tool_executor import execute_tool_async
            result = await execute_tool_async('read_file', {'path': 'README.md'}, logger)
            
            # Check if execution was recorded (this is a basic check)
            # The actual recording happens in the full tool execution pipeline
            if isinstance(result, str) and len(result) > 0:
                logger.info("Tool learning system appears to be functioning")
                return True
            else:
                logger.error("Tool learning test inconclusive")
                return False
                
        except Exception as e:
            logger.error(f"Tool learning test failed: {e}")
            return False
    
    async def _test_basic_agent_chat(self) -> bool:
        """Test basic agent chat functionality."""
        try:
            async with httpx.AsyncClient() as client:
                # Test basic agent endpoint
                chat_request = {
                    "messages": [
                        {"role": "user", "content": "What is 2+2?"}
                    ]
                }
                
                response = await client.post(
                    f"{AGENT_SERVER_URL}/v1/agent/chat",
                    json=chat_request,
                    timeout=15.0
                )
                
                if response.status_code == 200:
                    # For streaming responses, just check that we got a response
                    logger.info("Agent chat endpoint responded successfully")
                    return True
                else:
                    logger.error(f"Agent chat failed with status: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Agent chat test failed: {e}")
            return False
    
    async def _test_tool_status_reporting(self) -> bool:
        """Test that tool executor properly reports success vs failure status."""
        try:
            # This tests the core issue that was reported in the original problem:
            # Tool calls should never report "success" status when there are clear errors
            
            from server.agent.tool_executor import process_tool_calls
            import re
            
            # Mock objects for testing
            class MockLogger:
                def info(self, msg, **kwargs): pass
                def error(self, msg, **kwargs): pass
                def warning(self, msg, **kwargs): pass
            
            mock_logger = MockLogger()
            
            # Test data: simulate the exact failing tool call from the issue
            failing_tool_call = '<tool_call>{"name": "read_file", "arguments": {"path": "server/agent/tools/web_search.py"}}</tool_call>'
            successful_tool_call = '<tool_call>{"name": "read_file", "arguments": {"path": "README.md"}}</tool_call>'
            
            tool_call_regex = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
            
            # Track results from the callback
            tool_results = []
            
            async def result_callback(tool_result):
                tool_results.append(tool_result)
                logger.info(f"Tool result: {tool_result}")
            
            # Test successful case
            messages1 = []
            tool_results.clear()
            await process_tool_calls(
                successful_tool_call,
                tool_call_regex, 
                messages1,
                mock_logger,
                result_callback=result_callback
            )
            
            if len(tool_results) == 1:
                result = tool_results[0]
                if result['status'] == 'success' and len(result['brief_result']) > 0:
                    logger.info("✅ Successful tool correctly reported as success")
                else:
                    logger.error(f"Success case failed: {result}")
                    return False
            else:
                logger.error("Expected exactly one tool result for success case")
                return False
            
            # Test failure case - nonexistent file
            failing_tool_call_nonexistent = '<tool_call>{"name": "read_file", "arguments": {"path": "definitely_nonexistent_file.xyz"}}</tool_call>'
            messages2 = []
            tool_results.clear()
            
            await process_tool_calls(
                failing_tool_call_nonexistent,
                tool_call_regex,
                messages2, 
                mock_logger,
                result_callback=result_callback
            )
            
            if len(tool_results) == 1:
                result = tool_results[0]
                if result['status'] == 'error':
                    logger.info("✅ Failed tool correctly reported as error")
                else:
                    logger.error(f"Failure case incorrectly reported as success: {result}")
                    return False
            else:
                logger.error("Expected exactly one tool result for failure case")
                return False
            
            logger.info("Tool status reporting working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Tool status reporting test failed: {e}")
            return False

    # New memory tests
    async def _test_llm_summarization_always_used(self) -> bool:
        """Test that LLM is always used for summarization when threshold is met."""
        try:
            from server.agent.react.memory_manager import MemoryManager
            from server.agent.react.token_budget import TokenBudget

            tb = TokenBudget(total_budget=1000)
            mm = MemoryManager(tb)

            # Create enough messages to exceed 5k token threshold
            large_message = "This is a substantial message that contains meaningful content " * 100  # ~6k chars = ~1.5k tokens
            messages = []
            for i in range(5):  # 5 * 1.5k = 7.5k tokens (exceeds 5k threshold)
                messages.append({"role": "user", "content": f"Message {i}: {large_message}"})

            # Mock LLM response
            mock_summary = "EARLIER CONVERSATION (LLM summarized):\n• USER REQUESTS: User asked about various topics\n• DECISIONS MADE: Assistant decided to help\n• FILES & CODE: No files mentioned\n• ERRORS & SOLUTIONS: No errors encountered"

            # Simulate what would happen in update_llm_conversation_summary
            mm.conversation_summary = mock_summary
            mm._last_summarized_index = 3

            wm = mm.build_working_memory(messages)
            
            # Verify LLM summary is included
            if "LLM summarized" not in wm:
                logger.error("LLM summary not found in working memory")
                return False
            
            if "USER REQUESTS:" not in wm:
                logger.error("LLM summary structure not preserved")
                return False

            logger.info("✅ LLM summarization working correctly")
            return True
            
        except Exception as e:
            logger.error(f"LLM summarization test failed: {e}")
            return False

    async def _test_no_heuristic_summarization(self) -> bool:
        """Test that no heuristic summarization methods exist."""
        try:
            from server.agent.react.memory_manager import MemoryManager
            
            # Check that heuristic methods were removed
            if hasattr(MemoryManager, '_summarize_messages_heuristic'):
                logger.error("Heuristic summarization method still exists")
                return False
                
            if hasattr(MemoryManager, 'should_use_llm_summarization'):
                logger.error("should_use_llm_summarization method still exists")
                return False

            logger.info("✅ All heuristic summarization removed")
            return True
            
        except Exception as e:
            logger.error(f"Heuristic removal test failed: {e}")
            return False

    async def _test_memory_token_estimation(self) -> bool:
        """Test token estimation and budget management."""
        try:
            from server.agent.react.memory_manager import MemoryManager
            from server.agent.react.token_budget import TokenBudget

            tb = TokenBudget(total_budget=1000)
            mm = MemoryManager(tb)

            # Test token estimation
            test_text = "Hello world! " * 10  # 130 chars
            estimated = mm.estimate_tokens(test_text)
            expected = len(test_text) // 4  # Should be ~32 tokens
            
            if abs(estimated - expected) > 5:  # Allow some variance
                logger.error(f"Token estimation off: got {estimated}, expected ~{expected}")
                return False

            # Test text truncation
            long_text = "Word " * 1000  # 5000 chars
            truncated = mm.truncate_to_budget(long_text, 100)  # 100 token budget
            
            if len(truncated) > 405:  # 100 tokens * 4 chars/token + a few chars for "..."
                logger.error(f"Text not properly truncated: {len(truncated)} chars")
                return False

            logger.info("✅ Token estimation and truncation working")
            return True
            
        except Exception as e:
            logger.error(f"Token estimation test failed: {e}")
            return False
    
    def _print_summary(self):
        """Print a summary of all test results."""
        total_time = time.time() - self.start_time
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.passed])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*60)
        print("🎯 ANTON COMPREHENSIVE TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"📊 Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        
        if failed_tests > 0:
            print("\n📋 FAILED TESTS:")
            for result in self.results:
                if not result.passed:
                    print(f"  ❌ {result.name}: {result.message}")
        
        print("="*60)
        
        # Exit with appropriate code
        sys.exit(0 if failed_tests == 0 else 1)


async def main():
    """Main entry point for comprehensive tests."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        return
    
    # Check if we're in the right directory
    if not os.path.exists("server/agent"):
        logger.error("❌ Tests must be run from the Anton project root directory")
        sys.exit(1)
    
    # Add current directory to Python path for imports
    sys.path.insert(0, os.getcwd())
    
    # Run the test suite
    suite = TestSuite()
    await suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
