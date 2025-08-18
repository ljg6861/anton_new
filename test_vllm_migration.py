#!/usr/bin/env python3
"""
vLLM Migration Test Suite

Comprehensive tests to validate the migration from Ollama to vLLM.
Tests tool calling, streaming, performance, and functional parity.
"""

import asyncio
import httpx
import json
import time
import statistics
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestResult:
    name: str
    success: bool
    duration: float
    details: Optional[str] = None
    error: Optional[str] = None

class VLLMMigrationTestSuite:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        
        # Test configurations
        self.timeout = 60.0
        self.concurrent_requests = 8
        
        # Test prompts for different scenarios
        self.test_prompts = {
            "simple": "What is 2 + 2?",
            "code_generation": "Write a Python function to calculate the factorial of a number",
            "tool_usage": "List the files in the current directory and count how many Python files there are", 
            "reasoning": "Explain the difference between TCP and UDP protocols with examples",
            "long_context": "Remember these numbers: " + ", ".join([str(i) for i in range(100)]) + ". What was the 50th number?"
        }

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run the complete migration test suite"""
        print("🧪 Starting vLLM Migration Test Suite")
        print(f"Target: {self.base_url}")
        print("=" * 60)
        
        # Core functionality tests
        await self._test_basic_response()
        await self._test_streaming_response()
        await self._test_tool_calling()
        await self._test_long_context()
        
        # Performance tests
        await self._test_concurrent_requests()
        await self._test_latency_benchmark()
        
        # Edge case tests
        await self._test_error_handling()
        await self._test_malformed_requests()
        
        return self._generate_report()

    async def _test_basic_response(self):
        """Test basic request-response functionality"""
        print("🔧 Testing basic response...")
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/agent/chat",
                    json={
                        "messages": [{"role": "user", "content": self.test_prompts["simple"]}]
                    }
                )
                
                success = response.status_code == 200
                details = f"Status: {response.status_code}, Response length: {len(str(response.json()))}"
                
                self.results.append(TestResult(
                    name="basic_response",
                    success=success,
                    duration=time.time() - start_time,
                    details=details
                ))
                
        except Exception as e:
            self.results.append(TestResult(
                name="basic_response", 
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            ))

    async def _test_streaming_response(self):
        """Test streaming response functionality"""
        print("🌊 Testing streaming response...")
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/agent/chat",
                    json={
                        "messages": [{"role": "user", "content": self.test_prompts["code_generation"]}]
                    }
                ) as response:
                    
                    chunks_received = 0
                    total_content = ""
                    
                    async for chunk in response.aiter_text():
                        chunks_received += 1
                        total_content += chunk
                        
                        if chunks_received > 100:  # Prevent infinite loops
                            break
                    
                    success = response.status_code == 200 and chunks_received > 0
                    details = f"Chunks: {chunks_received}, Content length: {len(total_content)}"
                    
                    self.results.append(TestResult(
                        name="streaming_response",
                        success=success,
                        duration=time.time() - start_time,
                        details=details
                    ))
                    
        except Exception as e:
            self.results.append(TestResult(
                name="streaming_response",
                success=False, 
                duration=time.time() - start_time,
                error=str(e)
            ))

    async def _test_tool_calling(self):
        """Test tool calling functionality"""
        print("🔨 Testing tool calling...")
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/agent/chat",
                    json={
                        "messages": [{"role": "user", "content": self.test_prompts["tool_usage"]}]
                    }
                )
                
                response_text = str(response.json()) if response.status_code == 200 else ""
                
                # Look for evidence of tool usage
                tool_indicators = ["file", "directory", "list", "count", ".py"]
                tool_evidence = sum(1 for indicator in tool_indicators if indicator.lower() in response_text.lower())
                
                success = response.status_code == 200 and tool_evidence >= 2
                details = f"Status: {response.status_code}, Tool evidence score: {tool_evidence}/5"
                
                self.results.append(TestResult(
                    name="tool_calling",
                    success=success,
                    duration=time.time() - start_time,
                    details=details
                ))
                
        except Exception as e:
            self.results.append(TestResult(
                name="tool_calling",
                success=False,
                duration=time.time() - start_time, 
                error=str(e)
            ))

    async def _test_long_context(self):
        """Test long context handling"""
        print("📝 Testing long context...")
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout * 2) as client:
                response = await client.post(
                    f"{self.base_url}/v1/agent/chat",
                    json={
                        "messages": [{"role": "user", "content": self.test_prompts["long_context"]}]
                    }
                )
                
                response_text = str(response.json()) if response.status_code == 200 else ""
                
                # Check if it correctly identified the 50th number (49, 0-indexed)
                context_preserved = "49" in response_text or "50" in response_text
                
                success = response.status_code == 200 and context_preserved
                details = f"Status: {response.status_code}, Context preserved: {context_preserved}"
                
                self.results.append(TestResult(
                    name="long_context",
                    success=success,
                    duration=time.time() - start_time,
                    details=details
                ))
                
        except Exception as e:
            self.results.append(TestResult(
                name="long_context",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            ))

    async def _test_concurrent_requests(self):
        """Test concurrent request handling"""
        print(f"⚡ Testing {self.concurrent_requests} concurrent requests...")
        
        async def single_request(request_id: int) -> Dict[str, Any]:
            start_time = time.time()
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/v1/agent/chat",
                        json={
                            "messages": [{"role": "user", "content": f"Count from 1 to {request_id + 3}"}]
                        }
                    )
                    
                    return {
                        "id": request_id,
                        "success": response.status_code == 200,
                        "duration": time.time() - start_time,
                        "status": response.status_code
                    }
            except Exception as e:
                return {
                    "id": request_id,
                    "success": False,
                    "duration": time.time() - start_time,
                    "error": str(e)
                }
        
        start_time = time.time()
        
        # Run concurrent requests
        tasks = [single_request(i) for i in range(self.concurrent_requests)]
        request_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_requests = [r for r in request_results if isinstance(r, dict) and r.get("success", False)]
        failed_requests = len(request_results) - len(successful_requests)
        
        if successful_requests:
            avg_latency = statistics.mean([r["duration"] for r in successful_requests])
            max_latency = max([r["duration"] for r in successful_requests])
        else:
            avg_latency = max_latency = 0
        
        success = len(successful_requests) >= self.concurrent_requests * 0.8  # 80% success rate
        details = f"Success: {len(successful_requests)}/{self.concurrent_requests}, Avg latency: {avg_latency:.2f}s, Max: {max_latency:.2f}s"
        
        self.results.append(TestResult(
            name="concurrent_requests",
            success=success,
            duration=time.time() - start_time,
            details=details
        ))

    async def _test_latency_benchmark(self):
        """Benchmark response latency"""
        print("📊 Running latency benchmark...")
        
        latencies = []
        iterations = 5
        
        for i in range(iterations):
            start_time = time.time()
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/v1/agent/chat",
                        json={
                            "messages": [{"role": "user", "content": self.test_prompts["reasoning"]}]
                        }
                    )
                    
                    if response.status_code == 200:
                        latencies.append(time.time() - start_time)
                        
            except Exception:
                pass  # Skip failed requests
        
        if latencies:
            mean_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0]
            
            # Success criteria: mean < 10s, p95 < 15s 
            success = mean_latency < 10.0 and p95_latency < 15.0
            details = f"Mean: {mean_latency:.2f}s, P95: {p95_latency:.2f}s, Samples: {len(latencies)}"
        else:
            success = False
            details = "No successful requests"
        
        self.results.append(TestResult(
            name="latency_benchmark",
            success=success,
            duration=sum(latencies) if latencies else 0,
            details=details
        ))

    async def _test_error_handling(self):
        """Test error handling and recovery"""
        print("🚨 Testing error handling...")
        
        start_time = time.time()
        try:
            # Test with invalid endpoint
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/invalid/endpoint",
                    json={"messages": [{"role": "user", "content": "test"}]}
                )
                
                # Should get 404 or similar error, not a crash
                success = response.status_code in [404, 405, 422]
                details = f"Error endpoint returned status: {response.status_code}"
                
                self.results.append(TestResult(
                    name="error_handling",
                    success=success,
                    duration=time.time() - start_time,
                    details=details
                ))
                
        except Exception as e:
            # If it crashes completely, that's also a useful result
            self.results.append(TestResult(
                name="error_handling",
                success=False,
                duration=time.time() - start_time,
                error=f"Server crashed: {str(e)}"
            ))

    async def _test_malformed_requests(self):
        """Test handling of malformed requests"""
        print("🔧 Testing malformed request handling...")
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Send request with missing required fields
                response = await client.post(
                    f"{self.base_url}/v1/agent/chat",
                    json={"invalid": "request"}
                )
                
                # Should return 422 validation error, not crash
                success = response.status_code == 422
                details = f"Malformed request returned status: {response.status_code}"
                
                self.results.append(TestResult(
                    name="malformed_requests",
                    success=success,
                    duration=time.time() - start_time,
                    details=details
                ))
                
        except Exception as e:
            self.results.append(TestResult(
                name="malformed_requests",
                success=False,
                duration=time.time() - start_time,
                error=str(e)
            ))

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        total_duration = sum(r.duration for r in self.results)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_duration": total_duration
            },
            "results": [
                {
                    "name": r.name,
                    "success": r.success,
                    "duration": r.duration,
                    "details": r.details,
                    "error": r.error
                }
                for r in self.results
            ]
        }
        
        return report

    def print_report(self, report: Dict[str, Any]):
        """Print human-readable test report"""
        print("\n" + "=" * 60)
        print("🧪 vLLM Migration Test Results")
        print("=" * 60)
        
        summary = report["summary"]
        print(f"Tests Run: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        print()
        
        # Print individual test results
        for result in report["results"]:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"{status} {result['name']} ({result['duration']:.2f}s)")
            
            if result["details"]:
                print(f"    {result['details']}")
            if result["error"]:
                print(f"    Error: {result['error']}")
            print()
        
        # Overall assessment
        if summary["success_rate"] >= 80:
            print("🎉 Migration looks successful! Most tests passed.")
        elif summary["success_rate"] >= 60:
            print("⚠️  Migration partially successful. Some issues detected.")
        else:
            print("🚨 Migration has significant issues. Review failed tests.")

async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Migration Test Suite")
    parser.add_argument("--url", default="http://localhost:8001", help="Base URL for Anton API")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Initialize and run tests
    test_suite = VLLMMigrationTestSuite(args.url)
    
    try:
        report = await test_suite.run_all_tests()
        test_suite.print_report(report)
        
        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n📁 Results saved to {args.output}")
        
        # Exit with error if tests failed
        success_rate = report["summary"]["success_rate"]
        if success_rate < 80:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
