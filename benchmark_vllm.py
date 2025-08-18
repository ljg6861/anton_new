#!/usr/bin/env python3
"""
vLLM Performance Benchmark

Simple benchmarking script to measure latency and throughput improvements
after migrating from Ollama to vLLM.
"""

import asyncio
import httpx
import time
import statistics
import json
import csv
from typing import List, Dict, Any
from datetime import datetime

class PerformanceBenchmark:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.scenarios = [
            {
                "name": "simple_math",
                "prompt": "What is 15 * 47 + 23?",
                "expected_tokens": 20
            },
            {
                "name": "code_generation",
                "prompt": "Write a Python function to reverse a string and explain how it works",
                "expected_tokens": 150
            },
            {
                "name": "reasoning",
                "prompt": "Compare the advantages and disadvantages of microservices vs monolithic architecture",
                "expected_tokens": 300
            },
            {
                "name": "tool_interaction",
                "prompt": "Check what Python files are in the current directory and summarize their purpose",
                "expected_tokens": 200
            }
        ]

    async def run_benchmark(self, iterations: int = 5) -> Dict[str, Any]:
        """Run benchmark for each scenario"""
        print(f"🚀 Running performance benchmark ({iterations} iterations per scenario)")
        print(f"Target: {self.base_url}")
        print("-" * 60)
        
        results = {}
        
        for scenario in self.scenarios:
            print(f"📊 Benchmarking: {scenario['name']}")
            scenario_results = await self._benchmark_scenario(scenario, iterations)
            results[scenario['name']] = scenario_results
            
            # Print immediate results
            if scenario_results['successful_requests'] > 0:
                print(f"   Mean latency: {scenario_results['latency']['mean']:.2f}s")
                print(f"   P95 latency: {scenario_results['latency']['p95']:.2f}s")
                print(f"   Throughput: {scenario_results['throughput']['tokens_per_sec']:.1f} tok/s")
                print(f"   Success rate: {scenario_results['success_rate']:.1f}%")
            else:
                print(f"   ❌ All requests failed")
            print()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "backend": "vllm",
            "base_url": self.base_url,
            "iterations_per_scenario": iterations,
            "scenarios": results
        }

    async def _benchmark_scenario(self, scenario: Dict[str, Any], iterations: int) -> Dict[str, Any]:
        """Benchmark a single scenario"""
        latencies = []
        token_counts = []
        errors = 0
        
        for i in range(iterations):
            start_time = time.time()
            
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.base_url}/v1/agent/chat",
                        json={
                            "messages": [{"role": "user", "content": scenario["prompt"]}],
                            "temperature": 0.1  # Consistent results
                        }
                    )
                    
                    latency = time.time() - start_time
                    
                    if response.status_code == 200:
                        # Estimate tokens in response
                        response_text = str(response.json())
                        estimated_tokens = len(response_text.split())
                        
                        latencies.append(latency)
                        token_counts.append(estimated_tokens)
                    else:
                        errors += 1
                        
            except Exception as e:
                errors += 1
                print(f"   Request {i+1} failed: {str(e)[:50]}...")
        
        # Calculate statistics
        if latencies:
            mean_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0]
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0]
            
            mean_tokens = statistics.mean(token_counts)
            tokens_per_sec = mean_tokens / mean_latency if mean_latency > 0 else 0
            
            return {
                "successful_requests": len(latencies),
                "failed_requests": errors,
                "success_rate": (len(latencies) / iterations) * 100,
                "latency": {
                    "mean": mean_latency,
                    "median": median_latency,
                    "p95": p95_latency,
                    "p99": p99_latency,
                    "min": min(latencies),
                    "max": max(latencies)
                },
                "throughput": {
                    "mean_tokens": mean_tokens,
                    "tokens_per_sec": tokens_per_sec
                },
                "raw_latencies": latencies,
                "raw_token_counts": token_counts
            }
        else:
            return {
                "successful_requests": 0,
                "failed_requests": errors,
                "success_rate": 0,
                "error": "All requests failed"
            }

    def save_results(self, results: Dict[str, Any], format: str = "json"):
        """Save benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filename = f"benchmark_results_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            print(f"📁 Results saved to {filename}")
            
        elif format == "csv":
            filename = f"benchmark_results_{timestamp}.csv"
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    "scenario", "backend", "mean_latency", "p95_latency", "p99_latency",
                    "tokens_per_sec", "success_rate", "successful_requests"
                ])
                
                # Data rows
                for scenario_name, data in results["scenarios"].items():
                    if "error" not in data:
                        writer.writerow([
                            scenario_name,
                            results["backend"],
                            data["latency"]["mean"],
                            data["latency"]["p95"],
                            data["latency"]["p99"],
                            data["throughput"]["tokens_per_sec"],
                            data["success_rate"],
                            data["successful_requests"]
                        ])
            
            print(f"📊 CSV results saved to {filename}")

    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Backend: {results['backend']}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Iterations per scenario: {results['iterations_per_scenario']}")
        print()
        
        # Calculate overall statistics
        all_successful = sum(data.get("successful_requests", 0) for data in results["scenarios"].values())
        all_failed = sum(data.get("failed_requests", 0) for data in results["scenarios"].values())
        overall_success_rate = (all_successful / (all_successful + all_failed)) * 100 if (all_successful + all_failed) > 0 else 0
        
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        print(f"Total Successful Requests: {all_successful}")
        print(f"Total Failed Requests: {all_failed}")
        print()
        
        # Scenario breakdown
        print("Scenario Details:")
        print("-" * 40)
        
        for scenario_name, data in results["scenarios"].items():
            if "error" not in data:
                print(f"{scenario_name}:")
                print(f"  Mean Latency: {data['latency']['mean']:.2f}s")
                print(f"  P95 Latency: {data['latency']['p95']:.2f}s")
                print(f"  Throughput: {data['throughput']['tokens_per_sec']:.1f} tokens/sec")
                print(f"  Success Rate: {data['success_rate']:.1f}%")
                
                # Performance assessment
                if data['latency']['mean'] < 3.0:
                    print(f"  Assessment: ✅ Excellent latency")
                elif data['latency']['mean'] < 5.0:
                    print(f"  Assessment: ✅ Good latency")
                elif data['latency']['mean'] < 10.0:
                    print(f"  Assessment: ⚠️  Acceptable latency")
                else:
                    print(f"  Assessment: ❌ High latency")
                print()
            else:
                print(f"{scenario_name}: ❌ Failed - {data['error']}")
                print()
        
        # Performance goals assessment
        print("Performance Goals Assessment:")
        print("-" * 30)
        
        # Calculate average metrics across successful scenarios
        successful_scenarios = [data for data in results["scenarios"].values() if "error" not in data and data["successful_requests"] > 0]
        
        if successful_scenarios:
            avg_latency = statistics.mean([data["latency"]["mean"] for data in successful_scenarios])
            avg_p95 = statistics.mean([data["latency"]["p95"] for data in successful_scenarios])
            avg_throughput = statistics.mean([data["throughput"]["tokens_per_sec"] for data in successful_scenarios])
            
            # Goal: Mean latency < 5s
            latency_goal = avg_latency < 5.0
            print(f"Mean Latency < 5s: {'✅ PASS' if latency_goal else '❌ FAIL'} ({avg_latency:.2f}s)")
            
            # Goal: P95 latency < 10s
            p95_goal = avg_p95 < 10.0
            print(f"P95 Latency < 10s: {'✅ PASS' if p95_goal else '❌ FAIL'} ({avg_p95:.2f}s)")
            
            # Goal: Throughput > 20 tokens/sec
            throughput_goal = avg_throughput > 20.0
            print(f"Throughput > 20 tok/s: {'✅ PASS' if throughput_goal else '❌ FAIL'} ({avg_throughput:.1f} tok/s)")
            
            # Goal: Success rate > 90%
            success_goal = overall_success_rate > 90.0
            print(f"Success Rate > 90%: {'✅ PASS' if success_goal else '❌ FAIL'} ({overall_success_rate:.1f}%)")
            
            # Overall assessment
            passed_goals = sum([latency_goal, p95_goal, throughput_goal, success_goal])
            print(f"\nOverall: {passed_goals}/4 goals met")
            
            if passed_goals >= 3:
                print("🎉 vLLM migration is performing well!")
            elif passed_goals >= 2:
                print("⚠️  vLLM migration needs some tuning")
            else:
                print("🚨 vLLM migration has performance issues")
        else:
            print("❌ No successful scenarios to assess")

async def main():
    """Main benchmark runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Performance Benchmark")
    parser.add_argument("--url", default="http://localhost:8001", help="Base URL for Anton API")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per scenario")
    parser.add_argument("--output", choices=["json", "csv", "both"], default="json", help="Output format")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark(args.url)
    
    try:
        results = await benchmark.run_benchmark(args.iterations)
        benchmark.print_summary(results)
        
        # Save results if requested
        if args.save:
            if args.output in ["json", "both"]:
                benchmark.save_results(results, "json")
            if args.output in ["csv", "both"]:
                benchmark.save_results(results, "csv")
                
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user")
    except Exception as e:
        print(f"\n💥 Benchmark failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
