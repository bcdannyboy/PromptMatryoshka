"""Comprehensive performance and reliability integration tests.

Tests performance characteristics and reliability of the multi-provider system:
- Performance benchmarking across all 4 providers
- Load testing and stress testing scenarios  
- Memory usage patterns and leak detection
- Concurrent execution performance and thread safety
- Caching effectiveness and cache invalidation
- Scalability testing with increasing workloads
- Resource consumption monitoring and limits
- Performance regression detection
- Reliability under sustained high load
- Response time consistency and latency analysis
- Throughput measurements and bottleneck identification
- Provider switching performance impact

This test suite ensures the system performs well under various conditions
and maintains reliability characteristics expected for production use.
"""

import pytest
import os
import tempfile
import json
import shutil
import time
import threading
import psutil
import gc
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional, Tuple
import concurrent.futures
from datetime import datetime, timedelta
import statistics
import weakref
from memory_profiler import profile
import cProfile
import pstats
from io import StringIO

from promptmatryoshka.core import PromptMatryoshka
from promptmatryoshka.config import Config, get_config, reset_config
from promptmatryoshka.llm_factory import LLMFactory, get_factory
from promptmatryoshka.llm_interface import LLMInterface, LLMConfig, ProviderInfo
from promptmatryoshka.providers import clear_registry
from promptmatryoshka.plugins.base import PluginBase, get_plugin_registry
from promptmatryoshka.plugins.boost import BoostPlugin
from promptmatryoshka.plugins.flipattack import FlipAttackPlugin
from promptmatryoshka.exceptions import LLMError


class PerformanceLLMInterface(LLMInterface):
    """Mock LLM interface for performance testing with realistic delays."""
    
    def __init__(self, config, provider_name="perf_mock", latency_ms=50, **kwargs):
        super().__init__(config, provider_name=provider_name, **kwargs)
        self.provider_name = provider_name
        self.latency_ms = latency_ms
        self.call_count = 0
        self.total_processing_time = 0
        self.memory_usage = []
        self.start_time = time.time()
        
    def invoke(self, input, config=None, **kwargs):
        self.call_count += 1
        start_time = time.time()
        
        # Simulate realistic API latency
        time.sleep(self.latency_ms / 1000.0)
        
        # Simulate processing work
        processing_delay = len(input) * 0.0001  # Scale with input size
        time.sleep(processing_delay)
        
        # Track memory usage
        process = psutil.Process()
        self.memory_usage.append(process.memory_info().rss)
        
        end_time = time.time()
        self.total_processing_time += (end_time - start_time)
        
        # Generate response based on provider characteristics
        if self.provider_name == "openai":
            return f"OpenAI-{self.call_count}: {input[:100]}..."
        elif self.provider_name == "anthropic":
            return f"Claude-{self.call_count}: {input[:100]}..."
        elif self.provider_name == "ollama":
            return f"Llama-{self.call_count}: {input[:100]}..."
        elif self.provider_name == "huggingface":
            return f"HF-{self.call_count}: {input[:100]}..."
        else:
            return f"Mock-{self.call_count}: {input[:100]}..."
    
    async def ainvoke(self, input, config=None, **kwargs):
        return self.invoke(input, config, **kwargs)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this provider instance."""
        uptime = time.time() - self.start_time
        avg_response_time = self.total_processing_time / max(1, self.call_count)
        requests_per_second = self.call_count / max(1, uptime)
        
        return {
            "provider": self.provider_name,
            "call_count": self.call_count,
            "total_processing_time": self.total_processing_time,
            "average_response_time": avg_response_time,
            "requests_per_second": requests_per_second,
            "uptime": uptime,
            "memory_usage": {
                "samples": len(self.memory_usage),
                "min_mb": min(self.memory_usage) / 1024 / 1024 if self.memory_usage else 0,
                "max_mb": max(self.memory_usage) / 1024 / 1024 if self.memory_usage else 0,
                "avg_mb": statistics.mean(self.memory_usage) / 1024 / 1024 if self.memory_usage else 0
            }
        }
    
    def validate_config(self) -> bool:
        return True
    
    def health_check(self) -> bool:
        return True
    
    def get_provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.provider_name,
            version="1.0.0",
            models=[f"{self.provider_name}-model"],
            capabilities={"chat": True, "completion": True},
            limits={"max_tokens": 4000, "requests_per_minute": 1000}
        )


class MemoryTrackingPlugin(PluginBase):
    """Plugin that tracks memory usage during execution."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_samples = []
        self.execution_times = []
        
    def __call__(self, input_data):
        start_time = time.time()
        
        # Track memory before processing
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Simulate processing
        result = f"TRACKED: {input_data}"
        
        # Track memory after processing
        memory_after = process.memory_info().rss
        end_time = time.time()
        
        self.memory_samples.append({
            "before_mb": memory_before / 1024 / 1024,
            "after_mb": memory_after / 1024 / 1024,
            "delta_mb": (memory_after - memory_before) / 1024 / 1024,
            "execution_time": end_time - start_time
        })
        
        self.execution_times.append(end_time - start_time)
        
        return result
    
    def get_memory_stats(self) -> Dict[str, Any]:
        if not self.memory_samples:
            return {"no_data": True}
        
        deltas = [sample["delta_mb"] for sample in self.memory_samples]
        execution_times = [sample["execution_time"] for sample in self.memory_samples]
        
        return {
            "total_executions": len(self.memory_samples),
            "memory_delta": {
                "min_mb": min(deltas),
                "max_mb": max(deltas),
                "avg_mb": statistics.mean(deltas),
                "total_mb": sum(deltas)
            },
            "execution_time": {
                "min_ms": min(execution_times) * 1000,
                "max_ms": max(execution_times) * 1000,
                "avg_ms": statistics.mean(execution_times) * 1000,
                "total_ms": sum(execution_times) * 1000
            }
        }
    
    def get_info(self):
        return {
            "name": "memory_tracking_plugin",
            "version": "1.0.0",
            "description": "Plugin for tracking memory usage"
        }


class TestPerformanceBenchmarking:
    """Test performance benchmarking across providers and configurations."""
    
    def setup_method(self):
        """Set up performance testing environment."""
        reset_config()
        clear_registry()
        
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "perf_test_config.json")
        
        # Create comprehensive test configuration
        self.test_config = {
            "providers": {
                "openai": {
                    "api_key": "test-openai-key",
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-4o-mini",
                    "timeout": 120,
                    "max_retries": 3
                },
                "anthropic": {
                    "api_key": "test-anthropic-key", 
                    "base_url": "https://api.anthropic.com",
                    "default_model": "claude-3-5-sonnet-20241022",
                    "timeout": 120,
                    "max_retries": 3
                },
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "default_model": "llama3.2:3b",
                    "timeout": 300,
                    "max_retries": 2
                },
                "huggingface": {
                    "api_key": "test-hf-key",
                    "base_url": "https://api-inference.huggingface.co",
                    "default_model": "microsoft/DialoGPT-medium",
                    "timeout": 120,
                    "max_retries": 3
                }
            },
            "profiles": {
                "fast-openai": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                    "max_tokens": 1000
                },
                "production-anthropic": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022", 
                    "temperature": 0.0,
                    "max_tokens": 4000
                },
                "local-ollama": {
                    "provider": "ollama",
                    "model": "llama3.2:3b",
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                "efficient-hf": {
                    "provider": "huggingface",
                    "model": "microsoft/DialoGPT-medium",
                    "temperature": 0.2,
                    "max_tokens": 1500
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)
    
    def teardown_method(self):
        """Clean up performance testing environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        reset_config()
        clear_registry()
        gc.collect()  # Force garbage collection

    def test_provider_performance_comparison(self):
        """Test performance comparison across all 4 providers."""
        
        provider_configs = {
            "openai": {"latency_ms": 50, "provider_name": "openai"},
            "anthropic": {"latency_ms": 30, "provider_name": "anthropic"},
            "ollama": {"latency_ms": 200, "provider_name": "ollama"},
            "huggingface": {"latency_ms": 100, "provider_name": "huggingface"}
        }
        
        performance_results = {}
        
        for provider, config in provider_configs.items():
            with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
                def create_perf_provider(llm_config, **kwargs):
                    return PerformanceLLMInterface(llm_config, **config)
                
                mock_get_provider.return_value = create_perf_provider
                
                # Create pipeline with performance tracking
                pipeline = PromptMatryoshka(
                    plugins=[BoostPlugin(num_eos=1), MemoryTrackingPlugin()],
                    config_path=self.config_path,
                    provider=provider
                )
                
                # Run performance test
                test_inputs = [f"Performance test {i} for {provider}" for i in range(20)]
                
                start_time = time.time()
                results = []
                
                for test_input in test_inputs:
                    result = pipeline.jailbreak(test_input)
                    results.append(result)
                
                end_time = time.time()
                
                # Calculate performance metrics
                total_time = end_time - start_time
                avg_time_per_request = total_time / len(test_inputs)
                requests_per_second = len(test_inputs) / total_time
                
                performance_results[provider] = {
                    "total_time": total_time,
                    "avg_time_per_request": avg_time_per_request,
                    "requests_per_second": requests_per_second,
                    "total_requests": len(test_inputs),
                    "all_successful": all(r.endswith("</s>") for r in results)
                }
        
        # Verify all providers completed successfully
        for provider, metrics in performance_results.items():
            assert metrics["all_successful"], f"Provider {provider} had failed requests"
            assert metrics["requests_per_second"] > 0, f"Provider {provider} had zero throughput"
        
        # Print performance comparison (for debugging)
        print("\nProvider Performance Comparison:")
        for provider, metrics in performance_results.items():
            print(f"{provider}: {metrics['requests_per_second']:.2f} req/s, "
                  f"{metrics['avg_time_per_request']*1000:.2f}ms avg")

    def test_load_testing_scenarios(self):
        """Test system behavior under various load scenarios."""
        
        load_scenarios = [
            {"concurrent_users": 5, "requests_per_user": 10, "name": "light_load"},
            {"concurrent_users": 10, "requests_per_user": 20, "name": "medium_load"},
            {"concurrent_users": 20, "requests_per_user": 15, "name": "heavy_load"}
        ]
        
        for scenario in load_scenarios:
            with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
                def load_test_provider(config, **kwargs):
                    return PerformanceLLMInterface(
                        config, 
                        provider_name="load_test",
                        latency_ms=25  # Fast provider for load testing
                    )
                
                mock_get_provider.return_value = load_test_provider
                
                pipeline = PromptMatryoshka(
                    plugins=[BoostPlugin(num_eos=1)],
                    config_path=self.config_path,
                    provider="openai"
                )
                
                def user_workload(user_id):
                    """Simulate individual user workload."""
                    results = []
                    start_time = time.time()
                    
                    for i in range(scenario["requests_per_user"]):
                        test_input = f"Load test user {user_id} request {i}"
                        try:
                            result = pipeline.jailbreak(test_input)
                            results.append({
                                "success": True,
                                "result": result,
                                "timestamp": time.time()
                            })
                        except Exception as e:
                            results.append({
                                "success": False,
                                "error": str(e),
                                "timestamp": time.time()
                            })
                    
                    end_time = time.time()
                    return {
                        "user_id": user_id,
                        "results": results,
                        "total_time": end_time - start_time,
                        "success_rate": sum(1 for r in results if r["success"]) / len(results)
                    }
                
                # Execute load test
                start_time = time.time()
                
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=scenario["concurrent_users"]
                ) as executor:
                    futures = [
                        executor.submit(user_workload, user_id) 
                        for user_id in range(scenario["concurrent_users"])
                    ]
                    user_results = [
                        future.result() 
                        for future in concurrent.futures.as_completed(futures)
                    ]
                
                end_time = time.time()
                
                # Analyze load test results
                total_requests = sum(len(ur["results"]) for ur in user_results)
                successful_requests = sum(
                    sum(1 for r in ur["results"] if r["success"]) 
                    for ur in user_results
                )
                overall_success_rate = successful_requests / total_requests
                total_test_time = end_time - start_time
                throughput = total_requests / total_test_time
                
                # Verify load test results
                assert overall_success_rate >= 0.95, f"Load test {scenario['name']} had low success rate: {overall_success_rate}"
                assert throughput > 0, f"Load test {scenario['name']} had zero throughput"
                
                print(f"\n{scenario['name']}: {throughput:.2f} req/s, "
                      f"{overall_success_rate*100:.1f}% success rate")

    def test_memory_usage_patterns(self):
        """Test memory usage patterns and leak detection."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            provider_instances = []
            
            def memory_tracking_provider(config, **kwargs):
                instance = PerformanceLLMInterface(
                    config, 
                    provider_name="memory_test",
                    latency_ms=10
                )
                provider_instances.append(instance)
                return instance
            
            mock_get_provider.return_value = memory_tracking_provider
            
            # Track initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            pipeline = PromptMatryoshka(
                plugins=[MemoryTrackingPlugin()],
                config_path=self.config_path,
                provider="openai"
            )
            
            memory_samples = []
            
            # Run memory usage test over multiple iterations
            for iteration in range(50):
                # Vary input sizes to test memory scaling
                input_sizes = [10, 100, 1000, 5000]
                
                for size in input_sizes:
                    test_input = "x" * size
                    result = pipeline.jailbreak(test_input)
                    
                    # Sample memory usage
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append({
                        "iteration": iteration,
                        "input_size": size,
                        "memory_mb": current_memory,
                        "timestamp": time.time()
                    })
                    
                    assert result.endswith("</s>")
                
                # Force garbage collection periodically
                if iteration % 10 == 0:
                    gc.collect()
            
            # Analyze memory usage patterns
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            max_memory = max(sample["memory_mb"] for sample in memory_samples)
            
            # Verify memory usage is reasonable
            assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f}MB"
            assert max_memory < initial_memory + 200, f"Memory usage too high: {max_memory:.1f}MB"
            
            print(f"\nMemory Usage: Initial={initial_memory:.1f}MB, "
                  f"Final={final_memory:.1f}MB, Growth={memory_growth:.1f}MB")

    def test_concurrent_execution_performance(self):
        """Test performance under concurrent execution scenarios."""
        
        concurrency_levels = [1, 5, 10, 20, 50]
        performance_metrics = {}
        
        for concurrency in concurrency_levels:
            with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
                execution_log = []
                lock = threading.Lock()
                
                def concurrent_provider(config, **kwargs):
                    with lock:
                        execution_log.append(f"provider_created_{len(execution_log)}")
                    
                    return PerformanceLLMInterface(
                        config,
                        provider_name=f"concurrent_{concurrency}",
                        latency_ms=20
                    )
                
                mock_get_provider.return_value = concurrent_provider
                
                pipeline = PromptMatryoshka(
                    plugins=[BoostPlugin(num_eos=1)],
                    config_path=self.config_path,
                    provider="openai"
                )
                
                def concurrent_task(task_id):
                    start_time = time.time()
                    test_input = f"Concurrent task {task_id}"
                    result = pipeline.jailbreak(test_input)
                    end_time = time.time()
                    
                    return {
                        "task_id": task_id,
                        "execution_time": end_time - start_time,
                        "success": result.endswith("</s>"),
                        "result_length": len(result)
                    }
                
                # Execute concurrent tasks
                start_time = time.time()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [
                        executor.submit(concurrent_task, task_id) 
                        for task_id in range(concurrency * 2)  # 2x tasks per worker
                    ]
                    task_results = [
                        future.result() 
                        for future in concurrent.futures.as_completed(futures)
                    ]
                
                end_time = time.time()
                
                # Calculate performance metrics
                total_time = end_time - start_time
                avg_task_time = statistics.mean(r["execution_time"] for r in task_results)
                success_rate = sum(1 for r in task_results if r["success"]) / len(task_results)
                tasks_per_second = len(task_results) / total_time
                
                performance_metrics[concurrency] = {
                    "total_time": total_time,
                    "avg_task_time": avg_task_time,
                    "success_rate": success_rate,
                    "tasks_per_second": tasks_per_second,
                    "total_tasks": len(task_results)
                }
                
                # Verify performance meets expectations
                assert success_rate >= 0.98, f"Low success rate at concurrency {concurrency}"
                assert tasks_per_second > 0, f"Zero throughput at concurrency {concurrency}"
        
        # Analyze scalability characteristics
        print("\nConcurrency Performance Analysis:")
        for concurrency, metrics in performance_metrics.items():
            print(f"Concurrency {concurrency}: {metrics['tasks_per_second']:.2f} tasks/s, "
                  f"{metrics['avg_task_time']*1000:.2f}ms avg")

    def test_caching_effectiveness(self):
        """Test caching effectiveness and cache invalidation."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            cache_hits = {"count": 0}
            call_log = []
            
            def caching_provider(config, **kwargs):
                instance = PerformanceLLMInterface(
                    config,
                    provider_name="caching_test",
                    latency_ms=30
                )
                
                # Override invoke to simulate caching
                original_invoke = instance.invoke
                
                def cached_invoke(input, config=None, **kwargs):
                    call_log.append(input)
                    
                    # Simulate cache hit for repeated inputs
                    if call_log.count(input) > 1:
                        cache_hits["count"] += 1
                        time.sleep(0.001)  # Cache hit is much faster
                        return f"CACHED: {input[:50]}..."
                    else:
                        return original_invoke(input, config, **kwargs)
                
                instance.invoke = cached_invoke
                return instance
            
            mock_get_provider.return_value = caching_provider
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path,
                provider="openai"
            )
            
            # Test cache effectiveness with repeated inputs
            test_inputs = [
                "cache test 1",
                "cache test 2", 
                "cache test 1",  # Repeat
                "cache test 3",
                "cache test 2",  # Repeat
                "cache test 1",  # Repeat
            ]
            
            cache_timing = []
            
            for input_text in test_inputs:
                start_time = time.time()
                result = pipeline.jailbreak(input_text)
                end_time = time.time()
                
                cache_timing.append({
                    "input": input_text,
                    "execution_time": end_time - start_time,
                    "is_cached": "CACHED:" in result,
                    "result": result
                })
                
                assert result.endswith("</s>")
            
            # Analyze caching effectiveness
            cached_requests = [t for t in cache_timing if t["is_cached"]]
            non_cached_requests = [t for t in cache_timing if not t["is_cached"]]
            
            if cached_requests and non_cached_requests:
                avg_cached_time = statistics.mean(r["execution_time"] for r in cached_requests)
                avg_non_cached_time = statistics.mean(r["execution_time"] for r in non_cached_requests)
                
                # Cache hits should be significantly faster
                assert avg_cached_time < avg_non_cached_time * 0.5, "Caching not effective"
                
                print(f"\nCache Performance: Cached={avg_cached_time*1000:.2f}ms, "
                      f"Non-cached={avg_non_cached_time*1000:.2f}ms")

    def test_scalability_testing(self):
        """Test system scalability with increasing workloads."""
        
        workload_sizes = [10, 50, 100, 200, 500]
        scalability_results = {}
        
        for workload_size in workload_sizes:
            with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
                def scalable_provider(config, **kwargs):
                    return PerformanceLLMInterface(
                        config,
                        provider_name=f"scale_{workload_size}",
                        latency_ms=15
                    )
                
                mock_get_provider.return_value = scalable_provider
                
                pipeline = PromptMatryoshka(
                    plugins=[BoostPlugin(num_eos=1)],
                    config_path=self.config_path,
                    provider="openai"
                )
                
                # Generate workload
                test_inputs = [f"Scalability test {i}" for i in range(workload_size)]
                
                # Measure execution time
                start_time = time.time()
                
                results = []
                for test_input in test_inputs:
                    result = pipeline.jailbreak(test_input)
                    results.append(result)
                
                end_time = time.time()
                
                # Calculate scalability metrics
                total_time = end_time - start_time
                throughput = workload_size / total_time
                avg_latency = total_time / workload_size
                success_rate = sum(1 for r in results if r.endswith("</s>")) / len(results)
                
                scalability_results[workload_size] = {
                    "total_time": total_time,
                    "throughput": throughput,
                    "avg_latency": avg_latency,
                    "success_rate": success_rate
                }
                
                # Verify scalability requirements
                assert success_rate >= 0.99, f"Low success rate at workload {workload_size}"
                assert throughput > 0, f"Zero throughput at workload {workload_size}"
        
        # Analyze scalability trends
        print("\nScalability Analysis:")
        for workload_size, metrics in scalability_results.items():
            print(f"Workload {workload_size}: {metrics['throughput']:.2f} req/s, "
                  f"{metrics['avg_latency']*1000:.2f}ms avg latency")

    def test_resource_consumption_monitoring(self):
        """Test comprehensive resource consumption monitoring."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            def resource_monitored_provider(config, **kwargs):
                return PerformanceLLMInterface(
                    config,
                    provider_name="resource_monitor",
                    latency_ms=25
                )
            
            mock_get_provider.return_value = resource_monitored_provider
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1), MemoryTrackingPlugin()],
                config_path=self.config_path,
                provider="openai"
            )
            
            # Monitor resources during execution
            process = psutil.Process()
            resource_samples = []
            
            # Baseline measurement
            baseline_cpu = process.cpu_percent()
            baseline_memory = process.memory_info().rss / 1024 / 1024
            
            # Execute workload while monitoring resources
            for i in range(100):
                test_input = f"Resource monitoring test {i}"
                
                # Sample resources before
                cpu_before = process.cpu_percent()
                memory_before = process.memory_info().rss / 1024 / 1024
                
                start_time = time.time()
                result = pipeline.jailbreak(test_input)
                end_time = time.time()
                
                # Sample resources after
                cpu_after = process.cpu_percent()
                memory_after = process.memory_info().rss / 1024 / 1024
                
                resource_samples.append({
                    "iteration": i,
                    "cpu_percent": cpu_after,
                    "memory_mb": memory_after,
                    "execution_time": end_time - start_time,
                    "memory_delta": memory_after - memory_before
                })
                
                assert result.endswith("</s>")
                
                # Brief pause to allow CPU measurement
                time.sleep(0.01)
            
            # Analyze resource consumption
            avg_cpu = statistics.mean(sample["cpu_percent"] for sample in resource_samples)
            max_cpu = max(sample["cpu_percent"] for sample in resource_samples)
            avg_memory = statistics.mean(sample["memory_mb"] for sample in resource_samples)
            max_memory = max(sample["memory_mb"] for sample in resource_samples)
            total_memory_growth = max_memory - baseline_memory
            
            # Verify resource consumption is reasonable
            assert avg_cpu < 80, f"High average CPU usage: {avg_cpu:.1f}%"
            assert max_cpu < 95, f"CPU usage spike: {max_cpu:.1f}%"
            assert total_memory_growth < 50, f"Excessive memory growth: {total_memory_growth:.1f}MB"
            
            print(f"\nResource Usage: CPU avg={avg_cpu:.1f}% max={max_cpu:.1f}%, "
                  f"Memory avg={avg_memory:.1f}MB max={max_memory:.1f}MB")

    def test_performance_regression_detection(self):
        """Test performance regression detection mechanisms."""
        
        # Simulate different performance characteristics
        performance_scenarios = {
            "baseline": {"latency_ms": 30, "expected_min_rps": 20},
            "improved": {"latency_ms": 15, "expected_min_rps": 40}, 
            "degraded": {"latency_ms": 60, "expected_min_rps": 10}
        }
        
        regression_results = {}
        
        for scenario_name, scenario_config in performance_scenarios.items():
            with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
                def scenario_provider(config, **kwargs):
                    return PerformanceLLMInterface(
                        config,
                        provider_name=f"regression_{scenario_name}",
                        latency_ms=scenario_config["latency_ms"]
                    )
                
                mock_get_provider.return_value = scenario_provider
                
                pipeline = PromptMatryoshka(
                    plugins=[BoostPlugin(num_eos=1)],
                    config_path=self.config_path,
                    provider="openai"
                )
                
                # Run standardized performance test
                test_count = 30
                test_inputs = [f"Regression test {i}" for i in range(test_count)]
                
                start_time = time.time()
                execution_times = []
                
                for test_input in test_inputs:
                    exec_start = time.time()
                    result = pipeline.jailbreak(test_input)
                    exec_end = time.time()
                    
                    execution_times.append(exec_end - exec_start)
                    assert result.endswith("</s>")
                
                end_time = time.time()
                
                # Calculate performance metrics
                total_time = end_time - start_time
                requests_per_second = test_count / total_time
                avg_response_time = statistics.mean(execution_times)
                p95_response_time = sorted(execution_times)[int(0.95 * len(execution_times))]
                
                regression_results[scenario_name] = {
                    "requests_per_second": requests_per_second,
                    "avg_response_time": avg_response_time,
                    "p95_response_time": p95_response_time,
                    "expected_min_rps": scenario_config["expected_min_rps"]
                }
                
                # Check against expected performance
                assert requests_per_second >= scenario_config["expected_min_rps"] * 0.8, \
                    f"Performance regression in {scenario_name}: {requests_per_second:.2f} < {scenario_config['expected_min_rps']}"
        
        # Compare scenarios to detect regressions
        print("\nPerformance Regression Analysis:")
        for scenario, metrics in regression_results.items():
            print(f"{scenario}: {metrics['requests_per_second']:.2f} req/s, "
                  f"avg={metrics['avg_response_time']*1000:.2f}ms, "
                  f"p95={metrics['p95_response_time']*1000:.2f}ms")

    def test_provider_switching_performance_impact(self):
        """Test performance impact of provider switching."""
        
        providers = ["openai", "anthropic", "ollama", "huggingface"]
        switching_metrics = []
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            switch_count = {"count": 0}
            
            def switching_provider(config, **kwargs):
                switch_count["count"] += 1
                provider_name = kwargs.get('provider_name', 'unknown')
                
                # Different latencies for different providers
                latencies = {
                    "openai": 25,
                    "anthropic": 20,
                    "ollama": 100,
                    "huggingface": 50
                }
                
                return PerformanceLLMInterface(
                    config,
                    provider_name=provider_name,
                    latency_ms=latencies.get(provider_name, 30)
                )
            
            mock_get_provider.return_value = switching_provider
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            # Test provider switching performance
            for i in range(20):
                provider = providers[i % len(providers)]
                test_input = f"Provider switching test {i}"
                
                start_time = time.time()
                result = pipeline.jailbreak(test_input, provider=provider)
                end_time = time.time()
                
                switching_metrics.append({
                    "iteration": i,
                    "provider": provider,
                    "execution_time": end_time - start_time,
                    "switch_overhead": switch_count["count"]
                })
                
                assert result.endswith("</s>")
            
            # Analyze switching overhead
            provider_times = {}
            for provider in providers:
                provider_metrics = [m for m in switching_metrics if m["provider"] == provider]
                if provider_metrics:
                    avg_time = statistics.mean(m["execution_time"] for m in provider_metrics)
                    provider_times[provider] = avg_time
            
            print("\nProvider Switching Performance:")
            for provider, avg_time in provider_times.items():
                print(f"{provider}: {avg_time*1000:.2f}ms average")


class TestReliabilityCharacteristics:
    """Test reliability characteristics under sustained load."""
    
    def setup_method(self):
        """Set up reliability testing environment."""
        reset_config()
        clear_registry()
        
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "reliability_test_config.json")
        
        self.test_config = {
            "providers": {
                "reliable": {
                    "api_key": "test-reliable-key",
                    "base_url": "https://reliable.example.com",
                    "default_model": "reliable-model",
                    "timeout": 60,
                    "max_retries": 3
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)
    
    def teardown_method(self):
        """Clean up reliability testing environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        reset_config()
        clear_registry()

    def test_sustained_load_reliability(self):
        """Test reliability under sustained load over time."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            reliability_metrics = []
            
            def reliable_provider(config, **kwargs):
                return PerformanceLLMInterface(
                    config,
                    provider_name="sustained_load",
                    latency_ms=20
                )
            
            mock_get_provider.return_value = reliable_provider
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path,
                provider="reliable"
            )
            
            # Run sustained load test for extended period
            test_duration = 30  # seconds
            start_time = time.time()
            request_count = 0
            success_count = 0
            
            while time.time() - start_time < test_duration:
                test_input = f"Sustained load test {request_count}"
                
                try:
                    result = pipeline.jailbreak(test_input)
                    if result.endswith("</s>"):
                        success_count += 1
                    request_count += 1
                    
                    # Sample reliability metrics
                    if request_count % 20 == 0:
                        current_time = time.time() - start_time
                        success_rate = success_count / request_count
                        requests_per_second = request_count / current_time
                        
                        reliability_metrics.append({
                            "elapsed_time": current_time,
                            "total_requests": request_count,
                            "success_rate": success_rate,
                            "requests_per_second": requests_per_second
                        })
                
                except Exception as e:
                    request_count += 1
                    # Log error but continue test
                
                # Brief pause between requests
                time.sleep(0.01)
            
            # Analyze sustained load reliability
            final_success_rate = success_count / request_count
            final_rps = request_count / test_duration
            
            # Verify reliability requirements
            assert final_success_rate >= 0.95, f"Low reliability: {final_success_rate:.3f}"
            assert final_rps > 10, f"Low throughput under sustained load: {final_rps:.2f}"
            
            print(f"\nSustained Load: {final_rps:.2f} req/s, "
                  f"{final_success_rate*100:.1f}% success rate over {test_duration}s")

    def test_response_time_consistency(self):
        """Test response time consistency and variance."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            def consistent_provider(config, **kwargs):
                return PerformanceLLMInterface(
                    config,
                    provider_name="consistency_test",
                    latency_ms=30
                )
            
            mock_get_provider.return_value = consistent_provider
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path,
                provider="reliable"
            )
            
            # Collect response time samples
            response_times = []
            test_count = 100
            
            for i in range(test_count):
                test_input = f"Consistency test {i}"
                
                start_time = time.time()
                result = pipeline.jailbreak(test_input)
                end_time = time.time()
                
                response_times.append(end_time - start_time)
                assert result.endswith("</s>")
            
            # Analyze response time consistency
            mean_time = statistics.mean(response_times)
            median_time = statistics.median(response_times)
            std_dev = statistics.stdev(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            p50 = sorted_times[int(0.50 * len(sorted_times))]
            p95 = sorted_times[int(0.95 * len(sorted_times))]
            p99 = sorted_times[int(0.99 * len(sorted_times))]
            
            # Verify consistency requirements
            coefficient_of_variation = std_dev / mean_time
            assert coefficient_of_variation < 0.5, f"High response time variance: {coefficient_of_variation:.3f}"
            assert max_time < mean_time * 3, f"Response time outlier: {max_time:.3f}s vs {mean_time:.3f}s avg"
            
            print(f"\nResponse Time Consistency:")
            print(f"Mean: {mean_time*1000:.2f}ms, StdDev: {std_dev*1000:.2f}ms")
            print(f"P50: {p50*1000:.2f}ms, P95: {p95*1000:.2f}ms, P99: {p99*1000:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])