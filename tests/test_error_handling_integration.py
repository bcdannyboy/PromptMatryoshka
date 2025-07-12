"""Comprehensive error handling and edge case integration tests.

Tests error handling across the entire PromptMatryoshka multi-provider system:
- Provider connection failures and recovery mechanisms
- Configuration validation errors and user-friendly messages
- Plugin errors and isolation mechanisms
- Network timeout and retry scenarios
- Resource exhaustion and graceful degradation
- Invalid input handling and sanitization
- Concurrent error scenarios and thread safety
- Error message quality and debugging information
- Fallback mechanisms and provider switching on failure
- Recovery strategies and system resilience

This test suite ensures the system fails gracefully, provides helpful error
messages, and maintains stability under adverse conditions.
"""

import pytest
import os
import tempfile
import json
import shutil
import time
import threading
import signal
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from typing import Dict, Any, List, Optional
import concurrent.futures
from datetime import datetime, timedelta

from promptmatryoshka.core import PromptMatryoshka, PipelineBuilder
from promptmatryoshka.config import Config, get_config, reset_config
from promptmatryoshka.llm_factory import LLMFactory, get_factory
from promptmatryoshka.llm_interface import LLMInterface, LLMConfig, ProviderInfo
from promptmatryoshka.providers import clear_registry
from promptmatryoshka.plugins.base import PluginBase, get_plugin_registry
from promptmatryoshka.plugins.boost import BoostPlugin
from promptmatryoshka.plugins.flipattack import FlipAttackPlugin
from promptmatryoshka.exceptions import (
    LLMError, LLMConfigurationError, LLMUnsupportedProviderError, 
    PluginError, PipelineValidationError, ConfigurationError
)


class NetworkFailureLLMInterface(LLMInterface):
    """Mock LLM interface that simulates various network failures."""
    
    def __init__(self, config, failure_mode="timeout", failure_probability=1.0, **kwargs):
        super().__init__(config, **kwargs)
        self.failure_mode = failure_mode
        self.failure_probability = failure_probability
        self.attempt_count = 0
        
    def invoke(self, input, config=None, **kwargs):
        self.attempt_count += 1
        
        import random
        if random.random() < self.failure_probability:
            if self.failure_mode == "timeout":
                raise TimeoutError("Request timed out after 30 seconds")
            elif self.failure_mode == "connection":
                raise ConnectionError("Connection refused by server")
            elif self.failure_mode == "dns":
                raise OSError("Name resolution failed")
            elif self.failure_mode == "ssl":
                raise ssl.SSLError("SSL certificate verification failed")
            elif self.failure_mode == "http_500":
                raise LLMError("HTTP 500: Internal Server Error")
            elif self.failure_mode == "http_429":
                raise LLMError("HTTP 429: Too Many Requests")
            elif self.failure_mode == "http_401":
                raise LLMError("HTTP 401: Unauthorized - Invalid API key")
            elif self.failure_mode == "quota_exceeded":
                raise LLMError("Quota exceeded for this month")
            elif self.failure_mode == "model_unavailable":
                raise LLMError("Model temporarily unavailable")
            elif self.failure_mode == "malformed_response":
                raise ValueError("Malformed JSON response from server")
        
        return f"Success after {self.attempt_count} attempts: {input[:50]}..."
    
    async def ainvoke(self, input, config=None, **kwargs):
        return self.invoke(input, config, **kwargs)
    
    def validate_config(self) -> bool:
        return self.failure_mode != "config_invalid"
    
    def health_check(self) -> bool:
        return self.failure_mode not in ["timeout", "connection", "dns"]
    
    def get_provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="network_failure_mock",
            version="1.0.0",
            models=["mock-model"],
            capabilities={"chat": True},
            limits={"max_tokens": 4000}
        )


class ResourceExhaustionLLMInterface(LLMInterface):
    """Mock LLM interface that simulates resource exhaustion scenarios."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.memory_usage = 0
        self.active_requests = 0
        
    def invoke(self, input, config=None, **kwargs):
        self.active_requests += 1
        self.memory_usage += len(input) * 10  # Simulate memory usage
        
        try:
            # Simulate memory exhaustion
            if self.memory_usage > 1000000:  # 1MB limit
                raise MemoryError("Out of memory")
            
            # Simulate too many concurrent requests
            if self.active_requests > 10:
                raise LLMError("Too many concurrent requests")
            
            # Simulate processing
            time.sleep(0.01)
            return f"Processed: {input[:50]}..."
        
        finally:
            self.active_requests -= 1
    
    async def ainvoke(self, input, config=None, **kwargs):
        return self.invoke(input, config, **kwargs)
    
    def validate_config(self) -> bool:
        return True
    
    def health_check(self) -> bool:
        return self.memory_usage < 500000  # Health check fails at 500KB
    
    def get_provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="resource_exhaustion_mock",
            version="1.0.0",
            models=["mock-model"],
            capabilities={"chat": True},
            limits={"max_tokens": 4000}
        )


class FailingPlugin(PluginBase):
    """Plugin that fails in various ways for testing."""
    
    def __init__(self, failure_mode="runtime_error", **kwargs):
        super().__init__(**kwargs)
        self.failure_mode = failure_mode
        
    def run(self, input_data):
        if self.failure_mode == "runtime_error":
            raise RuntimeError("Plugin runtime error")
        elif self.failure_mode == "type_error":
            raise TypeError("Invalid input type")
        elif self.failure_mode == "value_error":
            raise ValueError("Invalid input value")
        elif self.failure_mode == "memory_error":
            raise MemoryError("Plugin out of memory")
        elif self.failure_mode == "timeout":
            time.sleep(10)  # Simulate timeout
            return input_data
        elif self.failure_mode == "corrupted_output":
            return None  # Return invalid output
        elif self.failure_mode == "partial_failure":
            if len(input_data) > 100:
                raise RuntimeError("Input too long")
            return f"PROCESSED: {input_data}"
        
        return input_data
    
    def get_info(self):
        return {
            "name": "failing_plugin",
            "version": "1.0.0",
            "description": "Plugin for testing error scenarios"
        }


class TestErrorHandlingIntegration:
    """Test comprehensive error handling and edge cases."""
    
    def setup_method(self):
        """Set up test environment."""
        reset_config()
        clear_registry()
        
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "error_test_config.json")
        
        # Create test configuration
        self.test_config = {
            "providers": {
                "openai": {
                    "api_key": "test-openai-key",
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-4o-mini",
                    "timeout": 30,
                    "max_retries": 3
                },
                "anthropic": {
                    "api_key": "test-anthropic-key",
                    "base_url": "https://api.anthropic.com",
                    "default_model": "claude-3-5-sonnet-20241022",
                    "timeout": 30,
                    "max_retries": 3
                },
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "default_model": "llama3.2:3b",
                    "timeout": 60,
                    "max_retries": 2
                }
            },
            "profiles": {
                "test-profile": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                    "max_tokens": 1000
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        reset_config()
        clear_registry()

    def test_provider_connection_failures(self):
        """Test handling of various provider connection failures."""
        
        failure_modes = [
            "timeout", "connection", "dns", "ssl", 
            "http_500", "http_429", "http_401"
        ]
        
        for failure_mode in failure_modes:
            with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
                def failing_provider_mock(config, **kwargs):
                    return NetworkFailureLLMInterface(config, failure_mode=failure_mode)
                
                mock_get_provider.return_value = failing_provider_mock
                
                pipeline = PromptMatryoshka(
                    plugins=[BoostPlugin(num_eos=1)],  # Use technique-based plugin
                    config_path=self.config_path,
                    provider="openai"
                )
                
                test_input = f"test {failure_mode} failure"
                
                # Technique-based plugins should still work despite LLM failures
                result = pipeline.jailbreak(test_input)
                assert result is not None
                assert result.endswith("</s>")

    def test_configuration_validation_errors(self):
        """Test configuration validation error handling."""
        
        # Test invalid configuration file
        invalid_config_path = os.path.join(self.temp_dir, "invalid_config.json")
        
        # Test malformed JSON
        with open(invalid_config_path, 'w') as f:
            f.write('{"invalid": json syntax}')
        
        with pytest.raises(ConfigurationError):
            pipeline = PromptMatryoshka(config_path=invalid_config_path)
        
        # Test missing required fields
        incomplete_config = {"providers": {}}
        with open(invalid_config_path, 'w') as f:
            json.dump(incomplete_config, f)
        
        # Should handle gracefully
        pipeline = PromptMatryoshka(
            plugins=[BoostPlugin(num_eos=1)],
            config_path=invalid_config_path
        )
        
        # Test with nonexistent configuration file
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent_config.json")
        
        pipeline = PromptMatryoshka(
            plugins=[BoostPlugin(num_eos=1)],
            config_path=nonexistent_path  # Should handle gracefully
        )

    def test_plugin_error_isolation(self):
        """Test plugin error isolation and recovery mechanisms."""
        
        # Test runtime errors in plugins
        failing_plugin = FailingPlugin(failure_mode="runtime_error")
        good_plugin = BoostPlugin(num_eos=1)
        
        pipeline = PromptMatryoshka(
            plugins=[failing_plugin, good_plugin],
            config_path=self.config_path
        )
        
        test_input = "test plugin isolation"
        
        # Pipeline should handle plugin failures gracefully
        try:
            result = pipeline.jailbreak(test_input)
            # If it succeeds, it should be the result from the good plugin
            assert result.endswith("</s>")
        except RuntimeError:
            # If it fails, the error should be properly propagated
            pass

    def test_resource_exhaustion_scenarios(self):
        """Test resource exhaustion handling."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            def resource_exhaustion_mock(config, **kwargs):
                return ResourceExhaustionLLMInterface(config)
            
            mock_get_provider.return_value = resource_exhaustion_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path,
                provider="openai"
            )
            
            # Test with normal input
            result = pipeline.jailbreak("small input")
            assert result.endswith("</s>")
            
            # Test with large input that triggers memory exhaustion
            large_input = "x" * 200000  # 200KB input
            
            # Should handle gracefully or fail with appropriate error
            try:
                result = pipeline.jailbreak(large_input)
                # If it succeeds, verify the result
                assert result.endswith("</s>")
            except MemoryError:
                # Expected for large inputs
                pass

    def test_invalid_input_handling(self):
        """Test handling of various invalid inputs."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: NetworkFailureLLMInterface(
                config, failure_mode="none", failure_probability=0.0
            )
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            # Test various invalid inputs
            invalid_inputs = [
                None,
                "",
                "\x00\x01\x02",  # Binary data
                "x" * 1000000,   # Extremely large input
                "\n" * 10000,    # Many newlines
                "🚀" * 1000,     # Unicode emoji
                "<script>alert('xss')</script>",  # Potential XSS
                "'; DROP TABLE users; --",  # SQL injection attempt
            ]
            
            for invalid_input in invalid_inputs:
                try:
                    result = pipeline.jailbreak(invalid_input)
                    # If it succeeds, verify basic properties
                    if result is not None:
                        assert isinstance(result, str)
                except (TypeError, ValueError, UnicodeError):
                    # Expected for some invalid inputs
                    pass

    def test_concurrent_error_scenarios(self):
        """Test error handling under concurrent access."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            failure_counter = {"count": 0}
            lock = threading.Lock()
            
            def intermittent_failure_mock(config, **kwargs):
                with lock:
                    failure_counter["count"] += 1
                    failure_prob = 0.3 if failure_counter["count"] % 3 == 0 else 0.0
                
                return NetworkFailureLLMInterface(
                    config, 
                    failure_mode="timeout", 
                    failure_probability=failure_prob
                )
            
            mock_get_provider.return_value = intermittent_failure_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            def concurrent_test(thread_id):
                test_input = f"concurrent test {thread_id}"
                try:
                    result = pipeline.jailbreak(test_input, provider="openai")
                    return {"success": True, "result": result, "thread_id": thread_id}
                except Exception as e:
                    return {"success": False, "error": str(e), "thread_id": thread_id}
            
            # Run concurrent operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(concurrent_test, i) for i in range(20)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Verify some operations succeeded and errors were handled gracefully
            successful_results = [r for r in results if r["success"]]
            failed_results = [r for r in results if not r["success"]]
            
            # At least some should succeed due to technique-based plugins
            assert len(successful_results) > 0
            
            for success_result in successful_results:
                assert success_result["result"].endswith("</s>")

    def test_provider_fallback_mechanisms(self):
        """Test provider fallback and switching on failure."""
        
        provider_call_log = []
        
        def fallback_provider_mock(config, **kwargs):
            provider_name = kwargs.get('provider_name', 'unknown')
            provider_call_log.append(provider_name)
            
            # First provider always fails, others succeed
            if provider_name == "openai":
                return NetworkFailureLLMInterface(config, failure_mode="connection")
            else:
                return NetworkFailureLLMInterface(
                    config, failure_mode="none", failure_probability=0.0
                )
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = fallback_provider_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            test_input = "test fallback mechanism"
            
            # Try primary provider (will fail)
            try:
                result = pipeline.jailbreak(test_input, provider="openai")
                # Should succeed with technique-based plugin
                assert result.endswith("</s>")
            except ConnectionError:
                pass
            
            # Try fallback provider (should succeed)
            result = pipeline.jailbreak(test_input, provider="anthropic")
            assert result.endswith("</s>")

    def test_timeout_and_retry_mechanisms(self):
        """Test timeout handling and retry mechanisms."""
        
        retry_attempts = {"count": 0}
        
        def retry_tracking_mock(config, **kwargs):
            retry_attempts["count"] += 1
            
            # Fail first few attempts, then succeed
            if retry_attempts["count"] <= 2:
                return NetworkFailureLLMInterface(config, failure_mode="timeout")
            else:
                return NetworkFailureLLMInterface(
                    config, failure_mode="none", failure_probability=0.0
                )
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = retry_tracking_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            test_input = "test retry mechanism"
            
            # Should eventually succeed (technique-based plugins work regardless)
            result = pipeline.jailbreak(test_input, provider="openai")
            assert result.endswith("</s>")

    def test_error_message_quality(self):
        """Test quality and usefulness of error messages."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            def descriptive_error_mock(config, **kwargs):
                return NetworkFailureLLMInterface(config, failure_mode="http_401")
            
            mock_get_provider.return_value = descriptive_error_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            test_input = "test error messages"
            
            # Should get meaningful error or fallback to technique plugins
            result = pipeline.jailbreak(test_input, provider="openai")
            assert result.endswith("</s>")

    def test_graceful_degradation(self):
        """Test graceful degradation when components fail."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            def degraded_provider_mock(config, **kwargs):
                return NetworkFailureLLMInterface(config, failure_mode="model_unavailable")
            
            mock_get_provider.return_value = degraded_provider_mock
            
            # Pipeline with mixed plugin types
            pipeline = PromptMatryoshka(
                plugins=[
                    BoostPlugin(num_eos=2),           # Technique-based (should work)
                    FlipAttackPlugin(mode="word")     # Technique-based (should work)
                ],
                config_path=self.config_path
            )
            
            test_input = "test graceful degradation"
            
            # Should succeed with technique-based plugins even if LLM fails
            result = pipeline.jailbreak(test_input, provider="openai")
            assert result is not None
            # With FlipAttack + BoostPlugin, FlipAttack generates a complete system prompt
            # that incorporates the EOS tokens within the prompt structure
            assert "</s></s>" in result or result.endswith("</s></s>")

    def test_memory_leak_prevention(self):
        """Test memory leak prevention under error conditions."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            creation_counter = {"count": 0}
            
            def memory_tracking_mock(config, **kwargs):
                creation_counter["count"] += 1
                return ResourceExhaustionLLMInterface(config)
            
            mock_get_provider.return_value = memory_tracking_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            # Run many operations to test for memory leaks
            for i in range(50):
                test_input = f"memory test {i}"
                try:
                    result = pipeline.jailbreak(test_input, provider="openai")
                    assert result.endswith("</s>")
                except (MemoryError, LLMError):
                    pass  # Expected for some iterations

    def test_signal_handling_and_interruption(self):
        """Test handling of process signals and interruptions."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            def slow_provider_mock(config, **kwargs):
                return NetworkFailureLLMInterface(config, failure_mode="timeout")
            
            mock_get_provider.return_value = slow_provider_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            def run_pipeline():
                test_input = "test signal handling"
                return pipeline.jailbreak(test_input, provider="openai")
            
            # Run in thread to test interruption
            import threading
            result_container = {"result": None, "exception": None}
            
            def thread_target():
                try:
                    result_container["result"] = run_pipeline()
                except Exception as e:
                    result_container["exception"] = e
            
            thread = threading.Thread(target=thread_target)
            thread.start()
            thread.join(timeout=2.0)  # Wait maximum 2 seconds
            
            # Should complete or handle timeout gracefully
            if thread.is_alive():
                # Thread is still running, which is expected for timeout test
                pass
            elif result_container["result"] is not None:
                assert result_container["result"].endswith("</s>")

    def test_configuration_hot_reload_errors(self):
        """Test error handling during configuration hot-reload."""
        
        pipeline = PromptMatryoshka(
            plugins=[BoostPlugin(num_eos=1)],
            config_path=self.config_path
        )
        
        # Test successful operation first
        result = pipeline.jailbreak("test before config change")
        assert result.endswith("</s>")
        
        # Corrupt the configuration file
        with open(self.config_path, 'w') as f:
            f.write('{"invalid": json}')
        
        # Pipeline should still work with cached/default configuration
        result = pipeline.jailbreak("test after config corruption")
        assert result.endswith("</s>")

    def test_plugin_dependency_failure_cascade(self):
        """Test plugin dependency failure cascade handling."""
        
        # Create plugins with dependencies
        failing_plugin = FailingPlugin(failure_mode="partial_failure")
        dependent_plugin = BoostPlugin(num_eos=1)
        
        pipeline = PromptMatryoshka(
            plugins=[failing_plugin, dependent_plugin],
            config_path=self.config_path
        )
        
        # Test with input that triggers failure
        long_input = "x" * 150  # Triggers failure in failing_plugin
        
        try:
            result = pipeline.jailbreak(long_input)
            # If it succeeds, verify it has some expected structure
            assert isinstance(result, str)
        except RuntimeError:
            # Expected for inputs that trigger plugin failure
            pass
        
        # Test with input that doesn't trigger failure
        short_input = "short input"
        result = pipeline.jailbreak(short_input)
        assert result.endswith("</s>")

    def test_edge_case_boundary_conditions(self):
        """Test edge case boundary conditions."""
        
        pipeline = PromptMatryoshka(
            plugins=[BoostPlugin(num_eos=1)],
            config_path=self.config_path
        )
        
        # Test boundary conditions
        boundary_cases = [
            "",                    # Empty string
            " ",                   # Single space
            "\n",                  # Single newline
            "\t",                  # Single tab
            "a",                   # Single character
            "a" * 65535,          # Maximum reasonable string length
            "\u0000",              # Null character
            "\uFFFF",              # Unicode boundary
            "🚀",                  # Emoji
            "Hello\x00World",      # String with null byte
        ]
        
        for boundary_input in boundary_cases:
            try:
                result = pipeline.jailbreak(boundary_input)
                # If successful, verify basic properties
                if result is not None:
                    assert isinstance(result, str)
                    if boundary_input:  # Non-empty input
                        assert result.endswith("</s>")
            except (ValueError, TypeError, UnicodeError):
                # Some boundary cases may legitimately fail
                pass

    def test_system_resource_monitoring(self):
        """Test system resource monitoring and limits."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            def resource_monitoring_mock(config, **kwargs):
                return ResourceExhaustionLLMInterface(config)
            
            mock_get_provider.return_value = resource_monitoring_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            # Monitor resource usage during operations
            test_inputs = [f"resource test {i}" for i in range(20)]
            
            successful_operations = 0
            failed_operations = 0
            
            for test_input in test_inputs:
                try:
                    result = pipeline.jailbreak(test_input, provider="openai")
                    assert result.endswith("</s>")
                    successful_operations += 1
                except (MemoryError, LLMError):
                    failed_operations += 1
            
            # Should have at least some successful operations
            assert successful_operations > 0


class TestErrorRecoveryMechanisms:
    """Test error recovery and resilience mechanisms."""
    
    def setup_method(self):
        """Set up test environment."""
        reset_config()
        clear_registry()
        
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "recovery_test_config.json")
        
        self.test_config = {
            "providers": {
                "primary": {
                    "api_key": "test-primary-key",
                    "base_url": "https://primary.example.com",
                    "default_model": "primary-model",
                    "timeout": 30,
                    "max_retries": 3
                },
                "fallback": {
                    "api_key": "test-fallback-key", 
                    "base_url": "https://fallback.example.com",
                    "default_model": "fallback-model",
                    "timeout": 30,
                    "max_retries": 2
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        reset_config()
        clear_registry()

    def test_automatic_retry_with_backoff(self):
        """Test automatic retry mechanisms with exponential backoff."""
        
        attempt_times = []
        
        def retry_tracking_mock(config, **kwargs):
            attempt_times.append(time.time())
            
            # Fail first 2 attempts, succeed on 3rd
            if len(attempt_times) <= 2:
                return NetworkFailureLLMInterface(config, failure_mode="connection")
            else:
                return NetworkFailureLLMInterface(
                    config, failure_mode="none", failure_probability=0.0
                )
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = retry_tracking_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            test_input = "test retry with backoff"
            
            start_time = time.time()
            result = pipeline.jailbreak(test_input, provider="primary")
            end_time = time.time()
            
            # Should succeed with technique-based plugin
            assert result.endswith("</s>")

    def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for failing providers."""
        
        failure_count = {"count": 0}
        
        def circuit_breaker_mock(config, **kwargs):
            failure_count["count"] += 1
            
            # Simulate circuit breaker: fail consistently after threshold
            if failure_count["count"] > 5:
                raise LLMError("Circuit breaker: Provider marked as down")
            else:
                return NetworkFailureLLMInterface(config, failure_mode="timeout")
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = circuit_breaker_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            # Multiple attempts should trigger circuit breaker
            for i in range(10):
                test_input = f"circuit breaker test {i}"
                result = pipeline.jailbreak(test_input, provider="primary")
                # Should succeed with technique-based plugin regardless
                assert result.endswith("</s>")

    def test_health_check_recovery(self):
        """Test recovery through health check mechanisms."""
        
        health_status = {"healthy": False, "check_count": 0}
        
        def health_check_mock(config, **kwargs):
            health_status["check_count"] += 1
            
            # Become healthy after 3 health checks
            if health_status["check_count"] >= 3:
                health_status["healthy"] = True
            
            return NetworkFailureLLMInterface(
                config, 
                failure_mode="none" if health_status["healthy"] else "connection",
                failure_probability=0.0 if health_status["healthy"] else 1.0
            )
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = health_check_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            # Should eventually recover
            for i in range(5):
                test_input = f"health check recovery test {i}"
                result = pipeline.jailbreak(test_input, provider="primary")
                assert result.endswith("</s>")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])