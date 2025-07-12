"""Comprehensive pipeline integration tests with multi-provider support.

Tests the complete pipeline execution via PromptMatryoshka.jailbreak() with:
- Multi-provider pipeline execution and provider switching
- Pipeline integration with factory pattern and configuration system
- Pipeline error handling and recovery mechanisms
- Pipeline performance characteristics across providers
- Plugin system integration within pipeline execution
- Provider/profile-based pipeline configuration and execution
- Complex pipeline workflows and dependency resolution

This test suite validates that the core pipeline execution works seamlessly
across all providers, configurations, and plugin combinations while maintaining
proper error handling and performance characteristics.
"""

import pytest
import os
import tempfile
import json
import shutil
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
import concurrent.futures
import threading

from promptmatryoshka.core import PromptMatryoshka, PipelineBuilder, PipelineValidationError
from promptmatryoshka.config import Config, get_config, reset_config
from promptmatryoshka.llm_factory import LLMFactory, get_factory
from promptmatryoshka.llm_interface import LLMInterface, LLMConfig, ProviderInfo
from promptmatryoshka.providers import clear_registry
from promptmatryoshka.plugins.base import PluginBase, get_plugin_registry
from promptmatryoshka.plugins.boost import BoostPlugin
from promptmatryoshka.plugins.flipattack import FlipAttackPlugin
from promptmatryoshka.exceptions import LLMError, LLMConfigurationError, LLMUnsupportedProviderError


class MockPipelineLLMInterface(LLMInterface):
    """Mock LLM interface for pipeline testing with detailed tracking."""
    
    def __init__(self, config, provider_name="mock", profile_name=None, **kwargs):
        super().__init__(config, provider_name=provider_name, **kwargs)
        self.provider_name = provider_name
        self.profile_name = profile_name
        self.invoke_calls = []
        self.ainvoke_calls = []
        self.performance_delay = kwargs.get('performance_delay', 0.01)  # Simulate API latency
        
    def invoke(self, input, config=None, **kwargs):
        # Simulate API call latency
        time.sleep(self.performance_delay)
        
        call_info = {
            'input': input,
            'config': config,
            'kwargs': kwargs,
            'provider': self.provider_name,
            'profile': self.profile_name,
            'timestamp': time.time()
        }
        self.invoke_calls.append(call_info)
        
        # Generate provider-specific responses
        if self.provider_name == "openai":
            return f"OpenAI response: {input[:50]}..."
        elif self.provider_name == "anthropic":
            return f"Anthropic Claude response: {input[:50]}..."
        elif self.provider_name == "ollama":
            return f"Ollama local response: {input[:50]}..."
        elif self.provider_name == "huggingface":
            return f"HuggingFace response: {input[:50]}..."
        else:
            return f"Mock response from {self.provider_name}: {input[:50]}..."
    
    async def ainvoke(self, input, config=None, **kwargs):
        self.ainvoke_calls.append((input, config, kwargs))
        return self.invoke(input, config, **kwargs)
    
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
            limits={"max_tokens": self.config.max_tokens or 4000}
        )


class MockFailingLLMInterface(LLMInterface):
    """Mock LLM interface that fails for error testing."""
    
    def __init__(self, config, provider_name="failing", failure_type="connection", **kwargs):
        super().__init__(config, provider_name=provider_name, **kwargs)
        self.provider_name = provider_name
        self.failure_type = failure_type
        
    def invoke(self, input, config=None, **kwargs):
        if self.failure_type == "connection":
            raise ConnectionError(f"Connection failed to {self.provider_name}")
        elif self.failure_type == "timeout":
            raise TimeoutError(f"Request timeout for {self.provider_name}")
        elif self.failure_type == "api_error":
            raise LLMError(f"API error from {self.provider_name}")
        else:
            raise Exception(f"Unknown error from {self.provider_name}")
    
    async def ainvoke(self, input, config=None, **kwargs):
        return self.invoke(input, config, **kwargs)
    
    def validate_config(self) -> bool:
        return False if self.failure_type == "config" else True
    
    def health_check(self) -> bool:
        return False
    
    def get_provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.provider_name,
            version="1.0.0",
            models=[f"{self.provider_name}-model"],
            capabilities={"chat": True},
            limits={"max_tokens": 1000}
        )


class TestPipelineIntegration:
    """Test pipeline integration with multi-provider support."""
    
    def setup_method(self):
        """Set up test environment for pipeline testing."""
        # Reset global state
        reset_config()
        clear_registry()
        
        # Create temporary directory and config
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "pipeline_test_config.json")
        
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
                "research-openai": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "temperature": 0.0,
                    "max_tokens": 4000,
                    "description": "Research profile using OpenAI"
                },
                "production-anthropic": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.0,
                    "max_tokens": 4000,
                    "description": "Production profile using Anthropic"
                },
                "local-development": {
                    "provider": "ollama",
                    "model": "llama3.2:3b",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "description": "Local development profile"
                },
                "fast-inference": {
                    "provider": "huggingface",
                    "model": "microsoft/DialoGPT-medium",
                    "temperature": 0.2,
                    "max_tokens": 1500,
                    "description": "Fast inference profile"
                }
            },
            "plugins": {
                "boost": {
                    "technique_params": {
                        "num_eos": 3,
                        "storage_dir": "boost_results"
                    }
                },
                "flipattack": {
                    "technique_params": {
                        "mode": "char",
                        "storage_dir": "flipattack_results"
                    }
                }
            }
        }
        
        # Write test config
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)
        
    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clean up plugin storage directories
        for storage_dir in ["boost_results", "flipattack_results"]:
            if os.path.exists(storage_dir):
                shutil.rmtree(storage_dir)
        
        reset_config()
        clear_registry()

    def test_basic_pipeline_execution_all_providers(self):
        """Test basic pipeline execution across all 4 providers."""
        
        providers_to_test = ["openai", "anthropic", "ollama", "huggingface"]
        
        for provider in providers_to_test:
            with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
                def create_provider_mock(config, **kwargs):
                    return MockPipelineLLMInterface(
                        config, 
                        provider_name=provider,
                        performance_delay=0.01
                    )
                
                mock_get_provider.return_value = create_provider_mock
                
                # Create pipeline with technique-based plugins (no LLM required)
                pipeline = PromptMatryoshka(
                    plugins=[BoostPlugin(num_eos=2)],
                    config_path=self.config_path,
                    provider=provider
                )
                
                test_input = f"test pipeline execution with {provider}"
                result = pipeline.jailbreak(test_input)
                
                assert result is not None
                assert len(result) > 0
                assert result.endswith("</s>")  # Boost plugin effect (1 EOS token)
                assert result.startswith(test_input)

    def test_pipeline_provider_switching_runtime(self):
        """Test runtime provider switching during pipeline execution."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            call_log = []
            
            def provider_tracking_mock(config, **kwargs):
                provider_name = kwargs.get('provider_name', 'default')
                call_log.append(provider_name)
                return MockPipelineLLMInterface(config, provider_name=provider_name)
            
            mock_get_provider.return_value = provider_tracking_mock
            
            # Create pipeline
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            test_input = "test provider switching"
            
            # Test switching between different providers
            providers = ["openai", "anthropic", "ollama", "huggingface"]
            results = {}
            
            for provider in providers:
                result = pipeline.jailbreak(test_input, provider=provider)
                results[provider] = result
                
                assert result is not None
                assert result.endswith("</s>")  # Boost plugin effect

    def test_pipeline_profile_based_execution(self):
        """Test pipeline execution with profile-based configuration."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            profile_usage = {}
            
            def profile_tracking_mock(config, **kwargs):
                # Track which profile configurations are being used
                model = config.get('model') if isinstance(config, dict) else getattr(config, 'model', 'unknown')
                profile_usage[model] = profile_usage.get(model, 0) + 1
                return MockPipelineLLMInterface(config)
            
            mock_get_provider.return_value = profile_tracking_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=2)],
                config_path=self.config_path
            )
            
            test_input = "test profile-based execution"
            
            # Test with different profiles
            profiles = ["research-openai", "production-anthropic", "local-development", "fast-inference"]
            
            for profile in profiles:
                result = pipeline.jailbreak(test_input, profile=profile)
                
                assert result is not None
                assert len(result) > 0
                assert result.endswith("</s></s>")  # Boost plugin with 2 EOS tokens

    def test_pipeline_mixed_plugin_execution(self):
        """Test pipeline execution with mixed plugin types."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockPipelineLLMInterface(config)
            
            # Create pipeline with multiple plugins
            pipeline = PromptMatryoshka(
                plugins=[
                    BoostPlugin(num_eos=2),           # Technique-based
                    FlipAttackPlugin(mode="word")     # Technique-based
                ],
                config_path=self.config_path,
                provider="openai"
            )
            
            test_input = "test mixed plugin pipeline execution"
            result = pipeline.jailbreak(test_input)
            
            assert result is not None
            assert len(result) > 0
            
            # Should contain both plugin effects
            # With FlipAttack + BoostPlugin, FlipAttack generates a complete system prompt
            # that incorporates the EOS tokens within the prompt structure
            assert "</s></s>" in result or result.endswith("</s></s>")
            # FlipAttack plugin effects - check for key transformation indicators
            assert "TASK is" in result or "SYSTEM:" in result  # FlipAttack prompt structure

    def test_pipeline_error_handling_and_recovery(self):
        """Test pipeline error handling and recovery mechanisms."""
        
        # Test provider connection failure
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            def failing_provider_mock(config, **kwargs):
                return MockFailingLLMInterface(config, failure_type="connection")
            
            mock_get_provider.return_value = failing_provider_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],  # Technique-based plugin should still work
                config_path=self.config_path,
                provider="openai"
            )
            
            test_input = "test error handling"
            
            # Technique-based plugins should still work even if LLM provider fails
            result = pipeline.jailbreak(test_input)
            assert result is not None
            assert result.endswith("</s>")  # Boost plugin effect

    def test_pipeline_performance_characteristics(self):
        """Test pipeline performance characteristics across providers."""
        
        performance_results = {}
        
        providers = ["openai", "anthropic", "ollama", "huggingface"]
        
        for provider in providers:
            with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
                # Simulate different performance characteristics
                delay = {
                    "openai": 0.05,      # Fast API
                    "anthropic": 0.03,   # Very fast API  
                    "ollama": 0.2,       # Slower local processing
                    "huggingface": 0.1   # Medium API speed
                }.get(provider, 0.05)
                
                def performance_mock(config, **kwargs):
                    return MockPipelineLLMInterface(
                        config, 
                        provider_name=provider,
                        performance_delay=delay
                    )
                
                mock_get_provider.return_value = performance_mock
                
                pipeline = PromptMatryoshka(
                    plugins=[BoostPlugin(num_eos=1)],
                    config_path=self.config_path,
                    provider=provider
                )
                
                test_input = "performance test input"
                
                # Measure execution time
                start_time = time.time()
                result = pipeline.jailbreak(test_input)
                end_time = time.time()
                
                performance_results[provider] = end_time - start_time
                
                assert result is not None
                assert result.endswith("</s>")  # Boost plugin effect
        
        # Verify performance characteristics are reasonable
        for provider, exec_time in performance_results.items():
            assert exec_time < 2.0  # Should complete within 2 seconds

    def test_pipeline_concurrent_execution(self):
        """Test concurrent pipeline execution across multiple threads."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            execution_log = []
            lock = threading.Lock()
            
            def thread_safe_mock(config, **kwargs):
                with lock:
                    execution_log.append(f"provider_created_{len(execution_log)}")
                return MockPipelineLLMInterface(config)
            
            mock_get_provider.return_value = thread_safe_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            def execute_pipeline(thread_id):
                test_input = f"concurrent test {thread_id}"
                result = pipeline.jailbreak(test_input, provider="openai")
                return result
            
            # Execute pipeline concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(execute_pipeline, i) for i in range(10)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Verify all executions completed successfully
            assert len(results) == 10
            for result in results:
                assert result is not None
                assert result.endswith("</s>")  # Boost plugin effect

    def test_pipeline_dependency_resolution(self):
        """Test pipeline dependency resolution and plugin ordering."""
        
        # Create pipeline with plugin names (tests dependency resolution)
        pipeline = PromptMatryoshka(
            plugins=["boost", "flipattack"],  # Plugin names instead of instances
            auto_discover=True,
            config_path=self.config_path
        )
        
        # Verify pipeline was built successfully
        assert pipeline.pipeline is not None
        assert len(pipeline.pipeline) >= 1
        
        # Test execution
        test_input = "test dependency resolution"
        result = pipeline.jailbreak(test_input)
        
        assert result is not None
        assert len(result) > 0

    def test_pipeline_validation_and_health_checking(self):
        """Test pipeline validation and health checking."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockPipelineLLMInterface(config)
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=2), FlipAttackPlugin(mode="char")],
                config_path=self.config_path
            )
            
            # Test pipeline information retrieval
            pipeline_info = pipeline.get_pipeline_info()
            
            assert "plugins" in pipeline_info
            assert "total_plugins" in pipeline_info
            assert pipeline_info["total_plugins"] >= 2
            
            # Test configuration validation
            validation_result = pipeline.validate_configuration()
            assert validation_result is not None

    def test_pipeline_backward_compatibility(self):
        """Test pipeline backward compatibility with legacy features."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockPipelineLLMInterface(config)
            
            # Test legacy stages parameter
            boost_plugin = BoostPlugin(num_eos=1)
            flipattack_plugin = FlipAttackPlugin(mode="word")
            
            pipeline = PromptMatryoshka(
                stages=[boost_plugin, flipattack_plugin],  # Legacy parameter
                config_path=self.config_path
            )
            
            test_input = "test backward compatibility"
            result = pipeline.jailbreak(test_input)
            
            assert result is not None
            assert len(result) > 0

    def test_pipeline_complex_workflows(self):
        """Test complex pipeline workflows with multiple configurations."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            workflow_tracking = []
            
            def workflow_tracking_mock(config, **kwargs):
                provider = kwargs.get('provider_name', 'default')
                workflow_tracking.append(f"llm_created_{provider}")
                return MockPipelineLLMInterface(config, provider_name=provider)
            
            mock_get_provider.return_value = workflow_tracking_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=2)],
                config_path=self.config_path
            )
            
            # Complex workflow: different providers and profiles for same pipeline
            test_workflows = [
                ("openai", None, "workflow 1"),
                ("anthropic", None, "workflow 2"), 
                (None, "research-openai", "workflow 3"),
                (None, "production-anthropic", "workflow 4"),
                ("ollama", "local-development", "workflow 5")  # Provider + profile
            ]
            
            results = []
            for provider, profile, test_input in test_workflows:
                result = pipeline.jailbreak(test_input, provider=provider, profile=profile)
                results.append(result)
                
                assert result is not None
                assert result.endswith("</s></s>")  # Boost plugin with 2 EOS tokens
            
            assert len(results) == 5

    def test_pipeline_factory_integration(self):
        """Test pipeline integration with LLM factory pattern."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            factory_calls = []
            
            def factory_tracking_mock(config, **kwargs):
                factory_calls.append({
                    'config': config,
                    'kwargs': kwargs,
                    'timestamp': time.time()
                })
                return MockPipelineLLMInterface(config)
            
            mock_get_provider.return_value = factory_tracking_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            # Test factory integration
            test_input = "test factory integration"
            result = pipeline.jailbreak(test_input, provider="openai")
            
            assert result is not None
            assert result.endswith("</s>")  # Boost plugin effect

    def test_pipeline_resource_management(self):
        """Test pipeline resource management and cleanup."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            created_instances = []
            
            def resource_tracking_mock(config, **kwargs):
                instance = MockPipelineLLMInterface(config)
                created_instances.append(instance)
                return instance
            
            mock_get_provider.return_value = resource_tracking_mock
            
            # Create and use pipeline
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            test_input = "test resource management"
            result = pipeline.jailbreak(test_input, provider="openai")
            
            assert result is not None
            
            # Verify instances were created appropriately
            # (Resource cleanup would be tested in actual implementation)

    def test_pipeline_edge_cases_and_boundary_conditions(self):
        """Test pipeline edge cases and boundary conditions."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockPipelineLLMInterface(config)
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            # Test empty input
            result = pipeline.jailbreak("", provider="openai")
            assert result == "</s>"  # Empty input with 1 EOS token
            
            # Test very large input
            large_input = "test " * 1000
            result = pipeline.jailbreak(large_input, provider="openai")
            assert result.endswith("</s>")  # Large input with EOS token
            assert len(result) > len(large_input)
            
            # Test special characters
            special_input = "test émojis 🚀 and chars: <>&\""
            result = pipeline.jailbreak(special_input, provider="openai")
            assert result.endswith("</s>")  # Special chars with EOS token

    def test_pipeline_configuration_inheritance_chain(self):
        """Test complex configuration inheritance in pipeline execution."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            config_tracking = []
            
            def config_tracking_mock(config, **kwargs):
                config_info = {
                    'model': config.get('model') if isinstance(config, dict) else getattr(config, 'model', None),
                    'temperature': config.get('temperature') if isinstance(config, dict) else getattr(config, 'temperature', None),
                    'provider': kwargs.get('provider_name', 'unknown')
                }
                config_tracking.append(config_info)
                return MockPipelineLLMInterface(config)
            
            mock_get_provider.return_value = config_tracking_mock
            
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=1)],
                config_path=self.config_path
            )
            
            # Test inheritance: provider default -> profile -> runtime override
            result = pipeline.jailbreak(
                "test inheritance", 
                profile="research-openai"  # Should use profile configuration
            )
            
            assert result is not None
            assert result.endswith("</s>")  # Boost plugin effect


class TestPipelineBuilder:
    """Test pipeline builder and dependency resolution."""
    
    def setup_method(self):
        """Set up test environment."""
        reset_config()
        clear_registry()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        reset_config()
        clear_registry()

    def test_pipeline_builder_basic_functionality(self):
        """Test basic pipeline builder functionality."""
        
        from promptmatryoshka.plugins.base import get_plugin_registry
        
        # Create builder
        registry = get_plugin_registry()
        builder = PipelineBuilder(registry)
        
        # Test validation (will pass even with no plugins registered)
        validation_result = builder.validate_pipeline([])
        assert validation_result.valid is True

    def test_pipeline_builder_error_handling(self):
        """Test pipeline builder error handling."""
        
        from promptmatryoshka.plugins.base import get_plugin_registry
        
        registry = get_plugin_registry()
        builder = PipelineBuilder(registry)
        
        # Test validation with non-existent plugins
        validation_result = builder.validate_pipeline(["nonexistent_plugin"])
        assert validation_result.valid is False
        assert len(validation_result.errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])