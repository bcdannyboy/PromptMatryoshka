"""Integration tests for configuration profiles with multi-provider support.

Tests all 6 predefined configuration profiles:
- research-openai: High-quality research profile using OpenAI GPT-4o
- production-anthropic: Production profile using Anthropic Claude
- local-development: Local development profile using Ollama
- fast-gpt35: Fast and cost-effective profile using GPT-3.5
- creative-anthropic: Creative profile with higher temperature
- local-llama: Local Llama 8B model for capable local processing

This test suite validates:
- Profile-based LLM creation and configuration inheritance
- Provider-specific parameter handling and validation
- Profile switching and runtime configuration
- Error handling for invalid profiles and configurations
- Performance characteristics across different profiles
"""

import pytest
import os
import tempfile
import json
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from promptmatryoshka.config import Config, get_config, reset_config, ProfileConfig
from promptmatryoshka.llm_factory import LLMFactory, get_factory
from promptmatryoshka.core import PromptMatryoshka
from promptmatryoshka.llm_interface import LLMInterface, LLMConfig, ProviderInfo
from promptmatryoshka.providers import clear_registry
from promptmatryoshka.exceptions import LLMConfigurationError, LLMValidationError


class MockProfileLLMInterface(LLMInterface):
    """Mock LLM interface for profile testing that tracks configuration."""
    
    def __init__(self, config, provider_name="mock", profile_name=None, **kwargs):
        super().__init__(config, provider_name=provider_name, **kwargs)
        self.provider_name = provider_name
        self.profile_name = profile_name
        self.invoke_calls = []
        
    def invoke(self, input, config=None, **kwargs):
        self.invoke_calls.append((input, config, kwargs))
        response = f"Profile response from {self.profile_name or 'default'} using {self.provider_name}: {input[:40]}..."
        return response
    
    async def ainvoke(self, input, config=None, **kwargs):
        return await self.invoke(input, config, **kwargs)
    
    def validate_config(self) -> bool:
        return True
    
    def health_check(self) -> bool:
        return True
    
    def get_provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.provider_name,
            version="1.0.0",
            models=[f"{self.provider_name}-model"],
            capabilities={"chat": True, "completion": True, "function_calling": True},
            limits={"max_tokens": self.config.max_tokens or 4000}
        )


class TestConfigurationProfiles:
    """Test all configuration profiles with multi-provider integration."""
    
    def setup_method(self):
        """Set up test environment for profile testing."""
        # Reset global state
        reset_config()
        clear_registry()
        
        # Create temporary directory and config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "profiles_test_config.json")
        
        # Create comprehensive test configuration with all 6 profiles
        self.test_config = {
            "providers": {
                "openai": {
                    "api_key": "test-openai-key",
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-4o-mini",
                    "timeout": 120,
                    "max_retries": 3,
                    "rate_limit": {
                        "requests_per_minute": 500,
                        "tokens_per_minute": 150000
                    }
                },
                "anthropic": {
                    "api_key": "test-anthropic-key",
                    "base_url": "https://api.anthropic.com",
                    "default_model": "claude-3-5-sonnet-20241022",
                    "timeout": 120,
                    "max_retries": 3,
                    "rate_limit": {
                        "requests_per_minute": 100,
                        "tokens_per_minute": 50000
                    }
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
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "request_timeout": 120,
                    "description": "High-quality research profile using OpenAI GPT-4o"
                },
                "production-anthropic": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.0,
                    "max_tokens": 4000,
                    "top_p": 1.0,
                    "request_timeout": 120,
                    "description": "Production profile using Anthropic Claude"
                },
                "local-development": {
                    "provider": "ollama",
                    "model": "llama3.2:3b",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "top_p": 0.9,
                    "request_timeout": 300,
                    "description": "Local development profile using Ollama"
                },
                "fast-gpt35": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.0,
                    "max_tokens": 2000,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "request_timeout": 60,
                    "description": "Fast and cost-effective profile using GPT-3.5"
                },
                "creative-anthropic": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "top_p": 0.9,
                    "request_timeout": 120,
                    "description": "Creative profile using Anthropic Claude with higher temperature"
                },
                "local-llama": {
                    "provider": "ollama",
                    "model": "llama3.2:8b",
                    "temperature": 0.2,
                    "max_tokens": 3000,
                    "top_p": 0.8,
                    "request_timeout": 300,
                    "description": "Local Llama 8B model for more capable local processing"
                }
            },
            "plugins": {
                "logitranslate": {
                    "profile": "research-openai"
                },
                "judge": {
                    "profile": "production-anthropic"
                },
                "boost": {
                    "technique_params": {
                        "num_eos": 5
                    }
                }
            }
        }
        
        # Write test config
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)
        
        # Initialize config
        self.config = Config(self.config_path)
        self.factory = LLMFactory()
        
    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        reset_config()
        clear_registry()

    def test_all_six_profiles_creation(self):
        """Test that all 6 configuration profiles can create LLM interfaces successfully."""
        expected_profiles = [
            ("research-openai", "openai", "gpt-4o", 0.0, 4000),
            ("production-anthropic", "anthropic", "claude-3-5-sonnet-20241022", 0.0, 4000),
            ("local-development", "ollama", "llama3.2:3b", 0.1, 2000),
            ("fast-gpt35", "openai", "gpt-3.5-turbo", 0.0, 2000),
            ("creative-anthropic", "anthropic", "claude-3-5-sonnet-20241022", 0.7, 4000),
            ("local-llama", "ollama", "llama3.2:8b", 0.2, 3000)
        ]
        
        for profile_name, expected_provider, expected_model, expected_temp, expected_tokens in expected_profiles:
            with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
                def create_profile_mock(config, **kwargs):
                    return MockProfileLLMInterface(
                        config, 
                        provider_name=expected_provider,
                        profile_name=profile_name
                    )
                
                mock_get_provider.return_value = create_profile_mock
                
                # Test profile creation through factory
                interface = self.factory.create_from_profile(profile_name)
                
                assert interface is not None
                assert isinstance(interface, MockProfileLLMInterface)
                assert interface.provider_name == expected_provider
                assert interface.profile_name == profile_name
                assert interface.config.model == expected_model
                assert interface.config.temperature == expected_temp
                assert interface.config.max_tokens == expected_tokens

    def test_profile_parameter_inheritance_and_validation(self):
        """Test profile parameter inheritance and validation logic."""
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockProfileLLMInterface(config)
            
            # Test research-openai profile with all OpenAI-specific parameters
            interface = self.factory.create_from_profile("research-openai")
            
            assert interface.config.temperature == 0.0
            assert interface.config.max_tokens == 4000
            assert interface.config.top_p == 1.0
            assert interface.config.frequency_penalty == 0.0
            assert interface.config.presence_penalty == 0.0
            
            # Test creative-anthropic profile with higher temperature
            interface = self.factory.create_from_profile("creative-anthropic")
            
            assert interface.config.temperature == 0.7
            assert interface.config.top_p == 0.9
            
            # Test local profiles with longer timeouts
            interface = self.factory.create_from_profile("local-development")
            assert interface.config.request_timeout == 300
            
            interface = self.factory.create_from_profile("local-llama")
            assert interface.config.request_timeout == 300

    def test_profile_overrides_functionality(self):
        """Test profile parameter overrides functionality."""
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockProfileLLMInterface(config)
            
            # Test overriding temperature and max_tokens
            overrides = {
                "temperature": 0.5,
                "max_tokens": 1500,
                "top_p": 0.8
            }
            
            interface = self.factory.create_from_profile("research-openai", overrides=overrides)
            
            # Overridden values should be applied
            assert interface.config.temperature == 0.5
            assert interface.config.max_tokens == 1500
            assert interface.config.top_p == 0.8
            
            # Non-overridden values should remain from profile
            assert interface.config.model == "gpt-4o"
            assert interface.config.frequency_penalty == 0.0

    def test_profile_switching_runtime_behavior(self):
        """Test runtime profile switching behavior."""
        from promptmatryoshka.plugins.boost import BoostPlugin
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockProfileLLMInterface(config)
            
            # Create pipeline
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin()],
                config_path=self.config_path
            )
            
            test_input = "test prompt for profile switching"
            
            # Test switching between different profiles
            profiles_to_test = [
                "research-openai",
                "production-anthropic", 
                "local-development",
                "fast-gpt35",
                "creative-anthropic",
                "local-llama"
            ]
            
            for profile in profiles_to_test:
                result = pipeline.jailbreak(test_input, profile=profile)
                assert result is not None
                assert len(result) > 0
                # Should contain the modified input (boost plugin adds </s> tokens)
                assert result.endswith("</s></s></s></s></s>")

    def test_provider_specific_parameter_handling(self):
        """Test provider-specific parameter handling across profiles."""
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockProfileLLMInterface(config)
            
            # Test OpenAI-specific parameters (frequency_penalty, presence_penalty)
            openai_profiles = ["research-openai", "fast-gpt35"]
            for profile in openai_profiles:
                interface = self.factory.create_from_profile(profile)
                # OpenAI profiles should have penalty parameters
                assert hasattr(interface.config, 'frequency_penalty')
                assert hasattr(interface.config, 'presence_penalty')
                assert interface.config.frequency_penalty is not None
                assert interface.config.presence_penalty is not None
            
            # Test Anthropic profiles (no penalty parameters in config)
            anthropic_profiles = ["production-anthropic", "creative-anthropic"]
            for profile in anthropic_profiles:
                interface = self.factory.create_from_profile(profile)
                # Should still work without OpenAI-specific parameters
                assert interface.config.temperature is not None
                assert interface.config.max_tokens is not None
            
            # Test Ollama profiles (local deployment characteristics)
            ollama_profiles = ["local-development", "local-llama"]
            for profile in ollama_profiles:
                interface = self.factory.create_from_profile(profile)
                # Should have longer timeouts for local processing
                assert interface.config.request_timeout >= 300

    def test_profile_validation_and_error_handling(self):
        """Test profile validation and comprehensive error handling."""
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockProfileLLMInterface(config)
            
            # Test non-existent profile
            with pytest.raises(LLMConfigurationError):
                self.factory.create_from_profile("nonexistent-profile")
            
            # Test profile with invalid overrides
            invalid_overrides = {
                "temperature": 5.0,  # Invalid temperature > 2.0
                "max_tokens": -100,  # Invalid negative tokens
                "top_p": 2.0  # Invalid top_p > 1.0
            }
            
            with pytest.raises(LLMValidationError):
                self.factory.create_from_profile("research-openai", overrides=invalid_overrides)

    def test_plugin_profile_integration(self):
        """Test plugin integration with specific profiles."""
        from promptmatryoshka.plugins.boost import BoostPlugin
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockProfileLLMInterface(config)
            
            # Test plugin using profile-based LLM creation
            plugin_config = self.test_config["plugins"]["logitranslate"]
            expected_profile = plugin_config["profile"]  # "research-openai"
            
            # Create LLM for plugin using profile
            interface = self.factory.create_for_plugin("logitranslate", plugin_config)
            
            assert interface is not None
            assert isinstance(interface, MockProfileLLMInterface)
            
            # Test plugin execution with profile-configured LLM
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin()],
                config_path=self.config_path,
                profile="research-openai"
            )
            
            result = pipeline.jailbreak("test plugin with profile")
            assert result is not None

    def test_profile_performance_characteristics(self):
        """Test performance characteristics across different profiles."""
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            creation_times = {}
            
            def timing_mock(config, **kwargs):
                import time
                # Simulate different creation times for different providers
                provider = kwargs.get('provider_name', 'unknown')
                if provider == "ollama":
                    time.sleep(0.01)  # Simulate longer local startup
                elif provider in ["openai", "anthropic"]:
                    time.sleep(0.005)  # Simulate API connection time
                return MockProfileLLMInterface(config, provider_name=provider)
            
            mock_get_provider.return_value = timing_mock
            
            # Test creation time for each profile
            import time
            profiles = ["research-openai", "production-anthropic", "local-development", 
                       "fast-gpt35", "creative-anthropic", "local-llama"]
            
            for profile in profiles:
                start_time = time.time()
                interface = self.factory.create_from_profile(profile)
                end_time = time.time()
                creation_times[profile] = end_time - start_time
                
                assert interface is not None
            
            # Verify reasonable performance characteristics
            assert all(time < 1.0 for time in creation_times.values())  # All should be under 1 second

    def test_profile_caching_behavior(self):
        """Test profile caching and instance reuse behavior."""
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            creation_count = 0
            
            def counting_mock(config, **kwargs):
                nonlocal creation_count
                creation_count += 1
                return MockProfileLLMInterface(config)
            
            mock_get_provider.return_value = counting_mock
            
            # Test caching behavior with same profile
            interface1 = self.factory.create_from_profile("research-openai", use_cache=True)
            interface2 = self.factory.create_from_profile("research-openai", use_cache=True)
            
            # Should use cached instance for same profile
            assert interface1 is interface2
            assert creation_count == 1
            
            # Test different profile creates new instance
            interface3 = self.factory.create_from_profile("production-anthropic", use_cache=True)
            assert interface3 is not interface1
            assert creation_count == 2
            
            # Test cache bypass
            interface4 = self.factory.create_from_profile("research-openai", use_cache=False)
            assert interface4 is not interface1
            assert creation_count == 3

    def test_profile_configuration_inheritance_chain(self):
        """Test complex configuration inheritance chain from provider to profile to overrides."""
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockProfileLLMInterface(config)
            
            # Test inheritance chain: provider default -> profile -> overrides
            overrides = {"temperature": 0.9}
            
            interface = self.factory.create_from_profile("research-openai", overrides=overrides)
            
            # Override should take precedence
            assert interface.config.temperature == 0.9
            
            # Profile values should be used for non-overridden parameters
            assert interface.config.model == "gpt-4o"  # From profile
            assert interface.config.max_tokens == 4000  # From profile
            
            # Provider defaults should be inherited for unspecified parameters
            # (These would be handled by the actual provider implementation)

    def test_environment_variable_resolution_in_profiles(self):
        """Test environment variable resolution within profile configurations."""
        # Create config with environment variable references
        env_config = {
            "providers": {
                "openai": {
                    "api_key": "${TEST_OPENAI_API_KEY}",
                    "default_model": "gpt-4"
                }
            },
            "profiles": {
                "env-test-profile": {
                    "provider": "openai",
                    "model": "${TEST_MODEL_NAME}",
                    "temperature": 0.0,
                    "max_tokens": 2000
                }
            }
        }
        
        env_config_path = os.path.join(self.temp_dir, "env_config.json")
        with open(env_config_path, 'w') as f:
            json.dump(env_config, f)
        
        # Test with environment variables set
        test_env = {
            "TEST_OPENAI_API_KEY": "env-test-key",
            "TEST_MODEL_NAME": "gpt-4-env"
        }
        
        with patch.dict(os.environ, test_env):
            with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
                mock_get_provider.return_value = lambda config, **kwargs: MockProfileLLMInterface(config)
                
                env_config_obj = Config(env_config_path)
                env_factory = LLMFactory()
                
                # Register the profile with the factory
                profile_config = env_config_obj.get_profile_config("env-test-profile")
                if profile_config:
                    profile_dict = profile_config.model_dump()
                    env_factory.register_profile("env-test-profile", profile_dict)
                
                interface = env_factory.create_from_profile("env-test-profile")
                
                # Environment variables should be resolved
                assert interface.config.model == "gpt-4-env"

    def test_concurrent_profile_usage(self):
        """Test concurrent usage of different profiles."""
        import concurrent.futures
        import threading
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            call_counts = {}
            lock = threading.Lock()
            
            def thread_safe_mock(config, **kwargs):
                profile = kwargs.get('profile_name', 'default')
                with lock:
                    call_counts[profile] = call_counts.get(profile, 0) + 1
                return MockProfileLLMInterface(config, profile_name=profile)
            
            mock_get_provider.return_value = thread_safe_mock
            
            def create_profile_worker(profile_name):
                return self.factory.create_from_profile(profile_name)
            
            # Test concurrent creation of different profiles
            profiles = ["research-openai", "production-anthropic", "local-development", "fast-gpt35"]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(create_profile_worker, profile) for profile in profiles]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Verify all profiles were created successfully
            assert len(results) == 4
            for result in results:
                assert isinstance(result, MockProfileLLMInterface)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])