"""Comprehensive integration tests for multi-provider LLM framework.

Tests the complete integration across all system components:
- Multi-provider LLM interface creation and management
- Configuration profile integration with providers
- Plugin system integration with multi-provider support
- CLI command integration and workflows
- Error handling and edge cases
- Performance and reliability characteristics

This test suite validates that the entire system works together seamlessly
across all 4 providers (OpenAI, Anthropic, Ollama, HuggingFace) and
6 configuration profiles.
"""

import pytest
import os
import tempfile
import json
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from promptmatryoshka.llm_factory import LLMFactory, get_factory
from promptmatryoshka.config import Config, get_config, reset_config
from promptmatryoshka.core import PromptMatryoshka
from promptmatryoshka.llm_interface import LLMInterface, LLMConfig, ProviderInfo
from promptmatryoshka.providers import (
    register_provider, get_provider, list_providers, 
    is_provider_available, discover_providers, clear_registry
)
from promptmatryoshka.exceptions import (
    LLMError, LLMConfigurationError, LLMUnsupportedProviderError, LLMValidationError
)


class MockLLMInterface(LLMInterface):
    """Mock LLM interface for integration testing."""
    
    def __init__(self, config, provider_name="mock", **kwargs):
        # Ensure config has required model field
        if isinstance(config, dict) and 'model' not in config:
            config['model'] = f"{provider_name}-default-model"
        super().__init__(config, provider_name=provider_name, **kwargs)
        self.invoke_calls = []
        self.ainvoke_calls = []
        self.provider_name = provider_name
        
    def invoke(self, input, config=None, **kwargs):
        self.invoke_calls.append((input, config, kwargs))
        return f"Mock response from {self.provider_name}: {input[:50]}..."
    
    async def ainvoke(self, input, config=None, **kwargs):
        self.ainvoke_calls.append((input, config, kwargs))
        return f"Mock async response from {self.provider_name}: {input[:50]}..."
    
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
            limits={"max_tokens": 4000, "context_window": 8000}
        )


class TestMultiProviderIntegration:
    """Integration tests for multi-provider system."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        # Reset global state
        reset_config()
        clear_registry()
        
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        
        # Create test configuration with all providers and profiles
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
                    "description": "High-quality research profile using OpenAI GPT-4o"
                },
                "production-anthropic": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.0,
                    "max_tokens": 4000,
                    "description": "Production profile using Anthropic Claude"
                },
                "local-development": {
                    "provider": "ollama",
                    "model": "llama3.2:3b",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "description": "Local development profile using Ollama"
                },
                "fast-gpt35": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.0,
                    "max_tokens": 2000,
                    "description": "Fast and cost-effective profile using GPT-3.5"
                },
                "creative-anthropic": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "description": "Creative profile using Anthropic Claude with higher temperature"
                },
                "local-llama": {
                    "provider": "ollama",
                    "model": "llama3.2:8b",
                    "temperature": 0.2,
                    "max_tokens": 3000,
                    "description": "Local Llama 8B model for more capable local processing"
                }
            },
            "plugins": {
                "logitranslate": {
                    "profile": "research-openai",
                    "technique_params": {
                        "validation_enabled": True,
                        "max_attempts": 3
                    }
                },
                "logiattack": {
                    "profile": "research-openai",
                    "technique_params": {
                        "validation_enabled": True,
                        "schema_strict": True
                    }
                },
                "judge": {
                    "profile": "production-anthropic",
                    "technique_params": {
                        "threshold": 0.8,
                        "multi_judge": True
                    }
                }
            }
        }
        
        # Write test config
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)
        
        # Mock all provider classes to return MockLLMInterface
        self.provider_patches = []
        for provider_name in ["openai", "anthropic", "ollama", "huggingface"]:
            def create_mock_provider(name=provider_name):
                def mock_provider_class(config, **kwargs):
                    return MockLLMInterface(config, provider_name=name, **kwargs)
                return mock_provider_class
            
            patch_obj = patch(f'promptmatryoshka.providers.get_provider') 
            mock_get_provider = patch_obj.start()
            mock_get_provider.return_value = create_mock_provider()
            self.provider_patches.append(patch_obj)
        
        # Mock provider availability
        self.availability_patch = patch('promptmatryoshka.providers.is_provider_available')
        mock_availability = self.availability_patch.start()
        mock_availability.return_value = True
        
        # Initialize config and factory
        self.config = Config(self.config_path)
        self.factory = LLMFactory()
        
        # Register test profiles with factory
        for profile_name, profile_config in self.test_config["profiles"].items():
            self.factory.register_profile(profile_name, profile_config)
        
    def teardown_method(self):
        """Clean up after each test."""
        # Stop all patches
        for patch_obj in self.provider_patches:
            patch_obj.stop()
        self.availability_patch.stop()
        
        # Clean up temporary files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Reset global state
        reset_config()
        clear_registry()

    def test_all_providers_available_and_discoverable(self):
        """Test that all 4 providers can be discovered and are available."""
        # Test provider discovery
        providers = discover_providers()
        
        expected_providers = ["openai", "anthropic", "ollama", "huggingface"]
        for provider in expected_providers:
            assert provider in providers
            assert providers[provider]["available"] is True
        
        # Test individual provider availability
        for provider in expected_providers:
            assert is_provider_available(provider) is True

    def test_all_configuration_profiles_creation(self):
        """Test that all 6 configuration profiles can create LLM interfaces."""
        expected_profiles = [
            "research-openai", "production-anthropic", "local-development",
            "fast-gpt35", "creative-anthropic", "local-llama"
        ]
        
        for profile_name in expected_profiles:
            with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
                def create_mock_for_profile(config, **kwargs):
                    return MockLLMInterface(
                        config, provider_name=self.test_config["profiles"][profile_name]["provider"]
                    )
                mock_get_provider.return_value = create_mock_for_profile
                
                # Test profile creation through factory
                interface = self.factory.create_from_profile(profile_name)
                
                assert interface is not None
                assert isinstance(interface, MockLLMInterface)
                assert interface.provider_name == self.test_config["profiles"][profile_name]["provider"]
                assert interface.config.model == self.test_config["profiles"][profile_name]["model"]
                assert interface.config.temperature == self.test_config["profiles"][profile_name]["temperature"]

    def test_provider_switching_and_fallback(self):
        """Test provider switching and fallback mechanisms."""
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            # Mock different providers
            def provider_factory(config, **kwargs):
                provider_name = kwargs.get('provider_name', 'default')
                return MockLLMInterface(config, provider_name=provider_name)
            
            mock_get_provider.return_value = provider_factory
            
            # Test switching between providers
            providers_to_test = ["openai", "anthropic", "ollama", "huggingface"]
            
            for provider in providers_to_test:
                config = {"model": f"{provider}-model", "temperature": 0.0}
                interface = self.factory.create_interface(provider, config)
                
                assert interface is not None
                assert isinstance(interface, MockLLMInterface)

    def test_plugin_integration_with_multi_provider_support(self):
        """Test plugin system integration with multi-provider configurations."""
        from promptmatryoshka.plugins.flipattack import FlipAttackPlugin
        from promptmatryoshka.plugins.boost import BoostPlugin
        
        # Test pipeline with different providers for different plugins
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockLLMInterface(config)
            
            # Create pipeline with plugins
            pipeline = PromptMatryoshka(
                plugins=[FlipAttackPlugin(), BoostPlugin()],
                config_path=self.config_path
            )
            
            # Test running with different providers
            test_input = "test prompt for multi-provider pipeline"
            
            # Test with different provider options
            result1 = pipeline.jailbreak(test_input, provider="openai")
            assert result1 is not None
            assert len(result1) > 0
            
            result2 = pipeline.jailbreak(test_input, provider="anthropic") 
            assert result2 is not None
            assert len(result2) > 0

    def test_configuration_validation_and_error_handling(self):
        """Test comprehensive configuration validation and error handling."""
        # Test invalid provider reference in profile
        invalid_config = self.test_config.copy()
        invalid_config["profiles"]["invalid-profile"] = {
            "provider": "nonexistent-provider",
            "model": "test-model",
            "temperature": 0.0,
            "max_tokens": 1000
        }
        
        invalid_config_path = os.path.join(self.temp_dir, "invalid_config.json")
        with open(invalid_config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        # Should raise configuration error due to invalid provider reference
        with pytest.raises(Exception):  # Could be ConfigurationError or ValueError
            Config(invalid_config_path)

    def test_provider_health_checking_and_validation(self):
        """Test provider health checking and validation functionality."""
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            # Mock healthy provider
            healthy_interface = MockLLMInterface({"model": "test"})
            healthy_interface.health_check = lambda: True
            
            # Mock unhealthy provider  
            unhealthy_interface = MockLLMInterface({"model": "test"})
            unhealthy_interface.health_check = lambda: False
            
            def provider_factory(config, **kwargs):
                if config.get("healthy", True):
                    return healthy_interface
                else:
                    return unhealthy_interface
            
            mock_get_provider.return_value = provider_factory
            
            # Test healthy provider
            healthy_config = {"model": "test", "healthy": True}
            interface = self.factory.create_interface("openai", healthy_config)
            assert interface.health_check() is True
            
            # Test unhealthy provider
            unhealthy_config = {"model": "test", "healthy": False}
            interface = self.factory.create_interface("openai", unhealthy_config)
            assert interface.health_check() is False

    def test_pipeline_integration_with_provider_overrides(self):
        """Test pipeline execution with provider and profile overrides."""
        from promptmatryoshka.plugins.boost import BoostPlugin
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockLLMInterface(config)
            
            # Create pipeline
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin()],
                config_path=self.config_path
            )
            
            test_prompt = "test prompt"
            
            # Test with provider override
            result1 = pipeline.jailbreak(test_prompt, provider="anthropic")
            assert result1 is not None
            
            # Test with profile override
            result2 = pipeline.jailbreak(test_prompt, profile="research-openai")
            assert result2 is not None
            
            # Test with both provider and profile (profile should take precedence)
            result3 = pipeline.jailbreak(test_prompt, provider="ollama", profile="production-anthropic")
            assert result3 is not None

    def test_error_handling_and_recovery_mechanisms(self):
        """Test comprehensive error handling and recovery mechanisms."""
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            # Mock provider that raises errors
            def failing_provider(config, **kwargs):
                raise Exception("Provider initialization failed")
            
            mock_get_provider.return_value = failing_provider
            
            # Test error handling in factory
            with pytest.raises(LLMUnsupportedProviderError):
                self.factory.create_interface("failing-provider", {"model": "test"})
            
            # Test error handling in profile creation
            with pytest.raises(Exception):
                self.factory.create_from_profile("research-openai")

    def test_concurrent_provider_operations(self):
        """Test concurrent operations across multiple providers."""
        import concurrent.futures
        import threading
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            # Thread-safe mock provider
            call_counts = {}
            lock = threading.Lock()
            
            def thread_safe_provider(config, **kwargs):
                provider = kwargs.get('provider_name', 'default')
                with lock:
                    call_counts[provider] = call_counts.get(provider, 0) + 1
                return MockLLMInterface(config, provider_name=provider)
            
            mock_get_provider.return_value = thread_safe_provider
            
            def create_interface_worker(provider):
                return self.factory.create_interface(provider, {"model": "test"})
            
            # Test concurrent creation of interfaces for different providers
            providers = ["openai", "anthropic", "ollama", "huggingface"]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(create_interface_worker, provider) for provider in providers]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Verify all interfaces were created successfully
            assert len(results) == 4
            for result in results:
                assert isinstance(result, MockLLMInterface)

    def test_performance_characteristics_and_caching(self):
        """Test performance characteristics and caching mechanisms."""
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            creation_count = 0
            
            def counting_provider(config, **kwargs):
                nonlocal creation_count
                creation_count += 1
                return MockLLMInterface(config)
            
            mock_get_provider.return_value = counting_provider
            
            config = {"model": "test", "temperature": 0.0}
            
            # Test caching behavior
            interface1 = self.factory.create_interface("openai", config, use_cache=True)
            interface2 = self.factory.create_interface("openai", config, use_cache=True)
            
            # Should use cached instance
            assert interface1 is interface2
            assert creation_count == 1
            
            # Test cache bypass
            interface3 = self.factory.create_interface("openai", config, use_cache=False)
            assert interface3 is not interface1
            assert creation_count == 2

    def test_plugin_specific_llm_creation(self):
        """Test plugin-specific LLM creation and configuration."""
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockLLMInterface(config)
            
            # Test plugin-specific configurations
            plugin_configs = ["logitranslate", "logiattack", "judge"]
            
            for plugin_name in plugin_configs:
                plugin_config = self.test_config["plugins"][plugin_name]
                expected_profile = plugin_config["profile"]
                
                # Create enhanced plugin config with model
                enhanced_config = plugin_config.copy()
                enhanced_config.update({
                    "model": f"{plugin_name}-model",
                    "temperature": 0.0,
                    "max_tokens": 2000
                })
                
                # Test using create_for_plugin method
                interface = self.factory.create_for_plugin(plugin_name, enhanced_config)
                
                assert interface is not None
                assert isinstance(interface, MockLLMInterface)

    def test_configuration_inheritance_and_overrides(self):
        """Test configuration inheritance and override mechanisms."""
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockLLMInterface(config)
            
            # Test profile with overrides
            overrides = {
                "temperature": 0.8,
                "max_tokens": 1000
            }
            
            interface = self.factory.create_from_profile("research-openai", overrides=overrides)
            
            assert interface.config.temperature == 0.8
            assert interface.config.max_tokens == 1000
            # Other profile values should be preserved
            assert interface.config.model == "gpt-4o"

    def test_environment_variable_resolution(self):
        """Test environment variable resolution in configuration."""
        # Test with environment variables set
        test_env = {
            "TEST_OPENAI_KEY": "env-openai-key",
            "TEST_ANTHROPIC_KEY": "env-anthropic-key"
        }
        
        env_config = {
            "providers": {
                "openai": {
                    "api_key": "${TEST_OPENAI_KEY}",
                    "default_model": "gpt-4"
                },
                "anthropic": {
                    "api_key": "${TEST_ANTHROPIC_KEY}",
                    "default_model": "claude-3"
                }
            },
            "profiles": {
                "test-profile": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.0,
                    "max_tokens": 1000
                }
            }
        }
        
        env_config_path = os.path.join(self.temp_dir, "env_config.json")
        with open(env_config_path, 'w') as f:
            json.dump(env_config, f)
        
        with patch.dict(os.environ, test_env):
            config = Config(env_config_path)
            
            # Test that environment variables were resolved
            openai_config = config.get_provider_config("openai")
            assert openai_config.api_key == "env-openai-key"
            
            anthropic_config = config.get_provider_config("anthropic")
            assert anthropic_config.api_key == "env-anthropic-key"


class TestCLIIntegration:
    """Integration tests for CLI commands with multi-provider support."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_cli_provider_commands_integration(self):
        """Test CLI provider management commands integration."""
        # This would test the 8 new CLI commands:
        # list-providers, check-provider, test-provider, list-profiles,
        # show-profile, validate-config, show-config, config-health
        
        # Test list-providers command integration
        with patch('promptmatryoshka.providers.discover_providers') as mock_discover:
            mock_discover.return_value = {
                "openai": {"available": True, "version": "1.0.0"},
                "anthropic": {"available": True, "version": "1.0.0"}
            }
            
            from promptmatryoshka.cli import list_providers_command
            from types import SimpleNamespace
            
            # Create mock args for the CLI command
            mock_args = SimpleNamespace(json=False)
            
            # Test should not raise an exception
            try:
                list_providers_command(mock_args)
                # If we reach here, the command executed successfully
                success = True
            except Exception as e:
                success = False
                print(f"CLI command failed: {e}")
            
            assert success is True

    def test_cli_run_command_with_provider_options(self):
        """Test CLI run command with provider and profile options."""
        # Test the enhanced run command with --provider and --profile options
        from promptmatryoshka.plugins.boost import BoostPlugin
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockLLMInterface(config)
            
            # Test running with provider option
            pipeline = PromptMatryoshka(plugins=[BoostPlugin()])
            result = pipeline.jailbreak("test input", provider="openai")
            
            assert result is not None
            assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])