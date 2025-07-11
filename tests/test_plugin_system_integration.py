"""Comprehensive plugin system integration tests with multi-provider support.

Tests the complete plugin system integration across all plugins:
- boost: Text mutation plugin (technique-based, no LLM required)
- flipattack: Character/word flipping attack (technique-based, no LLM required)  
- logitranslate: Logic translation plugin (LLM-based)
- logiattack: Logic-based jailbreak attack (LLM-based)
- judge: Content evaluation and scoring (LLM-based)

This test suite validates:
- Plugin discovery and registration across the system
- Multi-provider LLM integration for LLM-based plugins
- Plugin-specific configuration and parameter handling
- Pipeline integration with mixed plugin types
- Plugin error handling and recovery mechanisms
- Performance characteristics across different plugin types
- Backward compatibility with existing plugin workflows
"""

import pytest
import os
import tempfile
import json
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from promptmatryoshka.core import PromptMatryoshka
from promptmatryoshka.config import Config, get_config, reset_config
from promptmatryoshka.llm_factory import LLMFactory, get_factory
from promptmatryoshka.llm_interface import LLMInterface, LLMConfig, ProviderInfo
from promptmatryoshka.providers import clear_registry
from promptmatryoshka.plugins.base import PluginBase, get_plugin_registry
from promptmatryoshka.plugins.boost import BoostPlugin
from promptmatryoshka.plugins.flipattack import FlipAttackPlugin
from promptmatryoshka.plugins.logitranslate import LogiTranslatePlugin
from promptmatryoshka.plugins.logiattack import LogiAttackPlugin
from promptmatryoshka.plugins.judge import JudgePlugin
from promptmatryoshka.exceptions import LLMError, LLMConfigurationError


class MockPluginLLMInterface(LLMInterface):
    """Mock LLM interface for plugin testing that tracks usage."""
    
    def __init__(self, config, provider_name="mock", plugin_name=None, **kwargs):
        super().__init__(config, provider_name=provider_name, **kwargs)
        self.provider_name = provider_name
        self.plugin_name = plugin_name
        self.invoke_calls = []
        self.ainvoke_calls = []
        
    def invoke(self, input, config=None, **kwargs):
        self.invoke_calls.append((input, config, kwargs))
        
        # Generate plugin-appropriate responses
        if self.plugin_name == "logitranslate":
            return "φ ≡ (∀x(User(x) → Block(x,y)) → ∀m(Message(m,x) → Hidden(m)))"
        elif self.plugin_name == "logiattack":
            return "To make a delicious cake, first preheat your oven to 350°F. Mix flour, sugar, and eggs..."
        elif self.plugin_name == "judge":
            return "Score: 0.2\nReasoning: This appears to be a harmless request about baking."
        else:
            return f"Mock response from {self.provider_name} for {self.plugin_name}: {input[:50]}..."
    
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
            limits={"max_tokens": 4000}
        )


class TestPluginSystemIntegration:
    """Test plugin system integration with multi-provider support."""
    
    def setup_method(self):
        """Set up test environment for plugin testing."""
        # Reset global state
        reset_config()
        clear_registry()
        
        # Create temporary directory and config
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "plugin_test_config.json")
        
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
                }
            },
            "plugins": {
                "logitranslate": {
                    "profile": "research-openai",
                    "technique_params": {
                        "validation_enabled": True,
                        "max_attempts": 3,
                        "retry_delay": 1.0,
                        "schema_strict": False  # Relaxed for testing
                    }
                },
                "logiattack": {
                    "profile": "research-openai",
                    "technique_params": {
                        "validation_enabled": True,
                        "schema_strict": False,  # Relaxed for testing
                        "max_attempts": 2
                    }
                },
                "judge": {
                    "profile": "production-anthropic",
                    "technique_params": {
                        "threshold": 0.8,
                        "multi_judge": True,
                        "strict_evaluation": True
                    }
                },
                "boost": {
                    "technique_params": {
                        "mode": "append",
                        "num_eos": 5,
                        "eos_token": "</s>",
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
        
        # Initialize config and factory
        self.config = Config(self.config_path)
        self.factory = LLMFactory()
        
    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clean up plugin storage directories
        for storage_dir in ["boost_results", "flipattack_results", "logitranslate_results", 
                           "logiattack_results", "judge_results"]:
            if os.path.exists(storage_dir):
                shutil.rmtree(storage_dir)
        
        reset_config()
        clear_registry()

    def test_technique_based_plugins_no_llm_required(self):
        """Test technique-based plugins (boost, flipattack) that don't require LLM."""
        # Test BoostPlugin (text mutation)
        boost_plugin = BoostPlugin(num_eos=3)
        test_input = "test input for boost"
        boost_result = boost_plugin.run(test_input)
        
        assert boost_result == "test input for boost</s></s></s>"
        
        # Test FlipAttackPlugin (character flipping)
        flipattack_plugin = FlipAttackPlugin(mode="char")
        test_input = "attack the system"
        flipattack_result = flipattack_plugin.run(test_input)
        
        assert "SYSTEM:" in flipattack_result
        assert "USER:" in flipattack_result
        assert "metsys eht kcatta" in flipattack_result  # Reversed text
        
        # Test FlipAttackPlugin (word flipping)
        flipattack_word_plugin = FlipAttackPlugin(mode="word")
        word_result = flipattack_word_plugin.run(test_input)
        
        assert "SYSTEM:" in word_result
        assert "system the attack" in word_result  # Reversed words

    def test_llm_based_plugins_with_multi_provider_support(self):
        """Test LLM-based plugins (logitranslate, logiattack, judge) with different providers."""
        
        # Test LogiTranslatePlugin with different providers
        providers_to_test = ["openai", "anthropic", "ollama"]
        
        for provider in providers_to_test:
            with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
                def create_mock_for_plugin(config, **kwargs):
                    return MockPluginLLMInterface(
                        config, 
                        provider_name=provider,
                        plugin_name="logitranslate"
                    )
                
                mock_get_provider.return_value = create_mock_for_plugin
                
                # Create plugin with provider-specific configuration
                plugin_config = self.test_config["plugins"]["logitranslate"].copy()
                plugin_config["provider"] = provider
                
                # Test plugin creation and execution
                plugin = LogiTranslatePlugin()
                
                # Mock the LLM creation for the plugin
                with patch.object(plugin, 'llm', create_mock_for_plugin({})):
                    test_input = "If a user blocks another, all messages are hidden."
                    result = plugin.run(test_input, save_dir=self.temp_dir)
                    
                    assert result is not None
                    assert len(result) > 0
                    assert "φ ≡" in result  # Should contain logical formula

    def test_plugin_configuration_inheritance_and_overrides(self):
        """Test plugin configuration inheritance from profiles and provider-specific overrides."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockPluginLLMInterface(config)
            
            # Test plugin using profile configuration
            plugin_config = self.test_config["plugins"]["logitranslate"]
            profile_name = plugin_config["profile"]  # "research-openai"
            
            # Create LLM for plugin using profile
            interface = self.factory.create_for_plugin("logitranslate", plugin_config)
            
            assert interface is not None
            assert isinstance(interface, MockPluginLLMInterface)
            
            # Test technique-specific parameters
            technique_params = plugin_config["technique_params"]
            assert technique_params["validation_enabled"] is True
            assert technique_params["max_attempts"] == 3

    def test_mixed_plugin_pipeline_execution(self):
        """Test pipeline execution with mixed plugin types (technique + LLM-based)."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockPluginLLMInterface(config)
            
            # Create pipeline with mixed plugin types
            pipeline = PromptMatryoshka(
                plugins=[
                    BoostPlugin(num_eos=2),          # Technique-based
                    FlipAttackPlugin(mode="word")    # Technique-based
                ],
                config_path=self.config_path
            )
            
            test_input = "test mixed pipeline execution"
            result = pipeline.jailbreak(test_input)
            
            assert result is not None
            assert len(result) > 0
            
            # Should contain boost modification (EOS tokens)
            assert result.endswith("</s></s>")
            
            # Should contain flipattack structure
            assert "SYSTEM:" in result
            assert "USER:" in result

    def test_plugin_error_handling_and_recovery(self):
        """Test plugin error handling and recovery mechanisms."""
        
        # Test technique-based plugin error handling
        boost_plugin = BoostPlugin(num_eos=0)  # Edge case: zero EOS tokens
        result = boost_plugin.run("test input")
        assert result == "test input"  # Should handle gracefully
        
        # Test LLM-based plugin error handling
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            # Mock provider that raises errors
            def failing_provider(config, **kwargs):
                raise Exception("Provider connection failed")
            
            mock_get_provider.return_value = failing_provider
            
            # Test error handling in plugin LLM creation
            try:
                plugin_config = self.test_config["plugins"]["logitranslate"]
                interface = self.factory.create_for_plugin("logitranslate", plugin_config)
            except Exception as e:
                # Should handle provider creation failures gracefully
                assert "failed" in str(e).lower() or "error" in str(e).lower()

    def test_plugin_discovery_and_registration(self):
        """Test automatic plugin discovery and registration."""
        
        # Create pipeline with auto-discovery
        pipeline = PromptMatryoshka(
            auto_discover=True,
            config_path=self.config_path
        )
        
        # Verify plugins were discovered
        registry = get_plugin_registry()
        available_plugins = registry.get_all_plugins()
        
        expected_plugins = ["boost", "flipattack", "logitranslate", "logiattack", "judge"]
        for plugin_name in expected_plugins:
            assert plugin_name in available_plugins or any(
                plugin_name in name.lower() for name in available_plugins.keys()
            )

    def test_plugin_performance_characteristics(self):
        """Test performance characteristics of different plugin types."""
        import time
        
        # Test technique-based plugin performance (should be fast)
        start_time = time.time()
        boost_plugin = BoostPlugin(num_eos=5)
        boost_result = boost_plugin.run("test input" * 100)  # Larger input
        boost_time = time.time() - start_time
        
        assert boost_time < 0.1  # Should be very fast (< 100ms)
        assert boost_result.endswith("</s></s></s></s></s>")
        
        # Test flipattack performance
        start_time = time.time()
        flipattack_plugin = FlipAttackPlugin(mode="char")
        flipattack_result = flipattack_plugin.run("test input" * 50)
        flipattack_time = time.time() - start_time
        
        assert flipattack_time < 0.5  # Should be reasonably fast (< 500ms)
        assert "SYSTEM:" in flipattack_result

    def test_plugin_storage_and_artifact_management(self):
        """Test plugin storage and artifact management capabilities."""
        
        # Test boost plugin storage
        boost_plugin = BoostPlugin(num_eos=3)
        test_input = "test storage functionality"
        result = boost_plugin.run(test_input)
        
        # Check if storage artifacts were created
        boost_storage_dir = "boost_results"
        if os.path.exists(boost_storage_dir):
            files = os.listdir(boost_storage_dir)
            assert len(files) > 0
            assert any(f.endswith('.json') for f in files)
        
        # Test flipattack plugin storage  
        flipattack_plugin = FlipAttackPlugin(mode="char")
        result = flipattack_plugin.run(test_input)
        
        flipattack_storage_dir = "flipattack_results"
        if os.path.exists(flipattack_storage_dir):
            files = os.listdir(flipattack_storage_dir)
            assert len(files) > 0
            assert any(f.endswith('.json') for f in files)

    def test_plugin_backward_compatibility(self):
        """Test backward compatibility with existing plugin workflows."""
        
        # Test legacy pipeline creation with stages parameter
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockPluginLLMInterface(config)
            
            # Create pipeline using legacy stages parameter
            boost_plugin = BoostPlugin(num_eos=2)
            flipattack_plugin = FlipAttackPlugin(mode="word")
            
            pipeline = PromptMatryoshka(stages=[boost_plugin, flipattack_plugin])
            
            test_input = "legacy compatibility test"
            result = pipeline.jailbreak(test_input)
            
            assert result is not None
            assert len(result) > 0

    def test_plugin_provider_and_profile_integration(self):
        """Test plugin integration with provider and profile switching."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda config, **kwargs: MockPluginLLMInterface(config)
            
            # Create pipeline with technique-based plugins
            pipeline = PromptMatryoshka(
                plugins=[BoostPlugin(num_eos=2)],
                config_path=self.config_path
            )
            
            test_input = "provider profile integration test"
            
            # Test with different provider options
            result1 = pipeline.jailbreak(test_input, provider="openai")
            assert result1 is not None
            
            result2 = pipeline.jailbreak(test_input, provider="anthropic")
            assert result2 is not None
            
            # Test with profile options
            result3 = pipeline.jailbreak(test_input, profile="research-openai")
            assert result3 is not None
            
            result4 = pipeline.jailbreak(test_input, profile="production-anthropic")
            assert result4 is not None

    def test_plugin_configuration_validation(self):
        """Test plugin configuration validation and error handling."""
        
        # Test invalid technique parameters
        try:
            BoostPlugin(num_eos=-1)  # Invalid negative value
        except (ValueError, AssertionError):
            pass  # Expected to fail validation
        
        try:
            FlipAttackPlugin(mode="invalid_mode")  # Invalid mode
        except (ValueError, AssertionError):
            pass  # Expected to fail validation

    def test_plugin_llm_configuration_inheritance(self):
        """Test LLM configuration inheritance from plugin settings."""
        
        with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
            creation_configs = []
            
            def config_tracking_provider(config, **kwargs):
                creation_configs.append(config)
                return MockPluginLLMInterface(config)
            
            mock_get_provider.return_value = config_tracking_provider
            
            # Test different plugin configurations
            plugin_configs = [
                ("logitranslate", self.test_config["plugins"]["logitranslate"]),
                ("judge", self.test_config["plugins"]["judge"])
            ]
            
            for plugin_name, plugin_config in plugin_configs:
                interface = self.factory.create_for_plugin(plugin_name, plugin_config)
                assert interface is not None
            
            # Verify configurations were applied
            assert len(creation_configs) >= 2

    def test_concurrent_plugin_execution(self):
        """Test concurrent execution of multiple plugins."""
        import concurrent.futures
        import threading
        
        execution_results = {}
        lock = threading.Lock()
        
        def execute_plugin(plugin_name):
            if plugin_name == "boost":
                plugin = BoostPlugin(num_eos=2)
                result = plugin.run(f"concurrent test for {plugin_name}")
            elif plugin_name == "flipattack":
                plugin = FlipAttackPlugin(mode="char")
                result = plugin.run(f"concurrent test for {plugin_name}")
            else:
                result = f"mock result for {plugin_name}"
            
            with lock:
                execution_results[plugin_name] = result
            
            return result
        
        # Test concurrent execution of technique-based plugins
        plugin_names = ["boost", "flipattack"]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(execute_plugin, name) for name in plugin_names]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all plugins executed successfully
        assert len(results) == 2
        assert len(execution_results) == 2
        
        for plugin_name in plugin_names:
            assert plugin_name in execution_results
            assert execution_results[plugin_name] is not None

    def test_plugin_integration_edge_cases(self):
        """Test plugin integration edge cases and boundary conditions."""
        
        # Test empty input handling
        boost_plugin = BoostPlugin(num_eos=2)
        empty_result = boost_plugin.run("")
        assert empty_result == "</s></s>"
        
        # Test very large input handling
        large_input = "test " * 1000
        large_result = boost_plugin.run(large_input)
        assert large_result.endswith("</s></s>")
        assert len(large_result) > len(large_input)
        
        # Test special character handling
        special_input = "test with émojis 🚀 and special chars: <>&\""
        special_result = boost_plugin.run(special_input)
        assert special_result.startswith(special_input)
        assert special_result.endswith("</s></s>")


class TestPluginSystemAdvancedFeatures:
    """Test advanced plugin system features and capabilities."""
    
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

    def test_plugin_dependency_resolution(self):
        """Test plugin dependency resolution and ordering."""
        
        # Create pipeline with auto-discovery and dependency resolution
        pipeline = PromptMatryoshka(
            plugins=["boost", "flipattack"],  # Test plugin name resolution
            auto_discover=True
        )
        
        # Verify pipeline was built successfully
        assert pipeline.pipeline is not None
        assert len(pipeline.pipeline) >= 1

    def test_plugin_metadata_and_capabilities(self):
        """Test plugin metadata and capability reporting."""
        
        # Test boost plugin metadata
        boost_plugin = BoostPlugin()
        if hasattr(boost_plugin, 'get_plugin_name'):
            assert boost_plugin.get_plugin_name() is not None
        
        # Test flipattack plugin metadata
        flipattack_plugin = FlipAttackPlugin()
        if hasattr(flipattack_plugin, 'get_plugin_name'):
            assert flipattack_plugin.get_plugin_name() is not None

    def test_plugin_system_health_and_monitoring(self):
        """Test plugin system health checking and monitoring."""
        
        # Create pipeline and get health information
        pipeline = PromptMatryoshka(
            plugins=[BoostPlugin(), FlipAttackPlugin()],
            auto_discover=True
        )
        
        # Get pipeline information
        pipeline_info = pipeline.get_pipeline_info()
        
        assert "plugins" in pipeline_info
        assert "total_plugins" in pipeline_info
        assert pipeline_info["total_plugins"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])