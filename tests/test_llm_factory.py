"""Unit tests for LLM Factory infrastructure.

Tests the factory pattern implementation, provider registry, and configuration management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Union, List

from promptmatryoshka.llm_factory import LLMFactory, get_factory, create_llm_interface
from promptmatryoshka.llm_interface import LLMInterface, LLMConfig, ProviderInfo
from promptmatryoshka.exceptions import (
    LLMConfigurationError,
    LLMUnsupportedProviderError,
    LLMValidationError
)


class MockProvider(LLMInterface):
    """Mock provider for testing."""
    
    def __init__(self, config: Union[LLMConfig, Dict[str, Any]], **kwargs):
        super().__init__(config, provider_name="mock", **kwargs)
        self.invoke_calls = []
        self.ainvoke_calls = []
    
    def invoke(self, input, config=None, **kwargs):
        self.invoke_calls.append((input, config, kwargs))
        return "Mock response"
    
    async def ainvoke(self, input, config=None, **kwargs):
        self.ainvoke_calls.append((input, config, kwargs))
        return "Mock response"
    
    def validate_config(self) -> bool:
        return True
    
    def health_check(self) -> bool:
        return True
    
    def get_provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="mock",
            version="1.0.0",
            models=["mock-model"],
            capabilities={"chat": True},
            limits={"max_tokens": 1000}
        )


class TestLLMFactory:
    """Tests for LLMFactory class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = LLMFactory()
        
        # Mock provider registry functions
        self.registry_patcher = patch.multiple(
            'promptmatryoshka.llm_factory',
            is_provider_available=Mock(return_value=True),
            get_provider=Mock(return_value=MockProvider),
            list_providers=Mock(return_value=['mock', 'openai']),
            get_provider_info=Mock(return_value=ProviderInfo(
                name="mock",
                version="1.0.0",
                models=["mock-model"],
                capabilities={"chat": True},
                limits={"max_tokens": 1000}
            ))
        )
        self.registry_patcher.start()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.registry_patcher.stop()
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = LLMFactory()
        
        assert factory.logger is not None
        assert isinstance(factory._provider_cache, dict)
        assert isinstance(factory._config_profiles, dict)
        assert len(factory._config_profiles) > 0  # Should have default profiles
    
    def test_create_interface_with_dict_config(self):
        """Test creating interface with dictionary configuration."""
        config = {
            "model": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 1000
        }
        
        interface = self.factory.create_interface("mock", config)
        
        assert isinstance(interface, MockProvider)
        assert interface.config.model == "gpt-4"
        assert interface.config.temperature == 0.5
        assert interface.config.max_tokens == 1000
    
    def test_create_interface_with_llm_config(self):
        """Test creating interface with LLMConfig object."""
        config = LLMConfig(model="gpt-3.5-turbo", temperature=0.7)
        
        interface = self.factory.create_interface("mock", config)
        
        assert isinstance(interface, MockProvider)
        assert interface.config.model == "gpt-3.5-turbo"
        assert interface.config.temperature == 0.7
    
    def test_create_interface_with_caching(self):
        """Test interface creation with caching."""
        config = {"model": "gpt-4"}
        
        # First call should create new instance
        interface1 = self.factory.create_interface("mock", config, use_cache=True)
        
        # Second call should return cached instance
        interface2 = self.factory.create_interface("mock", config, use_cache=True)
        
        assert interface1 is interface2
        assert len(self.factory._provider_cache) == 1
    
    def test_create_interface_without_caching(self):
        """Test interface creation without caching."""
        config = {"model": "gpt-4"}
        
        # Both calls should create new instances
        interface1 = self.factory.create_interface("mock", config, use_cache=False)
        interface2 = self.factory.create_interface("mock", config, use_cache=False)
        
        assert interface1 is not interface2
        assert len(self.factory._provider_cache) == 0
    
    def test_create_interface_with_custom_cache_key(self):
        """Test interface creation with custom cache key."""
        config = {"model": "gpt-4"}
        
        interface = self.factory.create_interface("mock", config, cache_key="custom_key")
        
        assert "custom_key" in self.factory._provider_cache
        assert self.factory._provider_cache["custom_key"] is interface
    
    def test_create_interface_unsupported_provider(self):
        """Test creating interface with unsupported provider."""
        with patch('promptmatryoshka.llm_factory.is_provider_available', return_value=False):
            with pytest.raises(LLMUnsupportedProviderError):
                self.factory.create_interface("unsupported", {"model": "test"})
    
    def test_create_interface_provider_creation_failure(self):
        """Test interface creation when provider class creation fails."""
        with patch('promptmatryoshka.llm_factory.get_provider', side_effect=Exception("Provider error")):
            with pytest.raises(LLMUnsupportedProviderError):
                self.factory.create_interface("mock", {"model": "test"})
    
    def test_create_from_profile_existing_profile(self):
        """Test creating interface from existing profile."""
        # Use a default profile
        interface = self.factory.create_from_profile("fast")
        
        assert isinstance(interface, MockProvider)
        assert interface.config.model == "gpt-3.5-turbo"
        assert interface.config.temperature == 0.0
    
    def test_create_from_profile_with_overrides(self):
        """Test creating interface from profile with overrides."""
        overrides = {
            "temperature": 0.8,
            "max_tokens": 500
        }
        
        interface = self.factory.create_from_profile("fast", overrides=overrides)
        
        assert interface.config.temperature == 0.8
        assert interface.config.max_tokens == 500
        # Original profile values should be preserved for other params
        assert interface.config.model == "gpt-3.5-turbo"
    
    def test_create_from_profile_nonexistent_profile(self):
        """Test creating interface from nonexistent profile."""
        with pytest.raises(LLMConfigurationError):
            self.factory.create_from_profile("nonexistent")
    
    def test_create_from_profile_missing_provider(self):
        """Test creating interface from profile missing provider."""
        # Try to register a profile without provider - should fail
        with pytest.raises(LLMConfigurationError):
            self.factory.register_profile("no_provider", {"model": "test"})
    
    def test_create_for_plugin_basic(self):
        """Test creating interface for plugin with basic configuration."""
        plugin_config = {
            "model": "gpt-4",
            "temperature": 0.5,
            "provider": "mock"
        }
        
        interface = self.factory.create_for_plugin("test_plugin", plugin_config)
        
        assert isinstance(interface, MockProvider)
        assert interface.config.model == "gpt-4"
        assert interface.config.temperature == 0.5
    
    def test_create_for_plugin_with_nested_llm_config(self):
        """Test creating interface for plugin with nested LLM configuration."""
        plugin_config = {
            "llm": {
                "model": "gpt-4",
                "temperature": 0.7,
                "provider": "mock"
            },
            "other_setting": "value"
        }
        
        interface = self.factory.create_for_plugin("test_plugin", plugin_config)
        
        assert interface.config.model == "gpt-4"
        assert interface.config.temperature == 0.7
    
    def test_create_for_plugin_with_fallback_provider(self):
        """Test creating interface for plugin with fallback provider."""
        plugin_config = {
            "model": "gpt-4",
            "temperature": 0.5
            # No provider specified
        }
        
        interface = self.factory.create_for_plugin("test_plugin", plugin_config, fallback_provider="mock")
        
        assert isinstance(interface, MockProvider)
        assert interface.config.model == "gpt-4"
    
    def test_create_for_plugin_specific_defaults(self):
        """Test creating interface for plugin with plugin-specific defaults."""
        plugin_config = {
            "model": "gpt-4",
            "provider": "mock"
        }
        
        # Test logitranslate plugin defaults
        interface = self.factory.create_for_plugin("logitranslate", plugin_config)
        
        assert interface.config.temperature == 0.0
        assert interface.config.max_tokens == 2000
    
    def test_validate_provider_success(self):
        """Test successful provider validation."""
        result = self.factory.validate_provider("mock")
        
        assert result is True
    
    def test_validate_provider_failure(self):
        """Test provider validation failure."""
        with patch('promptmatryoshka.llm_factory.is_provider_available', return_value=False):
            with pytest.raises(LLMUnsupportedProviderError):
                self.factory.validate_provider("unsupported")
    
    def test_validate_config_dict_success(self):
        """Test successful configuration validation with dict."""
        config = {
            "model": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 1000
        }
        
        result = self.factory.validate_config(config)
        
        assert result is True
    
    def test_validate_config_llm_config_success(self):
        """Test successful configuration validation with LLMConfig."""
        config = LLMConfig(model="gpt-4", temperature=0.5)
        
        result = self.factory.validate_config(config)
        
        assert result is True
    
    def test_validate_config_invalid_type(self):
        """Test configuration validation with invalid type."""
        with pytest.raises(LLMValidationError):
            self.factory.validate_config("invalid_config")
    
    def test_validate_config_invalid_dict(self):
        """Test configuration validation with invalid dict."""
        config = {
            "model": "test",
            "temperature": 3.0  # Invalid temperature
        }
        
        with pytest.raises(LLMValidationError):
            self.factory.validate_config(config)
    
    def test_get_provider_info(self):
        """Test getting provider information."""
        info = self.factory.get_provider_info("mock")
        
        assert isinstance(info, ProviderInfo)
        assert info.name == "mock"
    
    def test_list_available_providers(self):
        """Test listing available providers."""
        providers = self.factory.list_available_providers()
        
        assert isinstance(providers, list)
        assert "mock" in providers
        assert "openai" in providers
    
    def test_discover_providers(self):
        """Test provider discovery."""
        with patch('promptmatryoshka.llm_factory.discover_providers') as mock_discover:
            mock_discover.return_value = {"mock": {"available": True}}
            
            providers = self.factory.discover_providers()
            
            assert isinstance(providers, dict)
            mock_discover.assert_called_once()
    
    def test_register_profile_success(self):
        """Test successful profile registration."""
        profile_config = {
            "provider": "mock",
            "model": "test-model",
            "temperature": 0.5
        }
        
        self.factory.register_profile("test_profile", profile_config)
        
        assert "test_profile" in self.factory._config_profiles
        assert self.factory._config_profiles["test_profile"] == profile_config
    
    def test_register_profile_missing_provider(self):
        """Test profile registration without provider."""
        profile_config = {
            "model": "test-model",
            "temperature": 0.5
        }
        
        with pytest.raises(LLMConfigurationError):
            self.factory.register_profile("test_profile", profile_config)
    
    def test_register_profile_invalid_config(self):
        """Test profile registration with invalid configuration."""
        profile_config = {
            "provider": "mock",
            "model": "test-model",
            "temperature": 3.0  # Invalid temperature
        }
        
        with pytest.raises(LLMValidationError):
            self.factory.register_profile("test_profile", profile_config)
    
    def test_list_profiles(self):
        """Test listing profiles."""
        profiles = self.factory.list_profiles()
        
        assert isinstance(profiles, list)
        assert "fast" in profiles
        assert "quality" in profiles
        assert "local" in profiles
        assert "creative" in profiles
    
    def test_get_profile_success(self):
        """Test getting existing profile."""
        profile = self.factory.get_profile("fast")
        
        assert isinstance(profile, dict)
        assert profile["provider"] == "openai"
        assert profile["model"] == "gpt-3.5-turbo"
    
    def test_get_profile_nonexistent(self):
        """Test getting nonexistent profile."""
        with pytest.raises(LLMConfigurationError):
            self.factory.get_profile("nonexistent")
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add something to cache
        self.factory.create_interface("mock", {"model": "test"}, use_cache=True)
        assert len(self.factory._provider_cache) > 0
        
        # Clear cache
        self.factory.clear_cache()
        
        assert len(self.factory._provider_cache) == 0
    
    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        # Add something to cache
        self.factory.create_interface("mock", {"model": "test"}, use_cache=True)
        
        stats = self.factory.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert "cached_instances" in stats
        assert "cache_keys" in stats
        assert "profiles" in stats
        assert "profile_names" in stats
        assert stats["cached_instances"] > 0
        assert len(stats["cache_keys"]) > 0
    
    def test_default_profiles_initialization(self):
        """Test that default profiles are properly initialized."""
        factory = LLMFactory()
        
        profiles = factory.list_profiles()
        
        assert "fast" in profiles
        assert "quality" in profiles
        assert "local" in profiles
        assert "creative" in profiles
        
        # Check profile contents
        fast_profile = factory.get_profile("fast")
        assert fast_profile["provider"] == "openai"
        assert fast_profile["model"] == "gpt-3.5-turbo"
        
        quality_profile = factory.get_profile("quality")
        assert quality_profile["provider"] == "openai"
        assert quality_profile["model"] == "gpt-4"
        
        local_profile = factory.get_profile("local")
        assert local_profile["provider"] == "ollama"
        assert local_profile["model"] == "llama2"
        
        creative_profile = factory.get_profile("creative")
        assert creative_profile["provider"] == "anthropic"
        assert creative_profile["model"] == "claude-3-sonnet"
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        config1 = {"model": "gpt-4", "temperature": 0.5}
        config2 = {"model": "gpt-4", "temperature": 0.5}
        config3 = {"model": "gpt-4", "temperature": 0.7}
        
        key1 = self.factory._generate_cache_key("provider", config1)
        key2 = self.factory._generate_cache_key("provider", config2)
        key3 = self.factory._generate_cache_key("provider", config3)
        
        assert key1 == key2  # Same config should generate same key
        assert key1 != key3  # Different config should generate different key
    
    def test_extract_llm_config_from_plugin(self):
        """Test LLM configuration extraction from plugin config."""
        plugin_config = {
            "model": "gpt-4",
            "temperature": 0.5,
            "llm": {
                "max_tokens": 1000,
                "top_p": 0.9
            },
            "other_setting": "value"
        }
        
        llm_config = self.factory._extract_llm_config_from_plugin("test_plugin", plugin_config)
        
        assert llm_config["model"] == "gpt-4"
        assert llm_config["temperature"] == 0.5
        assert llm_config["max_tokens"] == 1000
        assert llm_config["top_p"] == 0.9
        assert "other_setting" not in llm_config


class TestGlobalFactoryFunctions:
    """Tests for global factory functions."""
    
    def test_get_factory_singleton(self):
        """Test that get_factory returns singleton instance."""
        factory1 = get_factory()
        factory2 = get_factory()
        
        assert factory1 is factory2
        assert isinstance(factory1, LLMFactory)
    
    @patch('promptmatryoshka.llm_factory.get_factory')
    def test_create_llm_interface_convenience_function(self, mock_get_factory):
        """Test convenience function for creating LLM interface."""
        mock_factory = Mock()
        mock_interface = Mock()
        mock_factory.create_interface.return_value = mock_interface
        mock_get_factory.return_value = mock_factory
        
        config = {"model": "gpt-4"}
        result = create_llm_interface("openai", config, extra_param="value")
        
        assert result is mock_interface
        mock_factory.create_interface.assert_called_once_with("openai", config, extra_param="value")


if __name__ == "__main__":
    pytest.main([__file__])