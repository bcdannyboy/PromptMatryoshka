"""Unit tests for provider registry infrastructure.

Tests the provider registration, discovery, and management functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Type, Dict, Any, List

from promptmatryoshka.providers import (
    register_provider,
    get_provider,
    list_providers,
    list_builtin_providers,
    is_provider_available,
    get_provider_info,
    discover_providers,
    clear_registry,
    BUILTIN_PROVIDERS,
    _PROVIDER_REGISTRY,
    _PROVIDER_AVAILABILITY,
    _load_builtin_provider,
    _check_dependencies
)
from promptmatryoshka.llm_interface import LLMInterface, LLMConfig, ProviderInfo
from promptmatryoshka.exceptions import LLMUnsupportedProviderError


class MockProvider(LLMInterface):
    """Mock provider for testing."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config, provider_name="mock", **kwargs)
    
    def invoke(self, input, config=None, **kwargs):
        return "Mock response"
    
    async def ainvoke(self, input, config=None, **kwargs):
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


class TestProviderRegistry:
    """Tests for provider registry functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear registry before each test
        clear_registry()
    
    def teardown_method(self):
        """Clean up after each test."""
        # Clear registry after each test
        clear_registry()
    
    def test_register_provider_success(self):
        """Test successful provider registration."""
        register_provider("mock", MockProvider)
        
        assert "mock" in _PROVIDER_REGISTRY
        assert _PROVIDER_REGISTRY["mock"] == MockProvider
    
    def test_register_provider_case_insensitive(self):
        """Test that provider registration is case insensitive."""
        register_provider("MOCK", MockProvider)
        
        assert "mock" in _PROVIDER_REGISTRY
        assert _PROVIDER_REGISTRY["mock"] == MockProvider
    
    def test_register_provider_invalid_name(self):
        """Test registering provider with invalid name."""
        with pytest.raises(ValueError):
            register_provider("", MockProvider)
        
        with pytest.raises(ValueError):
            register_provider("   ", MockProvider)
    
    def test_register_provider_invalid_class(self):
        """Test registering provider with invalid class."""
        class NotAnLLMInterface:
            pass
        
        with pytest.raises(TypeError):
            register_provider("invalid", NotAnLLMInterface)
    
    def test_register_provider_override_warning(self):
        """Test that overriding existing provider logs warning."""
        register_provider("mock", MockProvider)
        
        with patch('promptmatryoshka.providers.logger') as mock_logger:
            register_provider("mock", MockProvider)
            mock_logger.warning.assert_called_once()
    
    def test_get_provider_success(self):
        """Test successful provider retrieval."""
        register_provider("mock", MockProvider)
        
        provider_class = get_provider("mock")
        
        assert provider_class == MockProvider
    
    def test_get_provider_case_insensitive(self):
        """Test that provider retrieval is case insensitive."""
        register_provider("mock", MockProvider)
        
        provider_class = get_provider("MOCK")
        
        assert provider_class == MockProvider
    
    def test_get_provider_not_found(self):
        """Test getting provider that doesn't exist."""
        with pytest.raises(LLMUnsupportedProviderError):
            get_provider("nonexistent")
    
    def test_get_provider_auto_load_builtin(self):
        """Test auto-loading of built-in providers."""
        # Mock the builtin loading - don't pre-populate the registry
        with patch('promptmatryoshka.providers._load_builtin_provider') as mock_load:
            mock_load.return_value = None
            # After loading, the provider should be in the registry
            def side_effect(name):
                _PROVIDER_REGISTRY[name] = MockProvider
            mock_load.side_effect = side_effect
            
            provider_class = get_provider("openai")
            
            assert provider_class == MockProvider
            mock_load.assert_called_once_with("openai")
    
    def test_list_providers_empty(self):
        """Test listing providers when registry is empty."""
        providers = list_providers()
        
        assert providers == []
    
    def test_list_providers_with_registered(self):
        """Test listing providers with registered providers."""
        register_provider("mock1", MockProvider)
        register_provider("mock2", MockProvider)
        
        providers = list_providers()
        
        assert "mock1" in providers
        assert "mock2" in providers
        assert len(providers) == 2
    
    def test_list_builtin_providers(self):
        """Test listing built-in providers."""
        builtin_providers = list_builtin_providers()
        
        assert "openai" in builtin_providers
        assert "anthropic" in builtin_providers
        assert "ollama" in builtin_providers
        assert "huggingface" in builtin_providers
    
    def test_is_provider_available_registered(self):
        """Test checking availability of registered provider."""
        register_provider("mock", MockProvider)
        
        assert is_provider_available("mock") is True
    
    def test_is_provider_available_not_registered(self):
        """Test checking availability of unregistered provider."""
        assert is_provider_available("nonexistent") is False
    
    def test_is_provider_available_cached(self):
        """Test that availability checks are cached."""
        register_provider("mock", MockProvider)
        
        # First call should compute availability
        result1 = is_provider_available("mock")
        
        # Second call should use cache
        result2 = is_provider_available("mock")
        
        assert result1 is True
        assert result2 is True
        assert "mock" in _PROVIDER_AVAILABILITY
    
    def test_is_provider_available_builtin_with_dependencies(self):
        """Test checking availability of built-in provider with dependencies."""
        with patch('promptmatryoshka.providers._check_dependencies') as mock_check:
            mock_check.return_value = True
            
            # Should try to load built-in provider
            with patch('promptmatryoshka.providers._load_builtin_provider') as mock_load:
                mock_load.return_value = None
                _PROVIDER_REGISTRY["openai"] = MockProvider
                
                result = is_provider_available("openai")
                
                assert result is True
                mock_check.assert_called_once()
    
    def test_is_provider_available_builtin_missing_dependencies(self):
        """Test checking availability of built-in provider with missing dependencies."""
        with patch('promptmatryoshka.providers._check_dependencies') as mock_check:
            mock_check.return_value = False
            
            result = is_provider_available("openai")
            
            assert result is False
    
    def test_get_provider_info_builtin(self):
        """Test getting info for built-in provider."""
        info = get_provider_info("openai")
        
        assert isinstance(info, dict)
        assert info["name"] == "openai"
        assert "description" in info
        assert "dependencies" in info
        assert "available" in info
    
    def test_get_provider_info_registered(self):
        """Test getting info for registered provider."""
        register_provider("mock", MockProvider)
        
        info = get_provider_info("mock")
        
        assert isinstance(info, dict)
        assert info["name"] == "mock"
        assert info["class"] == "MockProvider"
        assert info["available"] is True
        assert info["registered"] is True
    
    def test_get_provider_info_not_found(self):
        """Test getting info for non-existent provider."""
        with pytest.raises(LLMUnsupportedProviderError):
            get_provider_info("nonexistent")
    
    def test_discover_providers(self):
        """Test provider discovery."""
        register_provider("mock", MockProvider)
        
        providers = discover_providers()
        
        assert isinstance(providers, dict)
        assert "mock" in providers
        # Should include built-in providers
        assert "openai" in providers
        assert "anthropic" in providers
    
    def test_discover_providers_with_errors(self):
        """Test provider discovery with errors."""
        register_provider("mock", MockProvider)
        
        with patch('promptmatryoshka.providers.get_provider_info') as mock_get_info:
            mock_get_info.side_effect = lambda name: {
                "name": name,
                "available": True
            } if name == "mock" else Exception("Test error")
            
            providers = discover_providers()
            
            assert "mock" in providers
            assert providers["mock"]["available"] is True
    
    def test_clear_registry(self):
        """Test clearing the registry."""
        register_provider("mock", MockProvider)
        _PROVIDER_AVAILABILITY["mock"] = True
        
        assert len(_PROVIDER_REGISTRY) > 0
        assert len(_PROVIDER_AVAILABILITY) > 0
        
        clear_registry()
        
        assert len(_PROVIDER_REGISTRY) == 0
        assert len(_PROVIDER_AVAILABILITY) == 0


class TestBuiltinProviderLoading:
    """Tests for built-in provider loading."""
    
    def setup_method(self):
        """Set up test fixtures."""
        clear_registry()
    
    def teardown_method(self):
        """Clean up after each test."""
        clear_registry()
    
    def test_load_builtin_provider_success(self):
        """Test successful built-in provider loading."""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.OpenAIProvider = MockProvider
            mock_import.return_value = mock_module
            
            _load_builtin_provider("openai")
            
            assert "openai" in _PROVIDER_REGISTRY
            assert _PROVIDER_REGISTRY["openai"] == MockProvider
    
    def test_load_builtin_provider_import_error(self):
        """Test built-in provider loading with import error."""
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            
            with pytest.raises(LLMUnsupportedProviderError):
                _load_builtin_provider("openai")
    
    def test_load_builtin_provider_attribute_error(self):
        """Test built-in provider loading with missing class."""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            # Missing the expected class
            mock_import.return_value = mock_module
            
            with pytest.raises(LLMUnsupportedProviderError):
                _load_builtin_provider("openai")
    
    def test_load_builtin_provider_not_builtin(self):
        """Test loading provider that's not in built-in list."""
        with pytest.raises(LLMUnsupportedProviderError):
            _load_builtin_provider("nonexistent")
    
    def test_check_dependencies_success(self):
        """Test successful dependency checking."""
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = Mock()
            
            result = _check_dependencies(["existing_module"])
            
            assert result is True
    
    def test_check_dependencies_failure(self):
        """Test dependency checking with missing module."""
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            
            result = _check_dependencies(["missing_module"])
            
            assert result is False
    
    def test_check_dependencies_multiple_modules(self):
        """Test dependency checking with multiple modules."""
        with patch('importlib.import_module') as mock_import:
            # First module exists, second doesn't
            mock_import.side_effect = [Mock(), ImportError("Module not found")]
            
            result = _check_dependencies(["existing_module", "missing_module"])
            
            assert result is False
    
    def test_check_dependencies_dash_to_underscore(self):
        """Test that dashes in module names are converted to underscores."""
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = Mock()
            
            _check_dependencies(["module-with-dashes"])
            
            mock_import.assert_called_with("module_with_dashes")


class TestBuiltinProviderConfiguration:
    """Tests for built-in provider configuration."""
    
    def test_builtin_providers_structure(self):
        """Test that built-in providers have required structure."""
        for name, config in BUILTIN_PROVIDERS.items():
            assert isinstance(name, str)
            assert isinstance(config, dict)
            assert "module" in config
            assert "class" in config
            assert "dependencies" in config
            assert "description" in config
            assert isinstance(config["dependencies"], list)
    
    def test_builtin_providers_names(self):
        """Test that expected built-in providers are defined."""
        expected_providers = ["openai", "anthropic", "ollama", "huggingface"]
        
        for provider in expected_providers:
            assert provider in BUILTIN_PROVIDERS
    
    def test_builtin_provider_openai_config(self):
        """Test OpenAI provider configuration."""
        config = BUILTIN_PROVIDERS["openai"]
        
        assert config["module"] == "promptmatryoshka.providers.openai_provider"
        assert config["class"] == "OpenAIProvider"
        assert "langchain-openai" in config["dependencies"]
        assert "OpenAI" in config["description"]
    
    def test_builtin_provider_anthropic_config(self):
        """Test Anthropic provider configuration."""
        config = BUILTIN_PROVIDERS["anthropic"]
        
        assert config["module"] == "promptmatryoshka.providers.anthropic_provider"
        assert config["class"] == "AnthropicProvider"
        assert "langchain-anthropic" in config["dependencies"]
        assert "Anthropic" in config["description"]
    
    def test_builtin_provider_ollama_config(self):
        """Test Ollama provider configuration."""
        config = BUILTIN_PROVIDERS["ollama"]
        
        assert config["module"] == "promptmatryoshka.providers.ollama_provider"
        assert config["class"] == "OllamaProvider"
        assert "langchain-ollama" in config["dependencies"]
        assert "Ollama" in config["description"]
    
    def test_builtin_provider_huggingface_config(self):
        """Test HuggingFace provider configuration."""
        config = BUILTIN_PROVIDERS["huggingface"]
        
        assert config["module"] == "promptmatryoshka.providers.huggingface_provider"
        assert config["class"] == "HuggingFaceProvider"
        assert "langchain-huggingface" in config["dependencies"]
        assert "HuggingFace" in config["description"]


class TestAutoDiscovery:
    """Tests for auto-discovery functionality."""
    
    def test_auto_discovery_runs_on_import(self):
        """Test that auto-discovery runs when module is imported."""
        # This test verifies that the auto-discovery function runs
        # We can't easily test the actual import behavior, but we can
        # test that the function itself works
        
        with patch('promptmatryoshka.providers.is_provider_available') as mock_available:
            mock_available.return_value = True
            
            # Import the _auto_discover function and run it
            from promptmatryoshka.providers import _auto_discover
            
            # Should not raise an exception
            _auto_discover()
    
    def test_auto_discovery_with_failures(self):
        """Test auto-discovery with failures."""
        with patch('promptmatryoshka.providers.is_provider_available') as mock_available:
            mock_available.side_effect = Exception("Discovery error")
            
            # Import the _auto_discover function and run it
            from promptmatryoshka.providers import _auto_discover
            
            # Should handle exceptions gracefully
            _auto_discover()


class TestRegistryEdgeCases:
    """Tests for edge cases in registry functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        clear_registry()
    
    def teardown_method(self):
        """Clean up after each test."""
        clear_registry()
    
    def test_provider_name_whitespace_handling(self):
        """Test provider name whitespace handling."""
        register_provider("  mock  ", MockProvider)
        
        assert "mock" in _PROVIDER_REGISTRY
        
        provider_class = get_provider("  MOCK  ")
        assert provider_class == MockProvider
    
    def test_provider_availability_cache_persistence(self):
        """Test that availability cache persists across calls."""
        register_provider("mock", MockProvider)
        
        # First call should compute and cache
        result1 = is_provider_available("mock")
        assert result1 is True
        assert "mock" in _PROVIDER_AVAILABILITY
        
        # Modify registry but cache should still return cached value
        _PROVIDER_REGISTRY.clear()
        result2 = is_provider_available("mock")
        assert result2 is True  # Should return cached value
    
    def test_get_provider_info_with_partial_data(self):
        """Test getting provider info with partial data."""
        # Register provider without full info
        register_provider("minimal", MockProvider)
        
        info = get_provider_info("minimal")
        
        assert info["name"] == "minimal"
        assert info["available"] is True
        assert info["registered"] is True
        assert "dependencies" in info
    
    def test_discover_providers_handles_all_exceptions(self):
        """Test that discover_providers handles all types of exceptions."""
        register_provider("mock", MockProvider)
        
        with patch('promptmatryoshka.providers.get_provider_info') as mock_get_info:
            # Make it raise different types of exceptions
            def side_effect(name):
                if name == "mock":
                    raise RuntimeError("Unexpected error")
                return {"name": name, "available": True}
            
            mock_get_info.side_effect = side_effect
            
            providers = discover_providers()
            
            # Should handle the exception and include error info
            assert "mock" in providers
            assert providers["mock"]["available"] is False
            assert "error" in providers["mock"]


if __name__ == "__main__":
    pytest.main([__file__])