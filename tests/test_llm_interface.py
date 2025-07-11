"""Unit tests for LLM interface infrastructure.

Tests the abstract LLM interface, configuration models, and base functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any, Union, List

from promptmatryoshka.llm_interface import LLMInterface, LLMConfig, ProviderInfo
from promptmatryoshka.exceptions import (
    LLMConfigurationError,
    LLMHealthCheckError,
    LLMProviderError,
    LLMValidationError
)
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import LLMResult


class MockLLMInterface(LLMInterface):
    """Mock implementation of LLMInterface for testing."""
    
    def __init__(self, config: Union[LLMConfig, Dict[str, Any]], **kwargs):
        """Initialize mock LLM interface."""
        # Initialize tracking attributes before calling super()
        self.invoke_calls = []
        self.ainvoke_calls = []
        self.health_check_calls = []
        self.validate_config_calls = []
        self._should_fail_health_check = False
        self._should_fail_validation = False
        self._mock_response = "Mock response"
        
        super().__init__(config, provider_name="mock", **kwargs)
    
    def invoke(self, input, config=None, **kwargs):
        """Mock invoke method."""
        self.invoke_calls.append((input, config, kwargs))
        return self._mock_response
    
    async def ainvoke(self, input, config=None, **kwargs):
        """Mock async invoke method."""
        self.ainvoke_calls.append((input, config, kwargs))
        return self._mock_response
    
    def validate_config(self) -> bool:
        """Mock validate_config method."""
        self.validate_config_calls.append(True)
        if self._should_fail_validation:
            raise LLMConfigurationError("Mock validation error")
        return True
    
    def health_check(self) -> bool:
        """Mock health_check method."""
        self.health_check_calls.append(True)
        if self._should_fail_health_check:
            raise LLMHealthCheckError("Mock health check error")
        return True
    
    def get_provider_info(self) -> ProviderInfo:
        """Mock get_provider_info method."""
        return ProviderInfo(
            name="mock",
            version="1.0.0",
            models=["mock-model"],
            capabilities={"chat": True},
            limits={"max_tokens": 1000}
        )


class TestLLMConfig:
    """Tests for LLMConfig model."""
    
    def test_config_creation_with_valid_data(self):
        """Test creating LLMConfig with valid data."""
        config = LLMConfig(
            model="gpt-4",
            temperature=0.5,
            max_tokens=2000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            request_timeout=60,
            max_retries=3,
            retry_delay=1.0
        )
        
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.1
        assert config.request_timeout == 60
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
    
    def test_config_creation_with_minimal_data(self):
        """Test creating LLMConfig with minimal required data."""
        config = LLMConfig(model="gpt-3.5-turbo")
        
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.0
        assert config.max_tokens == 2000
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.request_timeout == 120
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
    
    def test_config_validation_temperature_bounds(self):
        """Test temperature validation."""
        # Valid temperatures
        LLMConfig(model="test", temperature=0.0)
        LLMConfig(model="test", temperature=1.0)
        LLMConfig(model="test", temperature=2.0)
        
        # Invalid temperatures - expect ValidationError from Pydantic v2
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LLMConfig(model="test", temperature=-0.1)
        
        with pytest.raises(ValidationError):
            LLMConfig(model="test", temperature=2.1)
    
    def test_config_validation_top_p_bounds(self):
        """Test top_p validation."""
        # Valid top_p values
        LLMConfig(model="test", top_p=0.0)
        LLMConfig(model="test", top_p=0.5)
        LLMConfig(model="test", top_p=1.0)
        
        # Invalid top_p values - expect ValidationError from Pydantic v2
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LLMConfig(model="test", top_p=-0.1)
        
        with pytest.raises(ValidationError):
            LLMConfig(model="test", top_p=1.1)
    
    def test_config_optional_fields(self):
        """Test optional configuration fields."""
        config = LLMConfig(
            model="test",
            api_key="secret",
            api_base="https://api.example.com",
            organization="org-123",
            custom_headers={"X-Custom": "value"},
            provider_specific={"custom_param": "value"}
        )
        
        assert config.api_key == "secret"
        assert config.api_base == "https://api.example.com"
        assert config.organization == "org-123"
        assert config.custom_headers == {"X-Custom": "value"}
        assert config.provider_specific == {"custom_param": "value"}


class TestProviderInfo:
    """Tests for ProviderInfo model."""
    
    def test_provider_info_creation(self):
        """Test creating ProviderInfo."""
        info = ProviderInfo(
            name="test-provider",
            version="1.0.0",
            models=["model1", "model2"],
            capabilities={"chat": True, "completion": False},
            limits={"max_tokens": 4096, "rate_limit": "1000/min"},
            health_status="healthy",
            last_health_check=datetime.now()
        )
        
        assert info.name == "test-provider"
        assert info.version == "1.0.0"
        assert info.models == ["model1", "model2"]
        assert info.capabilities == {"chat": True, "completion": False}
        assert info.limits == {"max_tokens": 4096, "rate_limit": "1000/min"}
        assert info.health_status == "healthy"
        assert isinstance(info.last_health_check, datetime)
    
    def test_provider_info_defaults(self):
        """Test ProviderInfo with default values."""
        info = ProviderInfo(name="test", version="1.0.0")
        
        assert info.name == "test"
        assert info.version == "1.0.0"
        assert info.models == []
        assert info.capabilities == {}
        assert info.limits == {}
        assert info.health_status == "unknown"
        assert info.last_health_check is None


class TestLLMInterface:
    """Tests for LLMInterface abstract base class."""
    
    def test_interface_initialization_with_dict_config(self):
        """Test initializing interface with dictionary configuration."""
        config_dict = {
            "model": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 1000
        }
        
        interface = MockLLMInterface(config_dict)
        
        assert interface.provider_name == "mock"
        assert interface.config.model == "gpt-4"
        assert interface.config.temperature == 0.5
        assert interface.config.max_tokens == 1000
        assert len(interface.validate_config_calls) == 1
    
    def test_interface_initialization_with_llm_config(self):
        """Test initializing interface with LLMConfig object."""
        config = LLMConfig(model="gpt-3.5-turbo", temperature=0.7)
        
        interface = MockLLMInterface(config)
        
        assert interface.provider_name == "mock"
        assert interface.config.model == "gpt-3.5-turbo"
        assert interface.config.temperature == 0.7
        assert len(interface.validate_config_calls) == 1
    
    def test_interface_initialization_with_invalid_config(self):
        """Test initializing interface with invalid configuration."""
        config_dict = {
            "model": "test",
            "temperature": 3.0  # Invalid temperature
        }
        
        with pytest.raises(LLMConfigurationError):
            MockLLMInterface(config_dict)
    
    def test_interface_initialization_with_validation_failure(self):
        """Test initializing interface when validation fails."""
        config_dict = {"model": "test"}
        
        with patch.object(MockLLMInterface, 'validate_config', side_effect=LLMConfigurationError("Test error")):
            with pytest.raises(LLMConfigurationError):
                MockLLMInterface(config_dict)
    
    def test_invoke_method(self):
        """Test the invoke method."""
        interface = MockLLMInterface({"model": "test"})
        
        result = interface.invoke("test input", config=None, extra_param="value")
        
        assert result == "Mock response"
        assert len(interface.invoke_calls) == 1
        assert interface.invoke_calls[0][0] == "test input"
        assert interface.invoke_calls[0][1] is None
        assert interface.invoke_calls[0][2] == {"extra_param": "value"}
    
    @pytest.mark.asyncio
    async def test_ainvoke_method(self):
        """Test the async invoke method."""
        interface = MockLLMInterface({"model": "test"})
        
        result = await interface.ainvoke("test input", config=None, extra_param="value")
        
        assert result == "Mock response"
        assert len(interface.ainvoke_calls) == 1
        assert interface.ainvoke_calls[0][0] == "test input"
        assert interface.ainvoke_calls[0][1] is None
        assert interface.ainvoke_calls[0][2] == {"extra_param": "value"}
    
    def test_stream_method_fallback(self):
        """Test stream method falls back to invoke."""
        interface = MockLLMInterface({"model": "test"})
        
        # Test streaming (should fall back to invoke)
        results = list(interface.stream("test input"))
        
        assert len(results) == 1
        assert results[0] == "Mock response"
        assert len(interface.invoke_calls) == 1
    
    @pytest.mark.asyncio
    async def test_astream_method_fallback(self):
        """Test async stream method falls back to ainvoke."""
        interface = MockLLMInterface({"model": "test"})
        
        # Test async streaming (should fall back to ainvoke)
        results = []
        async for result in interface.astream("test input"):
            results.append(result)
        
        assert len(results) == 1
        assert results[0] == "Mock response"
        assert len(interface.ainvoke_calls) == 1
    
    def test_batch_method(self):
        """Test batch processing method."""
        interface = MockLLMInterface({"model": "test"})
        
        inputs = ["input1", "input2", "input3"]
        results = interface.batch(inputs)
        
        assert len(results) == 3
        assert all(result == "Mock response" for result in results)
        assert len(interface.invoke_calls) == 3
    
    @pytest.mark.asyncio
    async def test_abatch_method(self):
        """Test async batch processing method."""
        interface = MockLLMInterface({"model": "test"})
        
        inputs = ["input1", "input2", "input3"]
        results = await interface.abatch(inputs)
        
        assert len(results) == 3
        assert all(result == "Mock response" for result in results)
        assert len(interface.ainvoke_calls) == 3
    
    def test_health_check_methods(self):
        """Test health check related methods."""
        interface = MockLLMInterface({"model": "test"})
        
        # Test initial health status
        assert interface.get_cached_health_status() == "unknown"
        assert interface.get_last_health_check() is None
        
        # Test refresh health status
        is_healthy = interface.refresh_health_status()
        
        assert is_healthy is True
        assert interface.get_cached_health_status() == "healthy"
        assert isinstance(interface.get_last_health_check(), datetime)
        assert len(interface.health_check_calls) == 1
    
    def test_health_check_failure(self):
        """Test health check failure handling."""
        interface = MockLLMInterface({"model": "test"})
        interface._should_fail_health_check = True

        with pytest.raises(LLMProviderError):
            interface.refresh_health_status()
        
        assert interface.get_cached_health_status() == "error"
        assert isinstance(interface.get_last_health_check(), datetime)
    
    def test_config_update_success(self):
        """Test successful configuration update."""
        interface = MockLLMInterface({"model": "test"})
        
        new_config = {"model": "new-model", "temperature": 0.8}
        interface.update_config(new_config)
        
        assert interface.config.model == "new-model"
        assert interface.config.temperature == 0.8
        # Should have been called twice: once during init, once during update
        assert len(interface.validate_config_calls) == 2
    
    def test_config_update_validation_failure(self):
        """Test configuration update with validation failure."""
        interface = MockLLMInterface({"model": "test"})
        original_model = interface.config.model
        
        # Make validation fail
        interface._should_fail_validation = True
        
        new_config = {"model": "new-model"}
        with pytest.raises(LLMConfigurationError):
            interface.update_config(new_config)
        
        # Configuration should be rolled back
        assert interface.config.model == original_model
    
    def test_config_update_with_llm_config_object(self):
        """Test configuration update with LLMConfig object."""
        interface = MockLLMInterface({"model": "test"})
        
        new_config = LLMConfig(model="new-model", temperature=0.9)
        interface.update_config(new_config)
        
        assert interface.config.model == "new-model"
        assert interface.config.temperature == 0.9
    
    def test_string_representations(self):
        """Test string representation methods."""
        interface = MockLLMInterface({"model": "test-model"})
        
        repr_str = repr(interface)
        str_str = str(interface)
        
        assert "MockLLMInterface" in repr_str
        assert "mock" in repr_str
        assert "test-model" in repr_str
        
        assert "mock" in str_str
        assert "test-model" in str_str
    
    def test_provider_info_method(self):
        """Test get_provider_info method."""
        interface = MockLLMInterface({"model": "test"})
        
        info = interface.get_provider_info()
        
        assert isinstance(info, ProviderInfo)
        assert info.name == "mock"
        assert info.version == "1.0.0"
        assert info.models == ["mock-model"]
        assert info.capabilities == {"chat": True}
        assert info.limits == {"max_tokens": 1000}


class TestLLMInterfaceErrorHandling:
    """Tests for LLM interface error handling."""
    
    def test_handle_provider_exception(self):
        """Test provider exception handling."""
        interface = MockLLMInterface({"model": "test"})
        
        original_exception = ValueError("Test error")
        mapped_error = interface._handle_provider_exception(original_exception, "test context")
        
        assert mapped_error.provider == "mock"
        assert "Test error" in str(mapped_error)
        assert mapped_error.original_exception == original_exception
    
    def test_batch_processing_with_exception(self):
        """Test batch processing with exception handling."""
        interface = MockLLMInterface({"model": "test"})
        
        # Mock invoke to raise exception
        def mock_invoke_with_error(input_data, config=None, **kwargs):
            if input_data == "error_input":
                raise ValueError("Test error")
            return "Mock response"
        
        interface.invoke = mock_invoke_with_error
        
        inputs = ["good_input", "error_input", "good_input2"]
        
        with pytest.raises(Exception):
            interface.batch(inputs)
    
    @pytest.mark.asyncio
    async def test_abatch_processing_with_exception(self):
        """Test async batch processing with exception handling."""
        interface = MockLLMInterface({"model": "test"})
        
        # Mock ainvoke to raise exception
        async def mock_ainvoke_with_error(input_data, config=None, **kwargs):
            if input_data == "error_input":
                raise ValueError("Test error")
            return "Mock response"
        
        interface.ainvoke = mock_ainvoke_with_error
        
        inputs = ["good_input", "error_input", "good_input2"]
        
        with pytest.raises(Exception):
            await interface.abatch(inputs)


if __name__ == "__main__":
    pytest.main([__file__])