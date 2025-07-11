"""Unit tests for LLM exception handling infrastructure.

Tests the exception hierarchy and provider exception mapping functionality.
"""

import pytest
from unittest.mock import Mock, patch

from promptmatryoshka.exceptions import (
    LLMError,
    LLMConfigurationError,
    LLMProviderError,
    LLMConnectionError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMInvalidResponseError,
    LLMModelNotFoundError,
    LLMQuotaExceededError,
    LLMValidationError,
    LLMUnsupportedProviderError,
    LLMHealthCheckError,
    map_provider_exception
)


class TestLLMError:
    """Tests for base LLMError class."""
    
    def test_basic_error_creation(self):
        """Test basic error creation."""
        error = LLMError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.provider is None
        assert error.error_code is None
        assert error.details == {}
        assert error.original_exception is None
    
    def test_error_with_full_details(self):
        """Test error creation with all details."""
        original_exception = ValueError("Original error")
        details = {"key": "value", "number": 42}
        
        error = LLMError(
            message="Test error",
            provider="test_provider",
            error_code="TEST_001",
            details=details,
            original_exception=original_exception
        )
        
        assert error.message == "Test error"
        assert error.provider == "test_provider"
        assert error.error_code == "TEST_001"
        assert error.details == details
        assert error.original_exception == original_exception
    
    def test_error_string_representation(self):
        """Test error string representation."""
        error = LLMError(
            message="Test error",
            provider="test_provider",
            error_code="TEST_001"
        )
        
        error_str = str(error)
        assert "Test error" in error_str
        assert "Provider: test_provider" in error_str
        assert "Code: TEST_001" in error_str
    
    def test_error_to_dict(self):
        """Test error serialization to dictionary."""
        error = LLMError(
            message="Test error",
            provider="test_provider",
            error_code="TEST_001",
            details={"key": "value"}
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "LLMError"
        assert error_dict["message"] == "Test error"
        assert error_dict["provider"] == "test_provider"
        assert error_dict["error_code"] == "TEST_001"
        assert error_dict["details"] == {"key": "value"}
    
    @patch('promptmatryoshka.exceptions.logger')
    def test_error_logging(self, mock_logger):
        """Test that errors are logged on creation."""
        LLMError("Test error", provider="test_provider", error_code="TEST_001")
        
        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args[0][0]
        assert "LLMError occurred" in call_args
        assert "Test error" in call_args
        assert "test_provider" in call_args
        assert "TEST_001" in call_args


class TestLLMConfigurationError:
    """Tests for LLMConfigurationError class."""
    
    def test_configuration_error_creation(self):
        """Test configuration error creation."""
        error = LLMConfigurationError(
            message="Invalid configuration",
            config_key="temperature",
            provider="openai"
        )
        
        assert error.message == "Invalid configuration"
        assert error.config_key == "temperature"
        assert error.provider == "openai"
    
    def test_configuration_error_inheritance(self):
        """Test that LLMConfigurationError inherits from LLMError."""
        error = LLMConfigurationError("Test error")
        
        assert isinstance(error, LLMError)
        assert isinstance(error, LLMConfigurationError)


class TestLLMRateLimitError:
    """Tests for LLMRateLimitError class."""
    
    def test_rate_limit_error_creation(self):
        """Test rate limit error creation."""
        error = LLMRateLimitError(
            message="Rate limit exceeded",
            retry_after=60,
            limit_type="requests_per_minute",
            provider="openai"
        )
        
        assert error.message == "Rate limit exceeded"
        assert error.retry_after == 60
        assert error.limit_type == "requests_per_minute"
        assert error.provider == "openai"
    
    def test_rate_limit_error_inheritance(self):
        """Test that LLMRateLimitError inherits from LLMError."""
        error = LLMRateLimitError("Test error")
        
        assert isinstance(error, LLMError)
        assert isinstance(error, LLMRateLimitError)


class TestLLMTimeoutError:
    """Tests for LLMTimeoutError class."""
    
    def test_timeout_error_creation(self):
        """Test timeout error creation."""
        error = LLMTimeoutError(
            message="Request timed out",
            timeout_duration=30.0,
            provider="anthropic"
        )
        
        assert error.message == "Request timed out"
        assert error.timeout_duration == 30.0
        assert error.provider == "anthropic"
    
    def test_timeout_error_inheritance(self):
        """Test that LLMTimeoutError inherits from LLMError."""
        error = LLMTimeoutError("Test error")
        
        assert isinstance(error, LLMError)
        assert isinstance(error, LLMTimeoutError)


class TestLLMInvalidResponseError:
    """Tests for LLMInvalidResponseError class."""
    
    def test_invalid_response_error_creation(self):
        """Test invalid response error creation."""
        error = LLMInvalidResponseError(
            message="Invalid response format",
            response_content="malformed json",
            provider="huggingface"
        )
        
        assert error.message == "Invalid response format"
        assert error.response_content == "malformed json"
        assert error.provider == "huggingface"
    
    def test_invalid_response_error_inheritance(self):
        """Test that LLMInvalidResponseError inherits from LLMError."""
        error = LLMInvalidResponseError("Test error")
        
        assert isinstance(error, LLMError)
        assert isinstance(error, LLMInvalidResponseError)


class TestLLMModelNotFoundError:
    """Tests for LLMModelNotFoundError class."""
    
    def test_model_not_found_error_creation(self):
        """Test model not found error creation."""
        error = LLMModelNotFoundError(
            message="Model not found",
            model_name="gpt-5",
            provider="openai"
        )
        
        assert error.message == "Model not found"
        assert error.model_name == "gpt-5"
        assert error.provider == "openai"
    
    def test_model_not_found_error_inheritance(self):
        """Test that LLMModelNotFoundError inherits from LLMError."""
        error = LLMModelNotFoundError("Test error")
        
        assert isinstance(error, LLMError)
        assert isinstance(error, LLMModelNotFoundError)


class TestLLMQuotaExceededError:
    """Tests for LLMQuotaExceededError class."""
    
    def test_quota_exceeded_error_creation(self):
        """Test quota exceeded error creation."""
        error = LLMQuotaExceededError(
            message="Quota exceeded",
            quota_type="tokens",
            current_usage=10000,
            quota_limit=10000,
            provider="openai"
        )
        
        assert error.message == "Quota exceeded"
        assert error.quota_type == "tokens"
        assert error.current_usage == 10000
        assert error.quota_limit == 10000
        assert error.provider == "openai"
    
    def test_quota_exceeded_error_inheritance(self):
        """Test that LLMQuotaExceededError inherits from LLMError."""
        error = LLMQuotaExceededError("Test error")
        
        assert isinstance(error, LLMError)
        assert isinstance(error, LLMQuotaExceededError)


class TestLLMValidationError:
    """Tests for LLMValidationError class."""
    
    def test_validation_error_creation(self):
        """Test validation error creation."""
        error = LLMValidationError(
            message="Validation failed",
            parameter_name="temperature",
            parameter_value=3.0,
            provider="openai"
        )
        
        assert error.message == "Validation failed"
        assert error.parameter_name == "temperature"
        assert error.parameter_value == 3.0
        assert error.provider == "openai"
    
    def test_validation_error_inheritance(self):
        """Test that LLMValidationError inherits from LLMError."""
        error = LLMValidationError("Test error")
        
        assert isinstance(error, LLMError)
        assert isinstance(error, LLMValidationError)


class TestLLMUnsupportedProviderError:
    """Tests for LLMUnsupportedProviderError class."""
    
    def test_unsupported_provider_error_creation(self):
        """Test unsupported provider error creation."""
        error = LLMUnsupportedProviderError(
            message="Provider not supported",
            provider_name="unknown_provider",
            supported_providers=["openai", "anthropic", "ollama"]
        )
        
        assert error.message == "Provider not supported"
        assert error.provider_name == "unknown_provider"
        assert error.supported_providers == ["openai", "anthropic", "ollama"]
    
    def test_unsupported_provider_error_inheritance(self):
        """Test that LLMUnsupportedProviderError inherits from LLMError."""
        error = LLMUnsupportedProviderError("Test error")
        
        assert isinstance(error, LLMError)
        assert isinstance(error, LLMUnsupportedProviderError)


class TestLLMHealthCheckError:
    """Tests for LLMHealthCheckError class."""
    
    def test_health_check_error_creation(self):
        """Test health check error creation."""
        error = LLMHealthCheckError(
            message="Health check failed",
            health_status="unhealthy",
            provider="ollama"
        )
        
        assert error.message == "Health check failed"
        assert error.health_status == "unhealthy"
        assert error.provider == "ollama"
    
    def test_health_check_error_inheritance(self):
        """Test that LLMHealthCheckError inherits from LLMError."""
        error = LLMHealthCheckError("Test error")
        
        assert isinstance(error, LLMError)
        assert isinstance(error, LLMHealthCheckError)


class TestMapProviderException:
    """Tests for map_provider_exception function."""
    
    def test_map_timeout_exception(self):
        """Test mapping timeout-related exceptions."""
        original_exception = Exception("Request timed out")
        
        mapped_error = map_provider_exception("openai", original_exception)
        
        assert isinstance(mapped_error, LLMTimeoutError)
        assert mapped_error.provider == "openai"
        assert mapped_error.original_exception == original_exception
        assert "Request timed out" in mapped_error.message
    
    def test_map_rate_limit_exception(self):
        """Test mapping rate limit-related exceptions."""
        original_exception = Exception("Rate limit exceeded")
        
        mapped_error = map_provider_exception("anthropic", original_exception)
        
        assert isinstance(mapped_error, LLMRateLimitError)
        assert mapped_error.provider == "anthropic"
        assert mapped_error.original_exception == original_exception
        assert "Rate limit exceeded" in mapped_error.message
    
    def test_map_authentication_exception(self):
        """Test mapping authentication-related exceptions."""
        original_exception = Exception("Invalid API key")
        
        mapped_error = map_provider_exception("openai", original_exception)
        
        assert isinstance(mapped_error, LLMAuthenticationError)
        assert mapped_error.provider == "openai"
        assert mapped_error.original_exception == original_exception
        assert "Invalid API key" in mapped_error.message
    
    def test_map_connection_exception(self):
        """Test mapping connection-related exceptions."""
        original_exception = Exception("Connection failed")
        
        mapped_error = map_provider_exception("ollama", original_exception)
        
        assert isinstance(mapped_error, LLMConnectionError)
        assert mapped_error.provider == "ollama"
        assert mapped_error.original_exception == original_exception
        assert "Connection failed" in mapped_error.message
    
    def test_map_model_not_found_exception(self):
        """Test mapping model not found exceptions."""
        original_exception = Exception("Model not found")
        
        mapped_error = map_provider_exception("huggingface", original_exception)
        
        assert isinstance(mapped_error, LLMModelNotFoundError)
        assert mapped_error.provider == "huggingface"
        assert mapped_error.original_exception == original_exception
        assert "Model not found" in mapped_error.message
    
    def test_map_quota_exceeded_exception(self):
        """Test mapping quota exceeded exceptions."""
        original_exception = Exception("Quota exceeded")
        
        mapped_error = map_provider_exception("openai", original_exception)
        
        assert isinstance(mapped_error, LLMQuotaExceededError)
        assert mapped_error.provider == "openai"
        assert mapped_error.original_exception == original_exception
        assert "Quota exceeded" in mapped_error.message
    
    def test_map_generic_exception(self):
        """Test mapping generic exceptions."""
        original_exception = Exception("Some unknown error")
        
        mapped_error = map_provider_exception("test_provider", original_exception)
        
        assert isinstance(mapped_error, LLMProviderError)
        assert mapped_error.provider == "test_provider"
        assert mapped_error.original_exception == original_exception
        assert "Some unknown error" in mapped_error.message
    
    def test_map_exception_with_context(self):
        """Test mapping exceptions with context."""
        original_exception = Exception("Test error")
        
        mapped_error = map_provider_exception("openai", original_exception, "test operation")
        
        assert isinstance(mapped_error, LLMProviderError)
        assert mapped_error.provider == "openai"
        assert mapped_error.original_exception == original_exception
        assert "Test error" in mapped_error.message
    
    def test_map_exception_case_insensitive(self):
        """Test that exception mapping is case insensitive."""
        original_exception = Exception("TIMEOUT ERROR")
        
        mapped_error = map_provider_exception("openai", original_exception)
        
        assert isinstance(mapped_error, LLMTimeoutError)
        assert mapped_error.provider == "openai"
    
    def test_map_multiple_keywords(self):
        """Test mapping with multiple keywords in error message."""
        original_exception = Exception("Connection timeout - rate limit exceeded")
        
        mapped_error = map_provider_exception("openai", original_exception)
        
        # Should match the first pattern (timeout in this case)
        assert isinstance(mapped_error, LLMTimeoutError)
        assert mapped_error.provider == "openai"


class TestExceptionInheritance:
    """Tests for exception inheritance hierarchy."""
    
    def test_all_exceptions_inherit_from_llm_error(self):
        """Test that all LLM exceptions inherit from LLMError."""
        exception_classes = [
            LLMConfigurationError,
            LLMProviderError,
            LLMConnectionError,
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMInvalidResponseError,
            LLMModelNotFoundError,
            LLMQuotaExceededError,
            LLMValidationError,
            LLMUnsupportedProviderError,
            LLMHealthCheckError
        ]
        
        for exception_class in exception_classes:
            assert issubclass(exception_class, LLMError)
            assert issubclass(exception_class, Exception)
    
    def test_exception_instances_are_catchable_as_llm_error(self):
        """Test that all exception instances can be caught as LLMError."""
        exceptions = [
            LLMConfigurationError("test"),
            LLMProviderError("test"),
            LLMConnectionError("test"),
            LLMAuthenticationError("test"),
            LLMRateLimitError("test"),
            LLMTimeoutError("test"),
            LLMInvalidResponseError("test"),
            LLMModelNotFoundError("test"),
            LLMQuotaExceededError("test"),
            LLMValidationError("test"),
            LLMUnsupportedProviderError("test"),
            LLMHealthCheckError("test")
        ]
        
        for exception in exceptions:
            assert isinstance(exception, LLMError)
            assert isinstance(exception, Exception)
    
    def test_exception_specific_catching(self):
        """Test that specific exceptions can be caught specifically."""
        try:
            raise LLMRateLimitError("Rate limit exceeded")
        except LLMRateLimitError as e:
            assert e.message == "Rate limit exceeded"
        except LLMError:
            pytest.fail("Should have caught LLMRateLimitError specifically")
        
        try:
            raise LLMConfigurationError("Config error")
        except LLMConfigurationError as e:
            assert e.message == "Config error"
        except LLMError:
            pytest.fail("Should have caught LLMConfigurationError specifically")


if __name__ == "__main__":
    pytest.main([__file__])