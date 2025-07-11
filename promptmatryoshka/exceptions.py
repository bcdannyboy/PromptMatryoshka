"""Exception classes for PromptMatryoshka LLM interface.

This module defines a comprehensive hierarchy of exceptions for handling
various failure modes in the multi-provider LLM interface system.

Classes:
    LLMError: Base exception class for all LLM-related errors.
    LLMConfigurationError: Configuration validation errors.
    LLMProviderError: Provider-specific errors.
    LLMConnectionError: Network and connection errors.
    LLMAuthenticationError: Authentication and API key errors.
    LLMRateLimitError: Rate limiting errors.
    LLMTimeoutError: Request timeout errors.
    LLMInvalidResponseError: Invalid or malformed response errors.
    LLMModelNotFoundError: Model not available errors.
    LLMQuotaExceededError: Quota or usage limit errors.
    LLMValidationError: Input validation errors.
    LLMUnsupportedProviderError: Unsupported provider errors.
    LLMHealthCheckError: Health check failure errors.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception class for all LLM-related errors.
    
    This is the root exception that all other LLM exceptions inherit from.
    It provides common functionality for error handling and logging.
    
    Attributes:
        message: Human-readable error message
        provider: The LLM provider that caused the error (optional)
        error_code: Provider-specific error code (optional)
        details: Additional error details (optional)
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """Initialize LLMError.
        
        Args:
            message: Human-readable error message
            provider: The LLM provider that caused the error
            error_code: Provider-specific error code
            details: Additional error details
            original_exception: The original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
        
        # Log the error for debugging
        logger.error(f"LLMError occurred: {message} (provider: {provider}, code: {error_code})")
        if original_exception:
            logger.error(f"Original exception: {original_exception}")
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        parts = [self.message]
        if self.provider:
            parts.append(f"Provider: {self.provider}")
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "provider": self.provider,
            "error_code": self.error_code,
            "details": self.details
        }


class LLMConfigurationError(LLMError):
    """Raised when LLM configuration is invalid or missing.
    
    This exception is raised when:
    - Required configuration parameters are missing
    - Configuration values are invalid or out of range
    - Provider-specific configuration is malformed
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        """Initialize LLMConfigurationError.
        
        Args:
            message: Human-readable error message
            config_key: The configuration key that caused the error
            provider: The LLM provider
            **kwargs: Additional arguments passed to LLMError
        """
        super().__init__(message, provider=provider, **kwargs)
        self.config_key = config_key


class LLMProviderError(LLMError):
    """Raised when a provider-specific error occurs.
    
    This is a general exception for provider-specific errors that don't
    fit into other more specific categories.
    """
    pass


class LLMConnectionError(LLMError):
    """Raised when connection to LLM provider fails.
    
    This exception is raised when:
    - Network connection fails
    - DNS resolution fails
    - SSL/TLS handshake fails
    - Connection timeout occurs
    """
    pass


class LLMAuthenticationError(LLMError):
    """Raised when authentication with LLM provider fails.
    
    This exception is raised when:
    - API key is missing or invalid
    - Authentication token is expired
    - Access is denied for the requested resource
    """
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded.
    
    This exception is raised when:
    - API rate limit is exceeded
    - Request quota is exhausted
    - Temporary throttling is in effect
    
    Attributes:
        retry_after: Seconds to wait before retrying (optional)
        limit_type: Type of limit that was exceeded (optional)
    """
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        limit_type: Optional[str] = None,
        **kwargs
    ):
        """Initialize LLMRateLimitError.
        
        Args:
            message: Human-readable error message
            retry_after: Seconds to wait before retrying
            limit_type: Type of limit that was exceeded
            **kwargs: Additional arguments passed to LLMError
        """
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.limit_type = limit_type


class LLMTimeoutError(LLMError):
    """Raised when request times out.
    
    This exception is raised when:
    - Request timeout is exceeded
    - Read timeout occurs
    - Connection timeout occurs
    """
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        **kwargs
    ):
        """Initialize LLMTimeoutError.
        
        Args:
            message: Human-readable error message
            timeout_duration: Duration of timeout in seconds
            **kwargs: Additional arguments passed to LLMError
        """
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration


class LLMInvalidResponseError(LLMError):
    """Raised when LLM returns invalid or malformed response.
    
    This exception is raised when:
    - Response format is invalid
    - Response cannot be parsed
    - Response is empty or truncated
    - Response contains unexpected content
    """
    
    def __init__(
        self,
        message: str,
        response_content: Optional[str] = None,
        **kwargs
    ):
        """Initialize LLMInvalidResponseError.
        
        Args:
            message: Human-readable error message
            response_content: The invalid response content
            **kwargs: Additional arguments passed to LLMError
        """
        super().__init__(message, **kwargs)
        self.response_content = response_content


class LLMModelNotFoundError(LLMError):
    """Raised when requested model is not available.
    
    This exception is raised when:
    - Model name is invalid or not found
    - Model is not accessible with current permissions
    - Model is deprecated or discontinued
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize LLMModelNotFoundError.
        
        Args:
            message: Human-readable error message
            model_name: The model name that was not found
            **kwargs: Additional arguments passed to LLMError
        """
        super().__init__(message, **kwargs)
        self.model_name = model_name


class LLMQuotaExceededError(LLMError):
    """Raised when usage quota is exceeded.
    
    This exception is raised when:
    - API usage quota is exceeded
    - Token limit is reached
    - Billing quota is exhausted
    """
    
    def __init__(
        self,
        message: str,
        quota_type: Optional[str] = None,
        current_usage: Optional[int] = None,
        quota_limit: Optional[int] = None,
        **kwargs
    ):
        """Initialize LLMQuotaExceededError.
        
        Args:
            message: Human-readable error message
            quota_type: Type of quota that was exceeded
            current_usage: Current usage amount
            quota_limit: Quota limit
            **kwargs: Additional arguments passed to LLMError
        """
        super().__init__(message, **kwargs)
        self.quota_type = quota_type
        self.current_usage = current_usage
        self.quota_limit = quota_limit


class LLMValidationError(LLMError):
    """Raised when input validation fails.
    
    This exception is raised when:
    - Input parameters are invalid
    - Required parameters are missing
    - Parameter values are out of range
    - Input format is incorrect
    """
    
    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_value: Optional[Any] = None,
        **kwargs
    ):
        """Initialize LLMValidationError.
        
        Args:
            message: Human-readable error message
            parameter_name: Name of the parameter that failed validation
            parameter_value: Value of the parameter that failed validation
            **kwargs: Additional arguments passed to LLMError
        """
        super().__init__(message, **kwargs)
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value


class LLMUnsupportedProviderError(LLMError):
    """Raised when an unsupported provider is requested.
    
    This exception is raised when:
    - Provider name is not recognized
    - Provider is not implemented
    - Provider is disabled or unavailable
    """
    
    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        supported_providers: Optional[list] = None,
        **kwargs
    ):
        """Initialize LLMUnsupportedProviderError.
        
        Args:
            message: Human-readable error message
            provider_name: Name of the unsupported provider
            supported_providers: List of supported providers
            **kwargs: Additional arguments passed to LLMError
        """
        super().__init__(message, **kwargs)
        self.provider_name = provider_name
        self.supported_providers = supported_providers or []


class LLMHealthCheckError(LLMError):
    """Raised when health check fails.
    
    This exception is raised when:
    - Provider health check fails
    - Service is unavailable
    - Provider is in maintenance mode
    """
    
    def __init__(
        self,
        message: str,
        health_status: Optional[str] = None,
        **kwargs
    ):
        """Initialize LLMHealthCheckError.
        
        Args:
            message: Human-readable error message
            health_status: Current health status
            **kwargs: Additional arguments passed to LLMError
        """
        super().__init__(message, **kwargs)
        self.health_status = health_status


def map_provider_exception(
    provider: str,
    original_exception: Exception,
    context: Optional[str] = None
) -> LLMError:
    """Map provider-specific exceptions to standardized LLM exceptions.
    
    This function maps exceptions from different providers to our standardized
    exception hierarchy, providing consistent error handling across providers.
    
    Args:
        provider: Name of the LLM provider
        original_exception: The original exception from the provider
        context: Additional context about when the error occurred
        
    Returns:
        LLMError: Standardized exception instance
    """
    error_message = str(original_exception)
    
    # Common patterns for different error types
    if any(keyword in error_message.lower() for keyword in ['timeout', 'timed out']):
        return LLMTimeoutError(
            f"Request timed out for {provider}: {error_message}",
            provider=provider,
            original_exception=original_exception
        )
    
    if any(keyword in error_message.lower() for keyword in ['rate limit', 'rate_limit', 'too many requests']):
        return LLMRateLimitError(
            f"Rate limit exceeded for {provider}: {error_message}",
            provider=provider,
            original_exception=original_exception
        )
    
    if any(keyword in error_message.lower() for keyword in ['unauthorized', 'invalid api key', 'authentication']):
        return LLMAuthenticationError(
            f"Authentication failed for {provider}: {error_message}",
            provider=provider,
            original_exception=original_exception
        )
    
    if any(keyword in error_message.lower() for keyword in ['connection', 'network', 'dns']):
        return LLMConnectionError(
            f"Connection failed for {provider}: {error_message}",
            provider=provider,
            original_exception=original_exception
        )
    
    if any(keyword in error_message.lower() for keyword in ['model not found', 'model does not exist']):
        return LLMModelNotFoundError(
            f"Model not found for {provider}: {error_message}",
            provider=provider,
            original_exception=original_exception
        )
    
    if any(keyword in error_message.lower() for keyword in ['quota', 'exceeded', 'limit reached']):
        return LLMQuotaExceededError(
            f"Quota exceeded for {provider}: {error_message}",
            provider=provider,
            original_exception=original_exception
        )
    
    # Default to generic provider error
    return LLMProviderError(
        f"Provider error for {provider}: {error_message}",
        provider=provider,
        original_exception=original_exception
    )