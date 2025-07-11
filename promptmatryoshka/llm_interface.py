"""Abstract LLM interface for PromptMatryoshka.

This module provides the abstract base class for all LLM provider implementations
in the PromptMatryoshka framework. It defines a standardized interface that abstracts
away the complexities of different LLM providers and APIs, enabling consistent
interaction with various language models.

The interface inherits from LangChain's Runnable to provide native compatibility
with LangChain's LCEL (LangChain Expression Language) and chain composition.

Classes:
    LLMInterface: Abstract base class for all LLM provider implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, AsyncIterator, Iterator
import asyncio
from datetime import datetime
import logging

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings

from .exceptions import (
    LLMError,
    LLMConfigurationError,
    LLMValidationError,
    LLMHealthCheckError,
    map_provider_exception
)
from .logging_utils import get_logger


class LLMConfig(BaseModel):
    """Configuration model for LLM providers.
    
    This model defines the standard configuration parameters that all
    LLM providers should support, with provider-specific extensions allowed.
    
    Attributes:
        model: The model name to use
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum number of tokens to generate
        top_p: Nucleus sampling parameter (0.0 to 1.0)
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
        request_timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        api_key: API key for authentication (optional)
        api_base: Base URL for API (optional)
        organization: Organization ID (optional)
        custom_headers: Custom headers to include in requests (optional)
        provider_specific: Provider-specific configuration (optional)
    """
    
    model: str = Field(..., description="Model name to use")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(2000, gt=0, description="Maximum tokens to generate")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    request_timeout: int = Field(120, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")
    retry_delay: float = Field(1.0, ge=0.0, description="Delay between retries")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    api_base: Optional[str] = Field(None, description="Base URL for API")
    organization: Optional[str] = Field(None, description="Organization ID")
    custom_headers: Optional[Dict[str, str]] = Field(None, description="Custom headers")
    provider_specific: Optional[Dict[str, Any]] = Field(None, description="Provider-specific config")
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature parameter."""
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
    
    @field_validator('top_p')
    @classmethod
    def validate_top_p(cls, v):
        """Validate top_p parameter."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('top_p must be between 0.0 and 1.0')
        return v
    
    model_config = ConfigDict(extra="allow")  # Allow provider-specific fields


class ProviderInfo(BaseModel):
    """Information about an LLM provider.
    
    Attributes:
        name: Provider name
        version: Provider version
        models: List of supported models
        capabilities: Provider capabilities
        limits: Provider limits and constraints
        health_status: Current health status
        last_health_check: Timestamp of last health check
    """
    
    name: str = Field(..., description="Provider name")
    version: str = Field(..., description="Provider version")
    models: List[str] = Field(default_factory=list, description="Supported models")
    capabilities: Dict[str, Any] = Field(default_factory=dict, description="Provider capabilities")
    limits: Dict[str, Any] = Field(default_factory=dict, description="Provider limits")
    health_status: str = Field("unknown", description="Current health status")
    last_health_check: Optional[datetime] = Field(None, description="Last health check timestamp")


class LLMInterface(Runnable, ABC):
    """Abstract base class for LLM provider implementations.
    
    This class provides the foundation for all LLM provider implementations
    in the PromptMatryoshka framework. It inherits from LangChain's Runnable
    to provide native compatibility with LangChain's ecosystem.
    
    All concrete implementations must implement the abstract methods defined here.
    The class provides standardized error handling, logging, and configuration
    management that all providers can utilize.
    
    Attributes:
        config: LLM configuration
        provider_name: Name of the provider
        logger: Logger instance for this provider
        _health_status: Cached health status
        _last_health_check: Timestamp of last health check
    """
    
    def __init__(
        self,
        config: Union[LLMConfig, Dict[str, Any]],
        provider_name: str,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the LLM interface.
        
        Args:
            config: LLM configuration (LLMConfig instance or dict)
            provider_name: Name of the provider
            logger: Optional logger instance
            
        Raises:
            LLMConfigurationError: If configuration is invalid
        """
        super().__init__()
        
        # Convert dict to LLMConfig if needed
        if isinstance(config, dict):
            try:
                self.config = LLMConfig(**config)
            except Exception as e:
                raise LLMConfigurationError(
                    f"Invalid configuration for {provider_name}: {str(e)}",
                    provider=provider_name,
                    original_exception=e
                )
        else:
            self.config = config
        
        self.provider_name = provider_name
        self.logger = logger or get_logger(f"LLM.{provider_name}")
        self._health_status = "unknown"
        self._last_health_check: Optional[datetime] = None
        
        # Validate configuration
        try:
            self.validate_config()
        except Exception as e:
            raise LLMConfigurationError(
                f"Configuration validation failed for {provider_name}: {str(e)}",
                provider=provider_name,
                original_exception=e
            )
        
        self.logger.info(f"Initialized {provider_name} LLM interface with model: {self.config.model}")
    
    # Abstract methods that must be implemented by providers
    
    @abstractmethod
    def invoke(
        self,
        input: Union[str, List[BaseMessage], Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> Union[str, BaseMessage, LLMResult]:
        """Invoke the LLM with the given input.
        
        This method sends the input to the LLM and returns the response.
        It should handle all provider-specific logic for making the API call.
        
        Args:
            input: The input to send to the LLM
            config: Optional runnable configuration
            **kwargs: Additional keyword arguments
            
        Returns:
            The LLM response
            
        Raises:
            LLMError: If the LLM call fails
        """
        pass
    
    @abstractmethod
    async def ainvoke(
        self,
        input: Union[str, List[BaseMessage], Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> Union[str, BaseMessage, LLMResult]:
        """Asynchronously invoke the LLM with the given input.
        
        This is the async version of invoke(). It should handle all
        provider-specific logic for making async API calls.
        
        Args:
            input: The input to send to the LLM
            config: Optional runnable configuration
            **kwargs: Additional keyword arguments
            
        Returns:
            The LLM response
            
        Raises:
            LLMError: If the LLM call fails
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the current configuration.
        
        This method should check that all required configuration parameters
        are present and valid for the specific provider.
        
        Returns:
            True if configuration is valid
            
        Raises:
            LLMConfigurationError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Perform a health check on the LLM provider.
        
        This method should verify that the provider is accessible and
        functioning correctly. It typically involves making a simple
        test API call.
        
        Returns:
            True if the provider is healthy
            
        Raises:
            LLMHealthCheckError: If the health check fails
        """
        pass
    
    @abstractmethod
    def get_provider_info(self) -> ProviderInfo:
        """Get information about the provider.
        
        This method should return detailed information about the provider,
        including supported models, capabilities, and current status.
        
        Returns:
            ProviderInfo instance with provider details
        """
        pass
    
    # Concrete methods with default implementations
    
    def stream(
        self,
        input: Union[str, List[BaseMessage], Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> Iterator[Union[str, BaseMessage]]:
        """Stream the LLM response.
        
        Default implementation that falls back to invoke() if streaming
        is not supported by the provider.
        
        Args:
            input: The input to send to the LLM
            config: Optional runnable configuration
            **kwargs: Additional keyword arguments
            
        Yields:
            Chunks of the LLM response
        """
        self.logger.warning(f"Streaming not implemented for {self.provider_name}, falling back to invoke()")
        response = self.invoke(input, config, **kwargs)
        yield response
    
    async def astream(
        self,
        input: Union[str, List[BaseMessage], Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> AsyncIterator[Union[str, BaseMessage]]:
        """Asynchronously stream the LLM response.
        
        Default implementation that falls back to ainvoke() if streaming
        is not supported by the provider.
        
        Args:
            input: The input to send to the LLM
            config: Optional runnable configuration
            **kwargs: Additional keyword arguments
            
        Yields:
            Chunks of the LLM response
        """
        self.logger.warning(f"Async streaming not implemented for {self.provider_name}, falling back to ainvoke()")
        response = await self.ainvoke(input, config, **kwargs)
        yield response
    
    def batch(
        self,
        inputs: List[Union[str, List[BaseMessage], Dict[str, Any]]],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        **kwargs: Any
    ) -> List[Union[str, BaseMessage, LLMResult]]:
        """Process multiple inputs in batch.
        
        Default implementation that processes inputs sequentially.
        Providers can override this for more efficient batch processing.
        
        Args:
            inputs: List of inputs to process
            config: Optional runnable configuration(s)
            **kwargs: Additional keyword arguments
            
        Returns:
            List of LLM responses
        """
        if config is None:
            configs = [None] * len(inputs)
        elif isinstance(config, list):
            configs = config
        else:
            configs = [config] * len(inputs)
        
        results = []
        for input_item, config_item in zip(inputs, configs):
            try:
                result = self.invoke(input_item, config_item, **kwargs)
                results.append(result)
            except Exception as e:
                mapped_error = map_provider_exception(self.provider_name, e, "batch processing")
                self.logger.error(f"Batch processing failed for input: {mapped_error}")
                raise mapped_error
        
        return results
    
    async def abatch(
        self,
        inputs: List[Union[str, List[BaseMessage], Dict[str, Any]]],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        **kwargs: Any
    ) -> List[Union[str, BaseMessage, LLMResult]]:
        """Asynchronously process multiple inputs in batch.
        
        Default implementation that processes inputs concurrently.
        Providers can override this for more efficient batch processing.
        
        Args:
            inputs: List of inputs to process
            config: Optional runnable configuration(s)
            **kwargs: Additional keyword arguments
            
        Returns:
            List of LLM responses
        """
        if config is None:
            configs = [None] * len(inputs)
        elif isinstance(config, list):
            configs = config
        else:
            configs = [config] * len(inputs)
        
        async def process_input(input_item, config_item):
            try:
                return await self.ainvoke(input_item, config_item, **kwargs)
            except Exception as e:
                mapped_error = map_provider_exception(self.provider_name, e, "async batch processing")
                self.logger.error(f"Async batch processing failed for input: {mapped_error}")
                raise mapped_error
        
        tasks = [process_input(input_item, config_item) for input_item, config_item in zip(inputs, configs)]
        return await asyncio.gather(*tasks)
    
    def get_cached_health_status(self) -> str:
        """Get the cached health status.
        
        Returns:
            Current health status string
        """
        return self._health_status
    
    def get_last_health_check(self) -> Optional[datetime]:
        """Get the timestamp of the last health check.
        
        Returns:
            Datetime of last health check or None if never checked
        """
        return self._last_health_check
    
    def refresh_health_status(self) -> bool:
        """Refresh the health status by performing a new health check.
        
        Returns:
            True if the provider is healthy
            
        Raises:
            LLMHealthCheckError: If the health check fails
        """
        try:
            is_healthy = self.health_check()
            self._health_status = "healthy" if is_healthy else "unhealthy"
            self._last_health_check = datetime.now()
            return is_healthy
        except Exception as e:
            self._health_status = "error"
            self._last_health_check = datetime.now()
            mapped_error = map_provider_exception(self.provider_name, e, "health check")
            self.logger.error(f"Health check failed: {mapped_error}")
            raise mapped_error
    
    def update_config(self, new_config: Union[LLMConfig, Dict[str, Any]]) -> None:
        """Update the configuration.
        
        Args:
            new_config: New configuration to apply
            
        Raises:
            LLMConfigurationError: If the new configuration is invalid
        """
        if isinstance(new_config, dict):
            try:
                config = LLMConfig(**new_config)
            except Exception as e:
                raise LLMConfigurationError(
                    f"Invalid configuration update for {self.provider_name}: {str(e)}",
                    provider=self.provider_name,
                    original_exception=e
                )
        else:
            config = new_config
        
        # Validate the new configuration
        old_config = self.config
        self.config = config
        
        try:
            self.validate_config()
            self.logger.info(f"Configuration updated for {self.provider_name}")
        except Exception as e:
            # Rollback on validation failure
            self.config = old_config
            raise LLMConfigurationError(
                f"Configuration validation failed for {self.provider_name}: {str(e)}",
                provider=self.provider_name,
                original_exception=e
            )
    
    def _handle_provider_exception(self, exception: Exception, context: str = None) -> LLMError:
        """Handle provider-specific exceptions.
        
        This method maps provider-specific exceptions to our standardized
        exception hierarchy and logs the error.
        
        Args:
            exception: The original exception
            context: Additional context about when the error occurred
            
        Returns:
            Standardized LLMError instance
        """
        mapped_error = map_provider_exception(self.provider_name, exception, context)
        self.logger.error(f"Provider exception in {context or 'operation'}: {mapped_error}")
        return mapped_error
    
    def __repr__(self) -> str:
        """Return string representation of the LLM interface."""
        return f"{self.__class__.__name__}(provider={self.provider_name}, model={self.config.model})"
    
    def __str__(self) -> str:
        """Return human-readable string representation."""
        return f"{self.provider_name} LLM Interface (model: {self.config.model})"