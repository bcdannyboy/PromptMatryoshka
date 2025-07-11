"""PromptMatryoshka package initializer.

Exposes main pipeline API and LLM interface infrastructure.
This module provides the public API for the PromptMatryoshka framework,
including the multi-provider LLM interface system and configuration management.
"""

# Core LLM infrastructure
from .llm_interface import LLMInterface, LLMConfig, ProviderInfo
from .llm_factory import LLMFactory, get_factory, create_llm_interface
from .exceptions import (
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

# Configuration system
from .config import (
    Config,
    ConfigurationError,
    PromptMatryoshkaConfig,
    ProviderConfig,
    ProfileConfig,
    PluginConfig,
    RateLimitConfig,
    get_config,
    load_config,
    reset_config,
    resolve_env_var
)

# Provider registry functions
from .providers import (
    register_provider,
    get_provider,
    list_providers,
    list_builtin_providers,
    is_provider_available,
    get_provider_info,
    discover_providers,
    clear_registry
)

# Version information
__version__ = "0.2.0"
__author__ = "PromptMatryoshka Team"
__description__ = "Multi-provider LLM interface for advanced prompt engineering techniques"

# Public API
__all__ = [
    # Core LLM classes
    "LLMInterface",
    "LLMConfig",
    "ProviderInfo",
    "LLMFactory",
    
    # Factory functions
    "get_factory",
    "create_llm_interface",
    
    # Configuration system
    "Config",
    "ConfigurationError",
    "PromptMatryoshkaConfig",
    "ProviderConfig",
    "ProfileConfig",
    "PluginConfig",
    "RateLimitConfig",
    "get_config",
    "load_config",
    "reset_config",
    "resolve_env_var",
    
    # Exception classes
    "LLMError",
    "LLMConfigurationError",
    "LLMProviderError",
    "LLMConnectionError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMInvalidResponseError",
    "LLMModelNotFoundError",
    "LLMQuotaExceededError",
    "LLMValidationError",
    "LLMUnsupportedProviderError",
    "LLMHealthCheckError",
    "map_provider_exception",
    
    # Provider registry
    "register_provider",
    "get_provider",
    "list_providers",
    "list_builtin_providers",
    "is_provider_available",
    "get_provider_info",
    "discover_providers",
    "clear_registry",
]