"""Factory pattern implementation for LLM providers in PromptMatryoshka.

This module provides the factory pattern implementation for creating LLM provider
instances. It handles provider discovery, configuration validation, and instance
creation with proper error handling.

Classes:
    LLMFactory: Factory class for creating LLM provider instances.
"""

from typing import Any, Dict, Optional, Union, List, Type
import logging
from datetime import datetime

from .llm_interface import LLMInterface, LLMConfig, ProviderInfo
from .exceptions import (
    LLMError,
    LLMConfigurationError,
    LLMUnsupportedProviderError,
    LLMValidationError,
    map_provider_exception
)
from .providers import (
    register_provider,
    get_provider,
    list_providers,
    is_provider_available,
    get_provider_info,
    discover_providers
)
from .logging_utils import get_logger


class LLMFactory:
    """Factory class for creating LLM provider instances.
    
    This class provides methods for creating LLM interfaces with different
    configuration sources and validation approaches. It handles provider
    discovery, configuration validation, and instance creation.
    
    The factory supports multiple creation methods:
    - create_interface: Create from explicit configuration
    - create_from_profile: Create from predefined profiles
    - create_for_plugin: Create for specific plugin usage
    
    Attributes:
        logger: Logger instance for factory operations
        _provider_cache: Cache for provider instances
        _config_profiles: Predefined configuration profiles
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the LLM factory.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or get_logger("LLMFactory")
        self._provider_cache: Dict[str, LLMInterface] = {}
        self._config_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default configuration profiles
        self._init_default_profiles()
        
        self.logger.info("LLM Factory initialized")
    
    def create_interface(
        self,
        provider: str,
        config: Union[LLMConfig, Dict[str, Any]],
        cache_key: Optional[str] = None,
        use_cache: bool = True,
        **kwargs: Any
    ) -> LLMInterface:
        """Create an LLM interface for the specified provider.
        
        This is the main factory method for creating LLM interfaces. It validates
        the provider, configuration, and creates the appropriate interface instance.
        
        Args:
            provider: Provider name (e.g., "openai", "anthropic", "ollama")
            config: LLM configuration (LLMConfig instance or dict)
            cache_key: Optional cache key for instance caching
            use_cache: Whether to use cached instances
            **kwargs: Additional provider-specific arguments
            
        Returns:
            LLMInterface instance for the specified provider
            
        Raises:
            LLMUnsupportedProviderError: If provider is not supported
            LLMConfigurationError: If configuration is invalid
            LLMError: If interface creation fails
        """
        provider = provider.lower().strip()
        
        # Generate cache key if not provided
        if cache_key is None:
            cache_key = self._generate_cache_key(provider, config)
        
        # Check cache first
        if use_cache and cache_key in self._provider_cache:
            self.logger.debug(f"Returning cached interface for {provider} (key: {cache_key})")
            return self._provider_cache[cache_key]
        
        # Validate provider availability
        if not is_provider_available(provider):
            available_providers = list_providers()
            self.logger.error(f"Provider '{provider}' is not available")
            raise LLMUnsupportedProviderError(
                f"Provider '{provider}' is not available",
                provider_name=provider,
                supported_providers=available_providers
            )
        
        # Get provider class
        try:
            provider_class = get_provider(provider)
        except Exception as e:
            self.logger.error(f"Failed to get provider class for '{provider}': {e}")
            raise LLMUnsupportedProviderError(
                f"Failed to get provider '{provider}'",
                provider_name=provider,
                original_exception=e
            )
        
        # Create provider instance
        try:
            self.logger.info(f"Creating {provider} interface with config: {type(config).__name__}")
            interface = provider_class(config, **kwargs)
            
            # Cache the instance if requested
            if use_cache:
                self._provider_cache[cache_key] = interface
                self.logger.debug(f"Cached interface for {provider} (key: {cache_key})")
            
            return interface
            
        except Exception as e:
            mapped_error = map_provider_exception(provider, e, "interface creation")
            self.logger.error(f"Failed to create {provider} interface: {mapped_error}")
            raise mapped_error
    
    def create_from_profile(
        self,
        profile_name: str,
        overrides: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
        use_cache: bool = True,
        **kwargs: Any
    ) -> LLMInterface:
        """Create an LLM interface from a predefined configuration profile.
        
        Configuration profiles provide predefined settings for common use cases.
        They can be overridden with specific parameters as needed.
        
        Args:
            profile_name: Name of the configuration profile
            overrides: Optional configuration overrides
            cache_key: Optional cache key for instance caching
            use_cache: Whether to use cached instances
            **kwargs: Additional provider-specific arguments
            
        Returns:
            LLMInterface instance configured with the profile
            
        Raises:
            LLMConfigurationError: If profile is not found or invalid
            LLMError: If interface creation fails
        """
        if profile_name not in self._config_profiles:
            available_profiles = list(self._config_profiles.keys())
            raise LLMConfigurationError(
                f"Configuration profile '{profile_name}' not found",
                config_key=profile_name,
                details={"available_profiles": available_profiles}
            )
        
        # Get base configuration from profile
        base_config = self._config_profiles[profile_name].copy()
        
        # Apply overrides if provided
        if overrides:
            base_config.update(overrides)
        
        # Extract provider from configuration
        if "provider" not in base_config:
            raise LLMConfigurationError(
                f"Profile '{profile_name}' missing provider specification",
                config_key="provider"
            )
        
        provider = base_config.pop("provider")
        
        # Generate cache key if not provided
        if cache_key is None:
            cache_key = f"profile_{profile_name}_{hash(str(sorted(base_config.items())))}"
        
        self.logger.info(f"Creating interface from profile '{profile_name}' for provider '{provider}'")
        
        return self.create_interface(
            provider=provider,
            config=base_config,
            cache_key=cache_key,
            use_cache=use_cache,
            **kwargs
        )
    
    def create_for_plugin(
        self,
        plugin_name: str,
        plugin_config: Dict[str, Any],
        fallback_provider: str = "openai",
        cache_key: Optional[str] = None,
        use_cache: bool = True,
        **kwargs: Any
    ) -> LLMInterface:
        """Create an LLM interface specifically configured for a plugin.
        
        This method creates LLM interfaces tailored for plugin usage, with
        plugin-specific configuration and fallback behavior.
        
        Args:
            plugin_name: Name of the plugin
            plugin_config: Plugin configuration containing LLM settings
            fallback_provider: Fallback provider if not specified in config
            cache_key: Optional cache key for instance caching
            use_cache: Whether to use cached instances
            **kwargs: Additional provider-specific arguments
            
        Returns:
            LLMInterface instance configured for the plugin
            
        Raises:
            LLMConfigurationError: If plugin configuration is invalid
            LLMError: If interface creation fails
        """
        # Extract LLM configuration from plugin config
        llm_config = self._extract_llm_config_from_plugin(plugin_name, plugin_config)
        
        # Determine provider
        provider = llm_config.get("provider", fallback_provider)
        
        # Generate cache key if not provided
        if cache_key is None:
            cache_key = f"plugin_{plugin_name}_{provider}_{hash(str(sorted(llm_config.items())))}"
        
        self.logger.info(f"Creating interface for plugin '{plugin_name}' using provider '{provider}'")
        
        return self.create_interface(
            provider=provider,
            config=llm_config,
            cache_key=cache_key,
            use_cache=use_cache,
            **kwargs
        )
    
    def validate_provider(self, provider: str) -> bool:
        """Validate if a provider is supported and available.
        
        Args:
            provider: Provider name to validate
            
        Returns:
            True if provider is valid and available
            
        Raises:
            LLMUnsupportedProviderError: If provider is not supported
        """
        provider = provider.lower().strip()
        
        if not is_provider_available(provider):
            available_providers = list_providers()
            raise LLMUnsupportedProviderError(
                f"Provider '{provider}' is not available",
                provider_name=provider,
                supported_providers=available_providers
            )
        
        return True
    
    def validate_config(self, config: Union[LLMConfig, Dict[str, Any]]) -> bool:
        """Validate LLM configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            LLMValidationError: If configuration is invalid
        """
        try:
            if isinstance(config, dict):
                # Validate by creating LLMConfig instance
                LLMConfig(**config)
            elif isinstance(config, LLMConfig):
                # Already validated
                pass
            else:
                raise LLMValidationError(
                    f"Invalid configuration type: {type(config).__name__}",
                    parameter_name="config",
                    parameter_value=config
                )
            
            return True
            
        except Exception as e:
            raise LLMValidationError(
                f"Configuration validation failed: {str(e)}",
                original_exception=e
            )
    
    def get_provider_info(self, provider: str) -> ProviderInfo:
        """Get information about a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            ProviderInfo instance with provider details
            
        Raises:
            LLMUnsupportedProviderError: If provider is not found
        """
        return get_provider_info(provider)
    
    def list_available_providers(self) -> List[str]:
        """List all available providers.
        
        Returns:
            List of available provider names
        """
        return [name for name in list_providers() if is_provider_available(name)]
    
    def discover_providers(self) -> Dict[str, Dict[str, Any]]:
        """Discover all providers and their availability.
        
        Returns:
            Dictionary mapping provider names to their information
        """
        return discover_providers()
    
    def register_profile(self, name: str, config: Dict[str, Any]) -> None:
        """Register a new configuration profile.
        
        Args:
            name: Profile name
            config: Profile configuration
            
        Raises:
            LLMConfigurationError: If profile configuration is invalid
        """
        if "provider" not in config:
            raise LLMConfigurationError(
                "Profile configuration must specify a provider",
                config_key="provider"
            )
        
        # Validate the configuration
        provider = config.get("provider")
        if provider:
            self.validate_provider(provider)
        
        config_copy = config.copy()
        config_copy.pop("provider", None)
        self.validate_config(config_copy)
        
        self._config_profiles[name] = config
        self.logger.info(f"Registered configuration profile '{name}'")
    
    def list_profiles(self) -> List[str]:
        """List all available configuration profiles.
        
        Returns:
            List of profile names
        """
        return list(self._config_profiles.keys())
    
    def get_profile(self, name: str) -> Dict[str, Any]:
        """Get a configuration profile.
        
        Args:
            name: Profile name
            
        Returns:
            Profile configuration
            
        Raises:
            LLMConfigurationError: If profile is not found
        """
        if name not in self._config_profiles:
            available_profiles = list(self._config_profiles.keys())
            raise LLMConfigurationError(
                f"Configuration profile '{name}' not found",
                config_key=name,
                details={"available_profiles": available_profiles}
            )
        
        return self._config_profiles[name].copy()
    
    def clear_cache(self) -> None:
        """Clear the provider instance cache."""
        self._provider_cache.clear()
        self.logger.info("Provider cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_instances": len(self._provider_cache),
            "cache_keys": list(self._provider_cache.keys()),
            "profiles": len(self._config_profiles),
            "profile_names": list(self._config_profiles.keys())
        }
    
    def _init_default_profiles(self) -> None:
        """Initialize default configuration profiles."""
        # Fast profile - optimized for speed
        self._config_profiles["fast"] = {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "max_tokens": 1000,
            "request_timeout": 30
        }
        
        # Quality profile - optimized for quality
        self._config_profiles["quality"] = {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.0,
            "max_tokens": 4000,
            "request_timeout": 120
        }
        
        # Local profile - for local models
        self._config_profiles["local"] = {
            "provider": "ollama",
            "model": "llama2",
            "temperature": 0.0,
            "max_tokens": 2000,
            "request_timeout": 300
        }
        
        # Creative profile - for creative tasks
        self._config_profiles["creative"] = {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "temperature": 0.7,
            "max_tokens": 4000,
            "request_timeout": 120
        }
        
        self.logger.debug(f"Initialized {len(self._config_profiles)} default profiles")
    
    def _generate_cache_key(self, provider: str, config: Union[LLMConfig, Dict[str, Any]]) -> str:
        """Generate a cache key for provider instances.
        
        Args:
            provider: Provider name
            config: Configuration
            
        Returns:
            Cache key string
        """
        if isinstance(config, LLMConfig):
            config_dict = config.model_dump()
        else:
            config_dict = config
        
        # Create a hash of the configuration
        config_str = str(sorted(config_dict.items()))
        config_hash = hash(config_str)
        
        return f"{provider}_{config_hash}"
    
    def _extract_llm_config_from_plugin(self, plugin_name: str, plugin_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract LLM configuration from plugin configuration.
        
        Args:
            plugin_name: Plugin name
            plugin_config: Plugin configuration
            
        Returns:
            Extracted LLM configuration
        """
        # Look for LLM configuration in common locations
        llm_config = {}
        
        # Check for direct LLM settings
        if "llm" in plugin_config:
            llm_config.update(plugin_config["llm"])
        
        # Check for common LLM parameters at top level
        llm_params = ["model", "temperature", "max_tokens", "top_p", "frequency_penalty", 
                     "presence_penalty", "request_timeout", "provider"]
        
        for param in llm_params:
            if param in plugin_config:
                llm_config[param] = plugin_config[param]
        
        # Plugin-specific defaults
        if plugin_name == "logitranslate":
            llm_config.setdefault("temperature", 0.0)
            llm_config.setdefault("max_tokens", 2000)
        elif plugin_name == "logiattack":
            llm_config.setdefault("temperature", 0.0)
            llm_config.setdefault("max_tokens", 2000)
        elif plugin_name == "judge":
            llm_config.setdefault("temperature", 0.0)
            llm_config.setdefault("max_tokens", 1000)
        
        return llm_config


# Global factory instance
_factory_instance: Optional[LLMFactory] = None


def get_factory() -> LLMFactory:
    """Get the global LLM factory instance.
    
    Returns:
        Global LLMFactory instance
    """
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = LLMFactory()
    return _factory_instance


def create_llm_interface(
    provider: str,
    config: Union[LLMConfig, Dict[str, Any]],
    **kwargs: Any
) -> LLMInterface:
    """Convenience function to create an LLM interface.
    
    Args:
        provider: Provider name
        config: LLM configuration
        **kwargs: Additional arguments
        
    Returns:
        LLMInterface instance
    """
    factory = get_factory()
    return factory.create_interface(provider, config, **kwargs)