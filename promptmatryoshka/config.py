"""Configuration management for PromptMatryoshka.

This module provides centralized configuration management for the PromptMatryoshka
project, including multi-provider LLM support, named profiles, and plugin settings.

The configuration system supports:
- Multi-provider LLM configurations (OpenAI, Anthropic, Ollama, HuggingFace)
- Named profiles/presets for different use cases
- Plugin-specific configuration overrides
- Environment variable resolution
- Comprehensive validation with helpful error messages

Classes:
    RateLimitConfig: Rate limiting configuration
    ProviderConfig: Base provider configuration
    ProfileConfig: Named profile configuration
    PluginConfig: Plugin-specific configuration
    PromptMatryoshkaConfig: Main configuration class
    ConfigurationError: Custom exception for configuration errors

Functions:
    get_config(): Get the singleton configuration instance
    load_config(): Load configuration from file
    resolve_env_var(): Resolve environment variable references
"""

import json
import os
import re
from typing import Dict, Any, Optional, List, Union, Type
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from .logging_utils import get_logger
from .llm_factory import LLMFactory, get_factory
from .llm_interface import LLMInterface
from .exceptions import LLMConfigurationError

logger = get_logger("Config")


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


def resolve_env_var(value: str, allow_missing: bool = False) -> Optional[str]:
    """Resolve environment variable references in configuration values.
    
    Args:
        value: Configuration value that may contain ${VAR_NAME} references
        allow_missing: If True, return None for missing variables instead of raising error
        
    Returns:
        Resolved value with environment variables expanded, or None if missing and allowed
        
    Raises:
        ConfigurationError: If required environment variable is not set and not allowed to be missing
    """
    if not isinstance(value, str):
        return value
    
    # Find all ${VAR_NAME} patterns
    pattern = r'\$\{([^}]+)\}'
    matches = re.findall(pattern, value)
    
    if not matches:
        return value
    
    resolved_value = value
    for var_name in matches:
        env_value = os.getenv(var_name)
        if env_value is None:
            if allow_missing:
                return None
            else:
                raise ConfigurationError(f"Environment variable '{var_name}' is not set")
        resolved_value = resolved_value.replace(f"${{{var_name}}}", env_value)
    
    return resolved_value


class RateLimitConfig(BaseModel):
    """Rate limiting configuration for providers."""
    
    requests_per_minute: Optional[int] = Field(None, ge=1, description="Requests per minute limit")
    tokens_per_minute: Optional[int] = Field(None, ge=1, description="Tokens per minute limit")
    requests_per_hour: Optional[int] = Field(None, ge=1, description="Requests per hour limit")
    tokens_per_hour: Optional[int] = Field(None, ge=1, description="Tokens per hour limit")
    
    model_config = ConfigDict(extra="allow")


class ProviderConfig(BaseModel):
    """Base configuration for LLM providers."""
    
    api_key: Optional[str] = Field(None, description="API key for authentication")
    base_url: Optional[str] = Field(None, description="Base URL for API endpoints")
    default_model: str = Field(..., description="Default model to use for this provider")
    organization: Optional[str] = Field(None, description="Organization ID")
    timeout: Optional[int] = Field(120, ge=1, description="Request timeout in seconds")
    max_retries: Optional[int] = Field(3, ge=0, description="Maximum retry attempts")
    retry_delay: Optional[float] = Field(1.0, ge=0.0, description="Delay between retries in seconds")
    rate_limit: Optional[RateLimitConfig] = Field(None, description="Rate limiting configuration")
    custom_headers: Optional[Dict[str, str]] = Field(None, description="Custom HTTP headers")
    provider_specific: Optional[Dict[str, Any]] = Field(None, description="Provider-specific settings")
    
    model_config = ConfigDict(extra="allow")
    
    @field_validator('api_key', mode='before')
    @classmethod
    def resolve_api_key(cls, v):
        """Resolve environment variables in API key."""
        if v is not None:
            resolved = resolve_env_var(v, allow_missing=True)
            return resolved if resolved is not None else v
        return v
    
    @field_validator('base_url', mode='before')
    @classmethod
    def resolve_base_url(cls, v):
        """Resolve environment variables in base URL."""
        if v is not None:
            resolved = resolve_env_var(v, allow_missing=True)
            return resolved if resolved is not None else v
        return v


class ProfileConfig(BaseModel):
    """Named profile configuration for different use cases."""
    
    provider: str = Field(..., description="Provider name for this profile")
    model: str = Field(..., description="Model to use")
    temperature: Optional[float] = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2000, gt=0, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    request_timeout: Optional[int] = Field(120, gt=0, description="Request timeout in seconds")
    description: Optional[str] = Field(None, description="Profile description")
    
    model_config = ConfigDict(extra="allow")
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature parameter."""
        if v is not None and not (0.0 <= v <= 2.0):
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
    
    @field_validator('top_p')
    @classmethod
    def validate_top_p(cls, v):
        """Validate top_p parameter."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError('top_p must be between 0.0 and 1.0')
        return v


class PluginConfig(BaseModel):
    """Plugin-specific configuration."""
    
    profile: Optional[str] = Field(None, description="Profile to use for this plugin")
    provider: Optional[str] = Field(None, description="Provider override for this plugin")
    model: Optional[str] = Field(None, description="Model override for this plugin")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature override")
    max_tokens: Optional[int] = Field(None, gt=0, description="Max tokens override")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p override")
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Frequency penalty override")
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Presence penalty override")
    request_timeout: Optional[int] = Field(None, gt=0, description="Request timeout override")
    technique_params: Optional[Dict[str, Any]] = Field(None, description="Technique-specific parameters")
    
    model_config = ConfigDict(extra="allow")


class PromptMatryoshkaConfig(BaseModel):
    """Main configuration class for PromptMatryoshka."""
    
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict, description="Provider configurations")
    profiles: Dict[str, ProfileConfig] = Field(default_factory=dict, description="Named profiles")
    plugins: Dict[str, PluginConfig] = Field(default_factory=dict, description="Plugin configurations")
    
    # Legacy support for backward compatibility
    models: Optional[Dict[str, str]] = Field(None, description="Legacy model mappings")
    llm_settings: Optional[Dict[str, Any]] = Field(None, description="Legacy LLM settings")
    plugin_settings: Optional[Dict[str, Any]] = Field(None, description="Legacy plugin settings")
    
    # System settings
    logging: Optional[Dict[str, Any]] = Field(None, description="Logging configuration")
    storage: Optional[Dict[str, Any]] = Field(None, description="Storage configuration")
    
    model_config = ConfigDict(extra="allow")
    
    @model_validator(mode='after')
    def validate_profiles_reference_valid_providers(self):
        """Ensure all profiles reference valid providers."""
        for profile_name, profile_config in self.profiles.items():
            if profile_config.provider not in self.providers:
                raise ValueError(f"Profile '{profile_name}' references unknown provider '{profile_config.provider}'")
        return self
    
    @model_validator(mode='after')
    def validate_plugins_reference_valid_profiles_and_providers(self):
        """Ensure plugin configurations reference valid profiles and providers."""
        for plugin_name, plugin_config in self.plugins.items():
            if plugin_config.profile and plugin_config.profile not in self.profiles:
                raise ValueError(f"Plugin '{plugin_name}' references unknown profile '{plugin_config.profile}'")
            if plugin_config.provider and plugin_config.provider not in self.providers:
                raise ValueError(f"Plugin '{plugin_name}' references unknown provider '{plugin_config.provider}'")
        return self
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration structure."""
        return {
            "providers": {
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-4o-mini",
                    "rate_limit": {
                        "requests_per_minute": 500,
                        "tokens_per_minute": 150000
                    }
                },
                "anthropic": {
                    "api_key": "${ANTHROPIC_API_KEY}",
                    "default_model": "claude-3-5-sonnet-20241022"
                },
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "default_model": "llama3.2:3b"
                },
                "huggingface": {
                    "api_key": "${HUGGINGFACE_API_KEY}",
                    "default_model": "microsoft/DialoGPT-medium"
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
                }
            },
            "plugins": {
                "logitranslate": {
                    "profile": "research-openai",
                    "technique_params": {
                        "validation_enabled": True,
                        "max_attempts": 3,
                        "retry_delay": 1.0
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
            },
            "logging": {
                "level": "INFO",
                "save_artifacts": True,
                "debug_mode": False
            },
            "storage": {
                "save_runs": True,
                "output_directory": "runs",
                "max_saved_runs": 100
            }
        }


class Config:
    """Configuration management class for PromptMatryoshka.
    
    This class provides centralized access to configuration values with support
    for multi-provider LLM configurations, named profiles, and plugin settings.
    It maintains backward compatibility with the existing plugin integration methods.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self._config: Optional[PromptMatryoshkaConfig] = None
        self._raw_config: Dict[str, Any] = {}
        self._factory: Optional[LLMFactory] = None
        self.load()
    
    def load(self) -> None:
        """Load configuration from file with fallback to defaults.
        
        Raises:
            ConfigurationError: If configuration file exists but is invalid
        """
        try:
            if os.path.exists(self.config_path):
                logger.info(f"Loading configuration from {self.config_path}")
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._raw_config = json.load(f)
                
                # Handle legacy configuration format
                if self._is_legacy_config(self._raw_config):
                    logger.info("Detected legacy configuration format, converting...")
                    self._raw_config = self._convert_legacy_config(self._raw_config)
                
                # Validate and create configuration
                self._config = PromptMatryoshkaConfig(**self._raw_config)
                logger.info("Configuration loaded successfully")
            else:
                logger.warning(f"Configuration file {self.config_path} not found. Using defaults.")
                default_config = PromptMatryoshkaConfig().get_default_config()
                self._config = PromptMatryoshkaConfig(**default_config)
                self._raw_config = default_config
                
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file {self.config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def _is_legacy_config(self, config: Dict[str, Any]) -> bool:
        """Check if configuration is in legacy format."""
        return "providers" not in config and "models" in config
    
    def _convert_legacy_config(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert legacy configuration format to new format."""
        new_config = {
            "providers": {
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-4o-mini"
                }
            },
            "profiles": {},
            "plugins": {}
        }
        
        # Copy system settings
        for key in ["logging", "storage"]:
            if key in legacy_config:
                new_config[key] = legacy_config[key]
        
        # Convert plugin settings to new format
        if "plugin_settings" in legacy_config:
            for plugin_name, plugin_config in legacy_config["plugin_settings"].items():
                new_config["plugins"][plugin_name] = {
                    "provider": "openai",
                    "model": plugin_config.get("model", "gpt-4o-mini"),
                    "temperature": plugin_config.get("temperature", 0.0),
                    "max_tokens": plugin_config.get("max_tokens", 2000),
                    "technique_params": {
                        "validation_enabled": plugin_config.get("validation_enabled", True)
                    }
                }
        
        # Keep legacy sections for backward compatibility
        new_config["models"] = legacy_config.get("models", {})
        new_config["llm_settings"] = legacy_config.get("llm_settings", {})
        new_config["plugin_settings"] = legacy_config.get("plugin_settings", {})
        
        return new_config
    
    def get_llm_factory(self) -> LLMFactory:
        """Get the LLM factory instance."""
        if self._factory is None:
            self._factory = get_factory()
            
            # Register profiles with the factory
            for profile_name, profile_config in self._config.profiles.items():
                profile_dict = profile_config.model_dump()
                self._factory.register_profile(profile_name, profile_dict)
        
        return self._factory
    
    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider configuration or None if not found
        """
        return self._config.providers.get(provider_name)
    
    def get_profile_config(self, profile_name: str) -> Optional[ProfileConfig]:
        """Get configuration for a specific profile.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Profile configuration or None if not found
        """
        return self._config.profiles.get(profile_name)
    
    def get_plugin_config(self, plugin_name: str) -> Optional[PluginConfig]:
        """Get configuration for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin configuration or None if not found
        """
        return self._config.plugins.get(plugin_name)
    
    def create_llm_for_plugin(self, plugin_name: str) -> LLMInterface:
        """Create an LLM interface for a specific plugin using the new factory.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            LLM interface instance
            
        Raises:
            ConfigurationError: If plugin configuration is invalid
        """
        try:
            factory = self.get_llm_factory()
            plugin_config = self.get_plugin_config(plugin_name)
            
            if plugin_config is None:
                # Fallback to legacy method
                return self._create_llm_legacy(plugin_name)
            
            # Use profile if specified
            if plugin_config.profile:
                overrides = {}
                
                # Add any plugin-specific overrides
                for field in ["model", "temperature", "max_tokens", "top_p", 
                             "frequency_penalty", "presence_penalty", "request_timeout"]:
                    value = getattr(plugin_config, field, None)
                    if value is not None:
                        overrides[field] = value
                
                return factory.create_from_profile(
                    plugin_config.profile,
                    overrides=overrides if overrides else None
                )
            
            # Use provider directly
            provider = plugin_config.provider or "openai"
            llm_config = {}
            
            # Build LLM configuration from plugin settings
            for field in ["model", "temperature", "max_tokens", "top_p", 
                         "frequency_penalty", "presence_penalty", "request_timeout"]:
                value = getattr(plugin_config, field, None)
                if value is not None:
                    llm_config[field] = value
            
            # Use provider's default model if not specified
            if "model" not in llm_config:
                provider_config = self.get_provider_config(provider)
                if provider_config:
                    llm_config["model"] = provider_config.default_model
                else:
                    llm_config["model"] = "gpt-4o-mini"
            
            return factory.create_interface(provider, llm_config)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create LLM for plugin '{plugin_name}': {e}")
    
    def _create_llm_legacy(self, plugin_name: str) -> LLMInterface:
        """Create LLM using legacy configuration method."""
        from langchain_openai import ChatOpenAI
        
        # Get settings using legacy methods
        model = self.get_model_for_plugin(plugin_name)
        llm_settings = self.get_llm_settings_for_plugin(plugin_name)
        
        # Create ChatOpenAI instance (legacy behavior)
        openai_kwargs = {
            "model": model,
            "temperature": llm_settings.get("temperature", 0.0),
            "max_tokens": llm_settings.get("max_tokens", 2000),
            "top_p": llm_settings.get("top_p", 1.0),
            "frequency_penalty": llm_settings.get("frequency_penalty", 0.0),
            "presence_penalty": llm_settings.get("presence_penalty", 0.0),
            "request_timeout": llm_settings.get("request_timeout", 120)
        }
        
        return ChatOpenAI(**openai_kwargs)
    
    # Backward compatibility methods
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key (legacy method).
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._raw_config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_for_plugin(self, plugin_name: str) -> str:
        """Get the model name for a specific plugin (legacy method).
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Model name for the plugin
        """
        # First check new plugin configuration
        plugin_config = self.get_plugin_config(plugin_name)
        if plugin_config:
            if plugin_config.model:
                return plugin_config.model
            elif plugin_config.profile:
                profile_config = self.get_profile_config(plugin_config.profile)
                if profile_config:
                    return profile_config.model
            elif plugin_config.provider:
                provider_config = self.get_provider_config(plugin_config.provider)
                if provider_config:
                    return provider_config.default_model
        
        # Fall back to legacy configuration
        if self._config.plugin_settings:
            plugin_settings = self._config.plugin_settings.get(plugin_name, {})
            if "model" in plugin_settings:
                return plugin_settings["model"]
        
        # Check legacy models section
        if self._config.models:
            model_key = f"{plugin_name}_model"
            if model_key in self._config.models:
                return self._config.models[model_key]
        
        # Final fallback
        return "gpt-4o-mini"
    
    def get_llm_settings_for_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """Get LLM settings for a specific plugin (legacy method).
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            LLM settings for the plugin
        """
        settings = {}
        
        # Start with global LLM settings if available
        if self._config.llm_settings:
            settings.update(self._config.llm_settings)
        
        # Check new plugin configuration
        plugin_config = self.get_plugin_config(plugin_name)
        if plugin_config:
            # Override with plugin-specific settings
            for field in ["temperature", "max_tokens", "top_p", 
                         "frequency_penalty", "presence_penalty", "request_timeout"]:
                value = getattr(plugin_config, field, None)
                if value is not None:
                    settings[field] = value
            
            # If using a profile, get profile settings
            if plugin_config.profile:
                profile_config = self.get_profile_config(plugin_config.profile)
                if profile_config:
                    for field in ["temperature", "max_tokens", "top_p", 
                                 "frequency_penalty", "presence_penalty", "request_timeout"]:
                        value = getattr(profile_config, field, None)
                        if value is not None and field not in settings:
                            settings[field] = value
        
        # Fall back to legacy plugin settings
        if self._config.plugin_settings:
            plugin_settings = self._config.plugin_settings.get(plugin_name, {})
            for key in ["temperature", "max_tokens", "top_p", 
                       "frequency_penalty", "presence_penalty", "request_timeout"]:
                if key in plugin_settings and key not in settings:
                    settings[key] = plugin_settings[key]
        
        # Set defaults if not specified
        settings.setdefault("temperature", 0.0)
        settings.setdefault("max_tokens", 2000)
        settings.setdefault("top_p", 1.0)
        settings.setdefault("frequency_penalty", 0.0)
        settings.setdefault("presence_penalty", 0.0)
        settings.setdefault("request_timeout", 120)
        
        return settings
    
    def reload(self) -> None:
        """Reload the configuration from file."""
        self._factory = None  # Clear factory cache
        self.load()
    
    def to_dict(self) -> Dict[str, Any]:
        """Get the entire configuration as a dictionary."""
        return self._raw_config.copy()
    
    def validate_configuration(self) -> bool:
        """Validate the current configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Check that all referenced providers exist
            for profile_name, profile_config in self._config.profiles.items():
                if profile_config.provider not in self._config.providers:
                    raise ConfigurationError(
                        f"Profile '{profile_name}' references unknown provider '{profile_config.provider}'"
                    )
            
            # Check that all plugin configurations are valid
            for plugin_name, plugin_config in self._config.plugins.items():
                if plugin_config.profile and plugin_config.profile not in self._config.profiles:
                    raise ConfigurationError(
                        f"Plugin '{plugin_name}' references unknown profile '{plugin_config.profile}'"
                    )
                if plugin_config.provider and plugin_config.provider not in self._config.providers:
                    raise ConfigurationError(
                        f"Plugin '{plugin_name}' references unknown provider '{plugin_config.provider}'"
                    )
            
            return True
            
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self._config.providers.keys())
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available profiles."""
        return list(self._config.profiles.keys())
    
    def get_available_plugins(self) -> List[str]:
        """Get list of configured plugins."""
        return list(self._config.plugins.keys())


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: str = "config.json") -> Config:
    """Get the singleton configuration instance.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def load_config(config_path: str = "config.json") -> Config:
    """Load and return a new configuration instance.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        New configuration instance
    """
    return Config(config_path)


def reset_config() -> None:
    """Reset the global configuration instance (used for testing)."""
    global _config_instance
    _config_instance = None