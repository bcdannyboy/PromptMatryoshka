"""Configuration management for PromptMatryoshka.

This module provides centralized configuration management for the PromptMatryoshka
project, including model selection, plugin settings, and runtime parameters.

Classes:
    Config: Main configuration class with methods for loading and accessing config values.
    ConfigurationError: Custom exception for configuration-related errors.

Functions:
    get_config(): Get the singleton configuration instance.
    load_config(): Load configuration from file with fallback defaults.
"""

import json
import os
from typing import Dict, Any, Optional
from promptmatryoshka.logging_utils import get_logger

logger = get_logger("Config")

class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass

class Config:
    """
    Configuration management class for PromptMatryoshka.
    
    Provides centralized access to configuration values with fallback defaults,
    validation, and error handling.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the configuration.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        self.config_path = config_path
        self._config = {}
        self._defaults = self._get_default_config()
        self.load()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration values.
        
        Returns:
            Dict[str, Any]: Default configuration dictionary.
        """
        return {
            "models": {
                "logitranslate_model": "gpt-4o-mini",
                "logiattack_model": "gpt-4o-mini",
                "judge_model": "gpt-4o-mini"
            },
            "llm_settings": {
                "temperature": 0.0,
                "max_tokens": 2000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "request_timeout": 120
            },
            "plugin_settings": {
                "logitranslate": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                    "max_tokens": 2000,
                    "validation_enabled": True
                },
                "logiattack": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                    "max_tokens": 2000,
                    "validation_enabled": True
                },
                "judge": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                    "max_tokens": 1000,
                    "validation_enabled": True
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
    
    def load(self) -> None:
        """
        Load configuration from file with fallback to defaults.
        
        Raises:
            ConfigurationError: If configuration file exists but is invalid.
        """
        try:
            if os.path.exists(self.config_path):
                logger.info(f"Loading configuration from {self.config_path}")
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                
                # Validate the loaded configuration
                self._validate_config(file_config)
                
                # Merge with defaults (file config takes precedence)
                self._config = self._merge_configs(self._defaults, file_config)
                logger.info("Configuration loaded successfully")
            else:
                logger.warning(f"Configuration file {self.config_path} not found. Using defaults.")
                self._config = self._defaults.copy()
                
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file {self.config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration structure and values.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary to validate.
            
        Raises:
            ConfigurationError: If configuration is invalid.
        """
        # Check required sections
        required_sections = ["models", "llm_settings", "plugin_settings"]
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing configuration section '{section}', will use defaults")
        
        # Validate model names if present
        if "models" in config:
            for model_key, model_value in config["models"].items():
                if not isinstance(model_value, str) or not model_value.strip():
                    raise ConfigurationError(f"Model '{model_key}' must be a non-empty string")
        
        # Validate LLM settings if present
        if "llm_settings" in config:
            llm_settings = config["llm_settings"]
            if "temperature" in llm_settings:
                temp = llm_settings["temperature"]
                if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                    raise ConfigurationError("Temperature must be a number between 0 and 2")
            
            if "max_tokens" in llm_settings:
                tokens = llm_settings["max_tokens"]
                if not isinstance(tokens, int) or tokens <= 0:
                    raise ConfigurationError("max_tokens must be a positive integer")
        
        # Validate plugin settings if present
        if "plugin_settings" in config:
            for plugin_name, plugin_config in config["plugin_settings"].items():
                if not isinstance(plugin_config, dict):
                    raise ConfigurationError(f"Plugin settings for '{plugin_name}' must be a dictionary")
    
    def _merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            default (Dict[str, Any]): Default configuration.
            override (Dict[str, Any]): Override configuration.
            
        Returns:
            Dict[str, Any]: Merged configuration.
        """
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key (str): Configuration key (supports dot notation, e.g., 'models.logitranslate_model').
            default (Any): Default value if key not found.
            
        Returns:
            Any: Configuration value or default.
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific plugin.
        
        Args:
            plugin_name (str): Name of the plugin.
            
        Returns:
            Dict[str, Any]: Plugin configuration dictionary.
        """
        return self.get(f"plugin_settings.{plugin_name}", {})
    
    def get_model_for_plugin(self, plugin_name: str) -> str:
        """
        Get the model name for a specific plugin.
        
        Args:
            plugin_name (str): Name of the plugin.
            
        Returns:
            str: Model name for the plugin.
        """
        # First check plugin-specific settings
        plugin_config = self.get_plugin_config(plugin_name)
        if "model" in plugin_config:
            return plugin_config["model"]
        
        # Then check general model settings
        model_key = f"{plugin_name}_model"
        return self.get(f"models.{model_key}", "gpt-4o-mini")
    
    def get_llm_settings_for_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get LLM settings for a specific plugin.
        
        Args:
            plugin_name (str): Name of the plugin.
            
        Returns:
            Dict[str, Any]: LLM settings for the plugin.
        """
        # Start with global LLM settings
        settings = self.get("llm_settings", {}).copy()
        
        # Override with plugin-specific settings
        plugin_config = self.get_plugin_config(plugin_name)
        for key in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "request_timeout"]:
            if key in plugin_config:
                settings[key] = plugin_config[key]
        
        return settings
    
    def reload(self) -> None:
        """
        Reload the configuration from file.
        
        Raises:
            ConfigurationError: If configuration file is invalid.
        """
        self.load()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the entire configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: Complete configuration dictionary.
        """
        return self._config.copy()

# Global configuration instance
_config_instance: Optional[Config] = None

def get_config(config_path: str = "config.json") -> Config:
    """
    Get the singleton configuration instance.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        Config: Configuration instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance

def load_config(config_path: str = "config.json") -> Config:
    """
    Load and return a new configuration instance.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        Config: New configuration instance.
    """
    return Config(config_path)