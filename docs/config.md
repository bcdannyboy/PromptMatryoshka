# Configuration Module Documentation

## Purpose & Overview

The [`config.py`](../promptmatryoshka/config.py) module provides centralized configuration management for the PromptMatryoshka framework. It handles model selection, plugin-specific settings, runtime parameters, and provides a robust configuration system with validation, defaults, and error handling. This module ensures consistent configuration across all components of the framework.

## Architecture

The configuration system follows a hierarchical structure with defaults, file-based overrides, and plugin-specific settings:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration System                        │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Default       │  │   File-based    │  │   Plugin        │ │
│  │   Configuration │  │   Configuration │  │   Configuration │ │
│  │                 │  │                 │  │                 │ │
│  │   • Models      │  │   • Overrides   │  │   • Per-plugin  │ │
│  │   • LLM Settings│  │   • Validation  │  │   • Settings    │ │
│  │   • Logging     │  │   • Merging     │  │   • Inheritance │ │
│  │   • Storage     │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                 │                               │
│                                 ▼                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Merged Configuration                       │   │
│  │              with Plugin Resolution                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Classes/Functions

### Config Class

The main configuration management class that provides centralized access to all configuration values.

```python
class Config:
    def __init__(self, config_path: str = "config.json")
```

**Key Methods:**

- [`load()`](../promptmatryoshka/config.py:99): Loads configuration from file with fallback to defaults
- [`get(key, default=None)`](../promptmatryoshka/config.py:189): Gets configuration value by key with dot notation support
- [`get_plugin_config(plugin_name)`](../promptmatryoshka/config.py:210): Gets configuration for a specific plugin
- [`get_model_for_plugin(plugin_name)`](../promptmatryoshka/config.py:222): Gets model name for a specific plugin
- [`get_llm_settings_for_plugin(plugin_name)`](../promptmatryoshka/config.py:241): Gets LLM settings for a specific plugin
- [`reload()`](../promptmatryoshka/config.py:262): Reloads configuration from file
- [`to_dict()`](../promptmatryoshka/config.py:271): Returns complete configuration as dictionary

### Exception Classes

- [`ConfigurationError`](../promptmatryoshka/config.py:22): Raised when configuration loading or validation fails

### Module Functions

- [`get_config(config_path="config.json")`](../promptmatryoshka/config.py:283): Gets the singleton configuration instance
- [`load_config(config_path="config.json")`](../promptmatryoshka/config.py:298): Loads and returns a new configuration instance

## Usage Examples

### Basic Configuration Usage

```python
from promptmatryoshka.config import get_config

# Get the global configuration instance
config = get_config()

# Get a configuration value
model_name = config.get("models.logitranslate_model")
print(f"LogiTranslate model: {model_name}")

# Get with default value
temperature = config.get("llm_settings.temperature", 0.7)
```

### Plugin-Specific Configuration

```python
# Get configuration for a specific plugin
plugin_config = config.get_plugin_config("logitranslate")
print(f"LogiTranslate config: {plugin_config}")

# Get model for a specific plugin
model = config.get_model_for_plugin("logiattack")
print(f"LogiAttack model: {model}")

# Get LLM settings for a plugin
llm_settings = config.get_llm_settings_for_plugin("judge")
print(f"Judge LLM settings: {llm_settings}")
```

### Configuration Management

```python
# Reload configuration from file
config.reload()

# Get complete configuration as dictionary
full_config = config.to_dict()

# Load a custom configuration file
from promptmatryoshka.config import load_config
custom_config = load_config("custom_config.json")
```

## Integration Points

### Plugin System Integration

The configuration system integrates deeply with the plugin system:

```python
# Plugin-specific model selection
model = config.get_model_for_plugin("logitranslate")

# Plugin-specific LLM settings
settings = config.get_llm_settings_for_plugin("logiattack")
```

### Logging Integration

Configuration includes logging settings:

```python
{
    "logging": {
        "level": "INFO",
        "save_artifacts": True,
        "debug_mode": False
    }
}
```

### Storage Integration

Configuration manages storage settings:

```python
{
    "storage": {
        "save_runs": True,
        "output_directory": "runs",
        "max_saved_runs": 100
    }
}
```

## Configuration Structure

### Default Configuration

The system provides comprehensive defaults:

```json
{
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
            "validation_enabled": true
        },
        "logiattack": {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 2000,
            "validation_enabled": true
        },
        "judge": {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 1000,
            "validation_enabled": true
        }
    },
    "logging": {
        "level": "INFO",
        "save_artifacts": true,
        "debug_mode": false
    },
    "storage": {
        "save_runs": true,
        "output_directory": "runs",
        "max_saved_runs": 100
    }
}
```

### Plugin-Specific Configuration

Plugin settings support inheritance and override patterns:

```json
{
    "plugin_settings": {
        "logitranslate": {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 2000,
            "validation_enabled": true,
            "custom_setting": "value"
        }
    }
}
```

### Dot Notation Access

Configuration values can be accessed using dot notation:

```python
# These are equivalent
temperature = config.get("llm_settings.temperature")
temperature = config.get("llm_settings")["temperature"]

# Plugin-specific access
plugin_model = config.get("plugin_settings.logitranslate.model")
```

## Error Handling

### Configuration Validation

The system validates configuration structure and values:

```python
def _validate_config(self, config: Dict[str, Any]) -> None:
    """Validate the configuration structure and values."""
    # Check required sections
    required_sections = ["models", "llm_settings", "plugin_settings"]
    for section in required_sections:
        if section not in config:
            logger.warning(f"Missing configuration section '{section}', will use defaults")
    
    # Validate model names
    if "models" in config:
        for model_key, model_value in config["models"].items():
            if not isinstance(model_value, str) or not model_value.strip():
                raise ConfigurationError(f"Model '{model_key}' must be a non-empty string")
```

### Error Recovery

The system provides graceful error recovery:

```python
try:
    if os.path.exists(self.config_path):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        self._validate_config(file_config)
        self._config = self._merge_configs(self._defaults, file_config)
    else:
        logger.warning(f"Configuration file {self.config_path} not found. Using defaults.")
        self._config = self._defaults.copy()
except json.JSONDecodeError as e:
    raise ConfigurationError(f"Invalid JSON in configuration file {self.config_path}: {e}")
```

### Configuration Merging

The system recursively merges configurations:

```python
def _merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries."""
    result = default.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = self._merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result
```

## Developer Notes

### Configuration File Format

Configuration files use JSON format with the following structure:

```json
{
    "models": {
        "plugin_name_model": "model_name"
    },
    "llm_settings": {
        "temperature": 0.0,
        "max_tokens": 2000
    },
    "plugin_settings": {
        "plugin_name": {
            "setting_name": "value"
        }
    }
}
```

### Plugin Configuration Resolution

The system uses a hierarchy for plugin configuration:

1. **Plugin-specific settings**: `plugin_settings.{plugin_name}.{setting}`
2. **Global LLM settings**: `llm_settings.{setting}`
3. **Model-specific settings**: `models.{plugin_name}_model`
4. **Default values**: Hard-coded defaults

### Singleton Pattern

The configuration system uses a singleton pattern for global access:

```python
# Global configuration instance
_config_instance: Optional[Config] = None

def get_config(config_path: str = "config.json") -> Config:
    """Get the singleton configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
```

### Configuration Validation Rules

The system validates:

- **Model Names**: Must be non-empty strings
- **Temperature**: Must be between 0 and 2
- **Max Tokens**: Must be positive integers
- **Plugin Settings**: Must be dictionaries

### Performance Considerations

- **Lazy Loading**: Configuration is loaded only when needed
- **Caching**: Configuration values are cached after loading
- **Validation**: Validation is performed once during loading

### Extension Points

The configuration system can be extended:

1. **Custom Validators**: Add validation for new configuration sections
2. **Environment Variables**: Support for environment variable overrides
3. **Multiple Config Files**: Support for multiple configuration sources
4. **Dynamic Reloading**: Hot-reload configuration changes

### Testing Integration

The configuration system supports testing:

```python
# Create test configuration
test_config = Config("test_config.json")

# Use custom configuration instance
custom_config = load_config("custom_config.json")
```

## Implementation Details

### Configuration Loading Sequence

1. **Initialize Defaults**: Load hard-coded default values
2. **Check File Existence**: Verify configuration file exists
3. **Parse JSON**: Parse configuration file as JSON
4. **Validate Structure**: Validate configuration structure and values
5. **Merge Configurations**: Recursively merge file config with defaults
6. **Cache Results**: Cache merged configuration for future access

### Model Resolution Algorithm

For plugin model resolution:

1. **Check Plugin Settings**: Look for `plugin_settings.{plugin}.model`
2. **Check Model Settings**: Look for `models.{plugin}_model`
3. **Use Default**: Fall back to `"gpt-4o-mini"`

### LLM Settings Resolution

For plugin LLM settings:

1. **Start with Global**: Copy global LLM settings
2. **Apply Plugin Overrides**: Override with plugin-specific settings
3. **Return Merged**: Return merged settings dictionary

### Error Handling Strategy

The configuration system handles errors gracefully:

- **Missing Files**: Fall back to defaults with warnings
- **Invalid JSON**: Raise ConfigurationError with details
- **Validation Errors**: Raise ConfigurationError with specific issues
- **Partial Configs**: Merge partial configurations with defaults