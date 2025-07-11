# Configuration Module Documentation

## Purpose & Overview

The [`config.py`](../promptmatryoshka/config.py) module provides centralized configuration management for the PromptMatryoshka framework. It handles **multi-provider LLM support**, configuration profiles, plugin-specific settings, and runtime parameters through a sophisticated configuration system with validation, defaults, and error handling. This module enables seamless switching between different LLM providers (OpenAI, Anthropic, Ollama, HuggingFace) and supports predefined configuration profiles for common use cases.

## 🚀 Default Configuration (Ready to Use)

**The repository includes a pre-configured [`config.json`](../config.json) file that works out of the box** for the demo workflow. You don't need to create or modify this file to get started.

### Quick Start Configuration

The default configuration includes:
- **Demo profiles** optimized for the AdvBench demo
- **Plugin settings** for the complete pipeline (FlipAttack → LogiTranslate → BOOST → LogiAttack → Judge)
- **Default OpenAI provider** settings (just needs your API key in `.env`)
- **Pre-configured pipeline** ready for testing

**To use the demo:**
1. Set your `OPENAI_API_KEY` in `.env` file
2. Run: `python3 promptmatryoshka/cli.py advbench --count 10 --judge --max-retries 5`
3. The framework automatically uses the default [`config.json`](../config.json)

### Default Configuration Structure

The included [`config.json`](../config.json) contains:
```json
{
  "providers": {
    "openai": {
      "api_key": "${OPENAI_API_KEY}",
      "base_url": "https://api.openai.com/v1",
      "default_model": "gpt-4o"
    }
  },
  "profiles": {
    "demo-gpt4o": {
      "provider": "openai",
      "model": "gpt-4o",
      "temperature": 0.0,
      "max_tokens": 4000,
      "description": "GPT-4o for LogiTranslate in demo"
    },
    "demo-gpt35": {
      "provider": "openai",
      "model": "gpt-3.5-turbo",
      "temperature": 0.0,
      "max_tokens": 4000,
      "description": "GPT-3.5-turbo for LogiAttack in demo"
    }
  },
  "plugins": {
    "logitranslate": {
      "enabled": true,
      "profile": "demo-gpt4o"
    },
    "logiattack": {
      "enabled": true,
      "profile": "demo-gpt35"
    },
    "judge": {
      "enabled": true,
      "profile": "demo-gpt4o"
    }
  }
}
```

**No additional configuration is required** - the demo works immediately with this setup.

## Architecture

The configuration system follows a multi-layered architecture with provider abstraction, profile management, and environment variable support:

```
┌─────────────────────────────────────────────────────────────────┐
│                Multi-Provider Configuration System              │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Environment   │  │   Configuration │  │   Provider      │ │
│  │   Variables     │  │   Profiles      │  │   Registry      │ │
│  │                 │  │                 │  │                 │ │
│  │   • API Keys    │  │   • Research    │  │   • OpenAI      │ │
│  │   • Endpoints   │  │   • Production  │  │   • Anthropic   │ │
│  │   • Settings    │  │   • Local Dev   │  │   • Ollama      │ │
│  │   • Validation  │  │   • Testing     │  │   • HuggingFace │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                 │                               │
│                                 ▼                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Merged Configuration                       │   │
│  │              with Provider Resolution                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Classes/Functions

### Config Class

The main configuration management class that provides centralized access to all configuration values with **multi-provider support**.

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
- [`get_provider_settings(provider_name)`](../promptmatryoshka/config.py): Gets settings for a specific LLM provider
- [`get_active_profile()`](../promptmatryoshka/config.py): Gets the currently active configuration profile
- [`set_profile(profile_name)`](../promptmatryoshka/config.py): Switches to a different configuration profile
- [`reload()`](../promptmatryoshka/config.py:262): Reloads configuration from file
- [`to_dict()`](../promptmatryoshka/config.py:271): Returns complete configuration as dictionary

### LLMFactory Class

Factory class for creating provider-specific LLM instances with automatic configuration.

```python
class LLMFactory:
    @staticmethod
    def create_llm(provider: str, model: str, **kwargs) -> LLMInterface
```

**Key Methods:**

- [`create_llm(provider, model, **kwargs)`](../promptmatryoshka/llm_factory.py): Creates an LLM instance for the specified provider
- [`get_available_providers()`](../promptmatryoshka/llm_factory.py): Returns list of available LLM providers
- [`validate_provider_config(provider, config)`](../promptmatryoshka/llm_factory.py): Validates provider-specific configuration

### Exception Classes

- [`ConfigurationError`](../promptmatryoshka/config.py:22): Raised when configuration loading or validation fails

### Module Functions

- [`get_config(config_path="config.json")`](../promptmatryoshka/config.py:283): Gets the singleton configuration instance
- [`load_config(config_path="config.json")`](../promptmatryoshka/config.py:298): Loads and returns a new configuration instance

## Usage Examples

### Demo Configuration (Default)

The default configuration works immediately with the demo:

```python
from promptmatryoshka.config import get_config

# Get the global configuration instance (uses default config.json)
config = get_config()

# The demo profiles are already configured
print(f"Demo GPT-4o profile: {config.get_profile_settings('demo-gpt4o')}")
print(f"Demo GPT-3.5 profile: {config.get_profile_settings('demo-gpt35')}")

# Plugin configurations are pre-set for the demo
logitranslate_config = config.get_plugin_config("logitranslate")
print(f"LogiTranslate uses: {logitranslate_config}")
```

### Basic Multi-Provider Configuration

```python
from promptmatryoshka.config import get_config
from promptmatryoshka.llm_factory import LLMFactory

# Get the global configuration instance
config = get_config()

# Get current active profile
profile = config.get_active_profile()
print(f"Active profile: {profile}")

# Switch to a different profile
config.set_profile("production-anthropic")

# Get provider-specific settings
openai_settings = config.get_provider_settings("openai")
anthropic_settings = config.get_provider_settings("anthropic")
```

### Multi-Provider Plugin Configuration

```python
# Get configuration for a specific plugin with provider support
plugin_config = config.get_plugin_config("logitranslate")
print(f"LogiTranslate config: {plugin_config}")

# Get model for a specific plugin (now supports multiple providers)
model = config.get_model_for_plugin("logiattack")
provider = config.get_provider_for_plugin("logiattack")
print(f"LogiAttack model: {model} (Provider: {provider})")

# Create LLM instance using factory
llm = LLMFactory.create_llm(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0
)
```

### Configuration Profiles

```python
# List available profiles
profiles = config.get_available_profiles()
print(f"Available profiles: {profiles}")

# Switch between profiles
config.set_profile("demo-gpt4o")        # For demo with GPT-4o
config.set_profile("demo-gpt35")        # For demo with GPT-3.5
config.set_profile("research-openai")   # For research with OpenAI
config.set_profile("production-anthropic")  # For production with Anthropic

# Get profile-specific settings
demo_settings = config.get_profile_settings("demo-gpt4o")
```

### Environment Variable Integration

```python
# Environment variables are automatically resolved
# ${OPENAI_API_KEY} in config becomes the actual API key
api_key = config.get("providers.openai.api_key")  # Resolves ${OPENAI_API_KEY}

# Custom environment variable resolution
custom_model = config.get("models.custom_model")  # Resolves ${CUSTOM_MODEL}
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

### Multi-Provider Configuration

The system now supports sophisticated multi-provider configurations with profiles and environment variables:

```json
{
    "active_profile": "research-openai",
    "profiles": {
        "research-openai": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 2000,
            "description": "Research configuration using OpenAI"
        },
        "production-anthropic": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.0,
            "max_tokens": 2000,
            "description": "Production configuration using Anthropic"
        },
        "local-development": {
            "provider": "ollama",
            "model": "llama2:7b",
            "temperature": 0.0,
            "max_tokens": 2000,
            "description": "Local development with Ollama"
        }
    },
    "providers": {
        "openai": {
            "api_key": "${OPENAI_API_KEY}",
            "organization": "${OPENAI_ORG_ID}",
            "base_url": "https://api.openai.com/v1",
            "models": {
                "gpt-4o-mini": {
                    "context_window": 128000,
                    "max_tokens": 16384
                },
                "gpt-4o": {
                    "context_window": 128000,
                    "max_tokens": 4096
                }
            }
        },
        "anthropic": {
            "api_key": "${ANTHROPIC_API_KEY}",
            "base_url": "https://api.anthropic.com",
            "models": {
                "claude-3-sonnet-20240229": {
                    "context_window": 200000,
                    "max_tokens": 4096
                },
                "claude-3-haiku-20240307": {
                    "context_window": 200000,
                    "max_tokens": 4096
                }
            }
        },
        "ollama": {
            "base_url": "${OLLAMA_BASE_URL}",
            "models": {
                "llama2:7b": {
                    "context_window": 4096,
                    "max_tokens": 2048
                }
            }
        }
    },
    "plugin_settings": {
        "logitranslate": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 2000,
            "validation_enabled": true
        },
        "logiattack": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 2000,
            "validation_enabled": true
        },
        "judge": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
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

### Configuration Profiles

The system supports predefined configuration profiles for different use cases:

```json
{
    "profiles": {
        "research-openai": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "description": "Research configuration using OpenAI"
        },
        "production-anthropic": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.0,
            "description": "Production configuration using Anthropic"
        },
        "local-development": {
            "provider": "ollama",
            "model": "llama2:7b",
            "temperature": 0.0,
            "description": "Local development with Ollama"
        }
    }
}
```

### Environment Variable Resolution

The system automatically resolves environment variables in configuration:

```json
{
    "providers": {
        "openai": {
            "api_key": "${OPENAI_API_KEY}",
            "organization": "${OPENAI_ORG_ID}"
        },
        "anthropic": {
            "api_key": "${ANTHROPIC_API_KEY}"
        }
    }
}
```

### Multi-Provider Plugin Configuration

Plugin settings now support provider-specific configurations:

```json
{
    "plugin_settings": {
        "logitranslate": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 2000,
            "validation_enabled": true,
            "custom_setting": "value"
        },
        "judge": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.0,
            "max_tokens": 1000
        }
    }
}
```

### Advanced Access Patterns

Configuration values can be accessed using multiple patterns:

```python
# Dot notation access
temperature = config.get("providers.openai.temperature")
profile_settings = config.get("profiles.research-openai")

# Provider-specific access
openai_settings = config.get_provider_settings("openai")
anthropic_models = config.get("providers.anthropic.models")

# Profile-based access
config.set_profile("production-anthropic")
current_model = config.get_model_for_plugin("logitranslate")
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