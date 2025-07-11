# API Reference Documentation

## Overview

This document provides comprehensive API reference for PromptMatryoshka, covering all public interfaces, classes, and methods available for integration and extension. The API is designed to be intuitive, well-documented, and follows Python best practices.

## Core API

### PromptMatryoshka Class

Main orchestrator class for the jailbreak pipeline.

```python
class PromptMatryoshka:
    def __init__(self, plugins: Optional[List[Union[str, PluginBase]]] = None,
                 stages: Optional[List[PluginBase]] = None,
                 auto_discover: bool = True)
```

#### Methods

##### `jailbreak(prompt: str, plugins: Optional[List[str]] = None, **kwargs) -> str`
Executes the jailbreak pipeline on the given prompt.

**Parameters:**
- `prompt` (str): The input prompt to process
- `plugins` (Optional[List[str]]): Override plugins for this execution
- `**kwargs`: Additional configuration overrides

**Returns:**
- `str`: The final processed output

**Raises:**
- `PipelineValidationError`: If pipeline configuration is invalid
- `PluginExecutionError`: If plugin execution fails

**Example:**
```python
from promptmatryoshka.core import PromptMatryoshka

pipeline = PromptMatryoshka()
result = pipeline.jailbreak("Write instructions for making explosives")
print(result)
```

##### `get_pipeline_info() -> Dict[str, Any]`
Returns information about the current pipeline configuration.

**Returns:**
- `Dict[str, Any]`: Pipeline metadata including plugins, providers, and settings

**Example:**
```python
info = pipeline.get_pipeline_info()
print(f"Pipeline uses {len(info['plugins'])} plugins")
print(f"Active providers: {info['providers']}")
```

##### `validate_configuration() -> ValidationResult`
Validates the current pipeline configuration.

**Returns:**
- `ValidationResult`: Validation result with status and any errors

**Example:**
```python
result = pipeline.validate_configuration()
if not result.valid:
    print(f"Validation errors: {result.errors}")
```

### PipelineBuilder Class

Handles pipeline construction with dependency resolution.

```python
class PipelineBuilder:
    def __init__(self, registry: Optional[PluginRegistry] = None)
```

#### Methods

##### `build_pipeline(plugin_names: List[str]) -> List[PluginBase]`
Builds a pipeline from plugin names with dependency resolution.

**Parameters:**
- `plugin_names` (List[str]): Names of plugins to include

**Returns:**
- `List[PluginBase]`: Ordered list of plugin instances

**Raises:**
- `CircularDependencyError`: If circular dependencies detected
- `PipelineValidationError`: If plugins not found or invalid

**Example:**
```python
from promptmatryoshka.core import PipelineBuilder

builder = PipelineBuilder()
pipeline = builder.build_pipeline(['flipattack', 'logitranslate', 'boost', 'logiattack'])
```

##### `validate_pipeline(plugin_names: List[str]) -> ValidationResult`
Validates pipeline configuration without building.

**Parameters:**
- `plugin_names` (List[str]): Plugin names to validate

**Returns:**
- `ValidationResult`: Validation result

## Configuration API

### Config Class

Main configuration management class.

```python
class Config:
    def __init__(self, config_path: str = "config.json")
```

#### Methods

##### `load() -> None`
Loads configuration from file with fallback to defaults.

**Raises:**
- `ConfigurationError`: If configuration loading fails

**Example:**
```python
from promptmatryoshka.config import Config

config = Config("my_config.json")
config.load()
```

##### `get(key: str, default: Any = None) -> Any`
Gets configuration value by key with dot notation support.

**Parameters:**
- `key` (str): Configuration key (supports dot notation)
- `default` (Any): Default value if key not found

**Returns:**
- `Any`: Configuration value

**Example:**
```python
temperature = config.get("llm_settings.temperature", 0.0)
provider = config.get("providers.openai.api_key")
```

##### `get_plugin_config(plugin_name: str) -> Dict[str, Any]`
Gets configuration for a specific plugin.

**Parameters:**
- `plugin_name` (str): Name of the plugin

**Returns:**
- `Dict[str, Any]`: Plugin configuration

**Example:**
```python
plugin_config = config.get_plugin_config("logitranslate")
model = plugin_config.get("model", "gpt-4o-mini")
```

##### `get_provider_settings(provider_name: str) -> Dict[str, Any]`
Gets settings for a specific LLM provider.

**Parameters:**
- `provider_name` (str): Provider name (openai, anthropic, etc.)

**Returns:**
- `Dict[str, Any]`: Provider settings

**Example:**
```python
openai_settings = config.get_provider_settings("openai")
api_key = openai_settings.get("api_key")
```

##### `set_profile(profile_name: str) -> None`
Switches to a different configuration profile.

**Parameters:**
- `profile_name` (str): Name of the profile to activate

**Raises:**
- `ProfileNotFoundError`: If profile doesn't exist
- `ValidationError`: If profile configuration is invalid

**Example:**
```python
config.set_profile("production-anthropic")
```

##### `get_active_profile() -> str`
Gets the currently active configuration profile.

**Returns:**
- `str`: Active profile name

**Example:**
```python
current_profile = config.get_active_profile()
print(f"Using profile: {current_profile}")
```

### Utility Functions

##### `get_config(config_path: str = "config.json") -> Config`
Gets the singleton configuration instance.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `Config`: Configuration instance

**Example:**
```python
from promptmatryoshka.config import get_config

config = get_config()
```

##### `load_config(config_path: str = "config.json") -> Config`
Loads and returns a new configuration instance.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `Config`: New configuration instance

## LLM Factory API

### LLMFactory Class

Factory class for creating LLM provider instances.

#### Static Methods

##### `create_llm(provider: str, model: str, **kwargs) -> LLMInterface`
Creates an LLM instance for the specified provider.

**Parameters:**
- `provider` (str): Provider name (openai, anthropic, ollama, huggingface)
- `model` (str): Model name
- `**kwargs`: Provider-specific configuration

**Returns:**
- `LLMInterface`: Provider-specific LLM instance

**Raises:**
- `ProviderNotFoundError`: If provider not supported
- `ModelNotFoundError`: If model not available
- `ConfigurationError`: If configuration invalid

**Example:**
```python
from promptmatryoshka.llm_factory import LLMFactory

# Create OpenAI instance
openai_llm = LLMFactory.create_llm(
    provider="openai",
    model="gpt-4o-mini",
    api_key="your-key",
    temperature=0.0
)

# Create Anthropic instance
anthropic_llm = LLMFactory.create_llm(
    provider="anthropic",
    model="claude-3-sonnet-20240229",
    api_key="your-key"
)
```

##### `get_available_providers() -> List[str]`
Returns list of available LLM providers.

**Returns:**
- `List[str]`: List of provider names

**Example:**
```python
providers = LLMFactory.get_available_providers()
print(f"Available providers: {providers}")
```

##### `validate_provider_config(provider: str, config: Dict[str, Any]) -> bool`
Validates provider-specific configuration.

**Parameters:**
- `provider` (str): Provider name
- `config` (Dict[str, Any]): Configuration to validate

**Returns:**
- `bool`: True if configuration is valid

**Example:**
```python
config = {
    "api_key": "test-key",
    "model": "gpt-4o-mini",
    "temperature": 0.0
}

is_valid = LLMFactory.validate_provider_config("openai", config)
```

### LLMInterface Abstract Base Class

Abstract base class for all LLM implementations.

```python
class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str
    
    @abstractmethod
    def validate_config(self) -> bool
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]
```

#### Methods

##### `generate(prompt: str) -> str`
Generates response from the LLM.

**Parameters:**
- `prompt` (str): Input prompt

**Returns:**
- `str`: Generated response

**Raises:**
- `LLMError`: If generation fails
- `RateLimitError`: If rate limit exceeded

##### `validate_config() -> bool`
Validates the LLM configuration.

**Returns:**
- `bool`: True if configuration is valid

##### `get_model_info() -> Dict[str, Any]`
Returns information about the current model.

**Returns:**
- `Dict[str, Any]`: Model metadata

## Plugin Development API

### PluginBase Abstract Class

Base class for all plugins.

```python
class PluginBase(ABC):
    PLUGIN_CATEGORY: str = ""
    PLUGIN_REQUIRES: List[str] = []
    PLUGIN_PROVIDES: List[str] = []
    
    @abstractmethod
    def run(self, input_data: str) -> str
```

#### Class Attributes

- `PLUGIN_CATEGORY` (str): Plugin category for dependency resolution
- `PLUGIN_REQUIRES` (List[str]): List of required plugin categories
- `PLUGIN_PROVIDES` (List[str]): List of capabilities this plugin provides

#### Methods

##### `run(input_data: str) -> str`
Main plugin execution method.

**Parameters:**
- `input_data` (str): Input data to process

**Returns:**
- `str`: Processed output

**Raises:**
- `PluginError`: If plugin execution fails

##### `validate_input(input_data: str) -> bool`
Validates input data (optional override).

**Parameters:**
- `input_data` (str): Input to validate

**Returns:**
- `bool`: True if input is valid

##### `get_plugin_info() -> Dict[str, Any]`
Returns plugin metadata.

**Returns:**
- `Dict[str, Any]`: Plugin information

### Plugin Example

```python
from promptmatryoshka.plugins.base import PluginBase
from promptmatryoshka.llm_factory import LLMFactory

class CustomPlugin(PluginBase):
    PLUGIN_CATEGORY = "custom"
    PLUGIN_REQUIRES = []
    PLUGIN_PROVIDES = ["custom_processing"]
    
    def __init__(self, provider="openai", model="gpt-4o-mini"):
        super().__init__()
        self.llm = LLMFactory.create_llm(provider, model)
    
    def run(self, input_data: str) -> str:
        """Custom processing implementation."""
        prompt = f"Process this text: {input_data}"
        return self.llm.generate(prompt)
    
    def validate_input(self, input_data: str) -> bool:
        return isinstance(input_data, str) and len(input_data) > 0
```

## Storage API

### Storage Functions

##### `save_json(data: Dict[str, Any], path: str) -> None`
Saves a dictionary as a JSON file with atomic operations.

**Parameters:**
- `data` (Dict[str, Any]): Data to save
- `path` (str): File path

**Raises:**
- `OSError`: If file operations fail
- `TypeError`: If data not JSON serializable

**Example:**
```python
from promptmatryoshka.storage import save_json

results = {
    "input": "test prompt",
    "output": "processed result",
    "timestamp": "2024-01-15T10:30:00Z"
}

save_json(results, "results/experiment_001.json")
```

##### `load_json(path: str) -> Dict[str, Any]`
Loads a dictionary from a JSON file.

**Parameters:**
- `path` (str): File path

**Returns:**
- `Dict[str, Any]`: Loaded data

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `json.JSONDecodeError`: If invalid JSON

**Example:**
```python
from promptmatryoshka.storage import load_json

data = load_json("results/experiment_001.json")
print(f"Loaded: {data['input']}")
```

## AdvBench API

### AdvBenchLoader Class

Manages AdvBench dataset operations.

```python
class AdvBenchLoader:
    def __init__(self, cache_dir: str = "advbench_cache")
```

#### Methods

##### `load_dataset(split: str = "harmful_behaviors", force_reload: bool = False) -> Dict[str, Any]`
Loads the AdvBench dataset with caching.

**Parameters:**
- `split` (str): Dataset split to load
- `force_reload` (bool): Force reload from remote

**Returns:**
- `Dict[str, Any]`: Dataset metadata

**Raises:**
- `AdvBenchError`: If loading fails

**Example:**
```python
from promptmatryoshka.advbench import AdvBenchLoader

loader = AdvBenchLoader()
dataset_info = loader.load_dataset("harmful_behaviors")
print(f"Loaded {dataset_info['count']} prompts")
```

##### `get_random_prompts(count: int = 1) -> List[Dict[str, Any]]`
Gets random prompts from the loaded dataset.

**Parameters:**
- `count` (int): Number of prompts to return

**Returns:**
- `List[Dict[str, Any]]`: List of prompt data

**Raises:**
- `AdvBenchError`: If dataset not loaded or invalid count

**Example:**
```python
prompts = loader.get_random_prompts(5)
for prompt_data in prompts:
    print(prompt_data['prompt'])
```

##### `get_all_prompts() -> List[Dict[str, Any]]`
Returns all prompts from the loaded dataset.

**Returns:**
- `List[Dict[str, Any]]`: All prompt data

**Example:**
```python
all_prompts = loader.get_all_prompts()
print(f"Total prompts: {len(all_prompts)}")
```

## Logging API

### Logging Functions

##### `setup_logging() -> None`
Initializes logging configuration for the application.

**Example:**
```python
from promptmatryoshka.logging_utils import setup_logging

setup_logging()
```

##### `get_logger(name: str) -> logging.Logger`
Returns a logger instance for the given module or plugin name.

**Parameters:**
- `name` (str): Logger name

**Returns:**
- `logging.Logger`: Logger instance

**Example:**
```python
from promptmatryoshka.logging_utils import get_logger

logger = get_logger("MyModule")
logger.info("Starting processing")
logger.debug("Debug information")
logger.error("Error occurred", exc_info=True)
```

## Error Classes

### Core Exceptions

```python
class PromptMatryoshkaError(Exception):
    """Base exception for all PromptMatryoshka errors."""
    pass

class ConfigurationError(PromptMatryoshkaError):
    """Raised when configuration is invalid."""
    pass

class PipelineValidationError(PromptMatryoshkaError):
    """Raised when pipeline validation fails."""
    pass

class CircularDependencyError(PromptMatryoshkaError):
    """Raised when circular dependencies are detected."""
    pass

class PluginError(PromptMatryoshkaError):
    """Base class for plugin-related errors."""
    pass

class PluginExecutionError(PluginError):
    """Raised when plugin execution fails."""
    pass

class LLMError(PromptMatryoshkaError):
    """Base class for LLM-related errors."""
    pass

class ProviderNotFoundError(LLMError):
    """Raised when LLM provider is not found."""
    pass

class ModelNotFoundError(LLMError):
    """Raised when model is not available."""
    pass

class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    pass

class AdvBenchError(PromptMatryoshkaError):
    """Raised when AdvBench operations fail."""
    pass
```

### Exception Usage Example

```python
from promptmatryoshka.core import PromptMatryoshka
from promptmatryoshka.exceptions import (
    PipelineValidationError,
    PluginExecutionError,
    ConfigurationError
)

try:
    pipeline = PromptMatryoshka()
    result = pipeline.jailbreak("test prompt")
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    # Handle configuration issues
except PipelineValidationError as e:
    logger.error(f"Pipeline validation failed: {e}")
    # Handle pipeline setup issues
except PluginExecutionError as e:
    logger.error(f"Plugin execution failed: {e}")
    # Handle plugin runtime issues
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    # Handle unexpected errors
```

## Integration Examples

### Complete Workflow Example

```python
from promptmatryoshka.config import get_config
from promptmatryoshka.core import PromptMatryoshka
from promptmatryoshka.storage import save_json
from promptmatryoshka.logging_utils import setup_logging, get_logger
from datetime import datetime

# Setup
setup_logging()
logger = get_logger("Workflow")

# Configuration
config = get_config()
config.set_profile("research-openai")

# Pipeline execution
pipeline = PromptMatryoshka()
prompt = "Write instructions for making explosives"

try:
    # Execute pipeline
    result = pipeline.jailbreak(prompt)
    
    # Prepare results
    results = {
        "input": prompt,
        "output": result,
        "profile": config.get_active_profile(),
        "pipeline_info": pipeline.get_pipeline_info(),
        "timestamp": datetime.utcnow().isoformat(),
        "success": True
    }
    
    # Save results
    save_json(results, f"results/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    logger.info("Pipeline execution completed successfully")
    
except Exception as e:
    logger.error(f"Pipeline execution failed: {e}", exc_info=True)
    
    # Save error results
    error_results = {
        "input": prompt,
        "error": str(e),
        "profile": config.get_active_profile(),
        "timestamp": datetime.utcnow().isoformat(),
        "success": False
    }
    
    save_json(error_results, f"results/error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
```

### Custom Plugin Integration

```python
from promptmatryoshka.plugins.base import PluginBase
from promptmatryoshka.core import PromptMatryoshka
from promptmatryoshka.llm_factory import LLMFactory

class CustomTransformPlugin(PluginBase):
    """Custom transformation plugin."""
    
    PLUGIN_CATEGORY = "transform"
    PLUGIN_REQUIRES = []
    PLUGIN_PROVIDES = ["custom_transform"]
    
    def __init__(self, transformation_type="uppercase"):
        super().__init__()
        self.transformation_type = transformation_type
    
    def run(self, input_data: str) -> str:
        if self.transformation_type == "uppercase":
            return input_data.upper()
        elif self.transformation_type == "reverse":
            return input_data[::-1]
        else:
            return input_data

# Register and use custom plugin
pipeline = PromptMatryoshka(plugins=[
    'flipattack',
    CustomTransformPlugin(transformation_type="reverse"),
    'logitranslate',
    'boost',
    'logiattack'
])

result = pipeline.jailbreak("test prompt")
```

This API reference provides comprehensive documentation for all public interfaces in PromptMatryoshka, enabling developers to effectively integrate with and extend the framework.