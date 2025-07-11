# LLM Interface Module Documentation

## Purpose & Overview

The [`llm_interface.py`](../promptmatryoshka/llm_interface.py) module provides a comprehensive multi-provider LLM interface system for the PromptMatryoshka framework. This module implements concrete interfaces for **OpenAI, Anthropic, Ollama, and HuggingFace providers**, enabling the framework to work seamlessly with various language models through a standardized interface with automatic provider detection, configuration management, and error handling.

## Architecture

The LLM interface follows a factory pattern with concrete implementations for multiple providers:

```
┌─────────────────────────────────────────────────────────────────┐
│                Multi-Provider LLM Interface Layer              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                LLMInterface                             │   │
│  │              (Abstract Base)                            │   │
│  │                                                         │   │
│  │  • generate(prompt) -> str                              │   │
│  │  • Authentication handling                              │   │
│  │  • Response parsing                                     │   │
│  │  • Error handling                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                 │                               │
│                                 ▼                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LLMFactory                                 │   │
│  │                                                         │   │
│  │  • Provider Registration                                │   │
│  │  • Dynamic Creation                                     │   │
│  │  • Configuration Management                             │   │
│  │  • Validation                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                 │                               │
│                                 ▼                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Provider Implementations                   │   │
│  │                                                         │   │
│  │  • OpenAIInterface       • AnthropicInterface           │   │
│  │  • OllamaInterface       • HuggingFaceInterface         │   │
│  │  • Error Handling        • Authentication              │   │
│  │  • Response Parsing      • Configuration               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Classes/Functions

### LLMInterface Class

The abstract base class that defines the standard interface for all LLM implementations.

```python
class LLMInterface:
    """
    Abstract base class for LLM API interactions.

    Methods:
        generate(prompt: str) -> str
            Sends prompt to the LLM and returns the generated response.
        validate_config() -> bool
            Validates the LLM configuration.
        get_model_info() -> dict
            Returns information about the current model.
    """
    pass
```

### LLMFactory Class

Factory class for creating and managing LLM provider instances.

```python
class LLMFactory:
    """Factory for creating LLM instances based on provider type."""
    
    @staticmethod
    def create_llm(provider: str, model: str, **kwargs) -> LLMInterface
    
    @staticmethod
    def get_available_providers() -> List[str]
    
    @staticmethod
    def validate_provider_config(provider: str, config: Dict) -> bool
```

### Provider Implementations

#### OpenAIInterface
```python
class OpenAIInterface(LLMInterface):
    """OpenAI GPT implementation of LLMInterface."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs)
    def generate(self, prompt: str) -> str
    def validate_config(self) -> bool
    def get_model_info(self) -> dict
```

#### AnthropicInterface
```python
class AnthropicInterface(LLMInterface):
    """Anthropic Claude implementation of LLMInterface."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", **kwargs)
    def generate(self, prompt: str) -> str
    def validate_config(self) -> bool
    def get_model_info(self) -> dict
```

#### OllamaInterface
```python
class OllamaInterface(LLMInterface):
    """Ollama local model implementation of LLMInterface."""
    
    def __init__(self, base_url: str, model: str = "llama2:7b", **kwargs)
    def generate(self, prompt: str) -> str
    def validate_config(self) -> bool
    def get_model_info(self) -> dict
```

#### HuggingFaceInterface
```python
class HuggingFaceInterface(LLMInterface):
    """HuggingFace model implementation of LLMInterface."""
    
    def __init__(self, api_key: str, model: str, **kwargs)
    def generate(self, prompt: str) -> str
    def validate_config(self) -> bool
    def get_model_info(self) -> dict
```

## Usage Examples

### Multi-Provider LLM Creation

```python
from promptmatryoshka.llm_factory import LLMFactory

# Create OpenAI interface
openai_llm = LLMFactory.create_llm(
    provider="openai",
    model="gpt-4o-mini",
    api_key="your-api-key",
    temperature=0.0
)

# Create Anthropic interface
anthropic_llm = LLMFactory.create_llm(
    provider="anthropic",
    model="claude-3-sonnet-20240229",
    api_key="your-api-key",
    temperature=0.0
)

# Create Ollama interface
ollama_llm = LLMFactory.create_llm(
    provider="ollama",
    model="llama2:7b",
    base_url="http://localhost:11434"
)
```

### Configuration-Based Creation

```python
from promptmatryoshka.llm_factory import LLMFactory
from promptmatryoshka.config import get_config

# Create LLM from configuration
config = get_config()
provider_settings = config.get_provider_settings("openai")

llm = LLMFactory.create_llm(
    provider="openai",
    **provider_settings
)
```

### Usage in Plugins

```python
from promptmatryoshka.llm_interface import LLMInterface
from promptmatryoshka.llm_factory import LLMFactory

class LogiTranslatePlugin(PluginBase):
    """Plugin that uses LLM interface for translation."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        super().__init__()
        self.llm = LLMFactory.create_llm(
            provider=provider,
            model=model,
            **self.get_provider_config(provider)
        )
    
    def run(self, prompt: str) -> str:
        """Run translation using LLM interface."""
        translation_prompt = f"Translate to logic: {prompt}"
        return self.llm.generate(translation_prompt)
```

### Provider Validation and Info

```python
from promptmatryoshka.llm_factory import LLMFactory

# List available providers
providers = LLMFactory.get_available_providers()
print(f"Available providers: {providers}")

# Validate provider configuration
config = {"api_key": "test-key", "model": "gpt-4o-mini"}
is_valid = LLMFactory.validate_provider_config("openai", config)

# Get model information
llm = LLMFactory.create_llm("openai", "gpt-4o-mini")
model_info = llm.get_model_info()
print(f"Model info: {model_info}")
```

## Integration Points

### Plugin System Integration

The LLM interface is designed to integrate seamlessly with the plugin system:

- **Plugin Dependencies**: Plugins can depend on LLM interfaces for model interactions
- **Configuration Integration**: LLM settings from [`config.py`](../promptmatryoshka/config.py) can be passed to LLM interfaces
- **Error Handling**: LLM interfaces handle API errors and provide consistent error reporting

### Configuration Integration

LLM interfaces integrate with the configuration system:

```python
from promptmatryoshka.config import get_config

config = get_config()
llm_settings = config.get_llm_settings_for_plugin("logitranslate")
model = config.get_model_for_plugin("logitranslate")

# Create LLM interface with configuration
llm_interface = OpenAIInterface(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=model,
    **llm_settings
)
```

### Logging Integration

LLM interfaces should integrate with the logging system:

```python
from promptmatryoshka.logging_utils import get_logger

class BaseLLMInterface(LLMInterface):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def generate(self, prompt: str) -> str:
        self.logger.debug(f"Generating response for prompt: {prompt[:50]}...")
        # Implementation here
        self.logger.debug("Response generated successfully")
```

## Configuration

### LLM Settings

The interface should support standard LLM configuration parameters:

```python
class LLMInterface:
    def __init__(self, model: str, temperature: float = 0.0, 
                 max_tokens: int = 2000, top_p: float = 1.0,
                 frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                 request_timeout: int = 120):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.request_timeout = request_timeout
```

### Authentication

LLM interfaces should handle authentication securely:

```python
class SecureLLMInterface(LLMInterface):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required")
```

## Error Handling

### API Error Handling

LLM interfaces should handle common API errors:

```python
class RobustLLMInterface(LLMInterface):
    def generate(self, prompt: str) -> str:
        try:
            return self._make_api_call(prompt)
        except RateLimitError:
            self.logger.warning("Rate limit exceeded, retrying...")
            time.sleep(60)
            return self._make_api_call(prompt)
        except AuthenticationError:
            self.logger.error("Authentication failed")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise
```

### Retry Logic

Implement retry logic for transient failures:

```python
import time
from functools import wraps

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator
```

## Developer Notes

### Future Implementation Strategy

The LLM interface is designed to be extended with concrete implementations:

1. **OpenAI Interface**: For GPT models via OpenAI API
2. **Anthropic Interface**: For Claude models via Anthropic API
3. **HuggingFace Interface**: For open-source models via HuggingFace
4. **Local Interface**: For locally hosted models

### Abstract Methods

Future implementations should implement these core methods:

```python
from abc import ABC, abstractmethod

class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from the LLM."""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the LLM configuration."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        pass
```

### Performance Considerations

- **Connection Pooling**: Reuse connections for better performance
- **Caching**: Cache responses for identical prompts
- **Batch Processing**: Support batch API calls where available
- **Async Support**: Consider async interfaces for better concurrency

### Security Considerations

- **API Key Management**: Secure storage and handling of API keys
- **Input Validation**: Validate prompts before sending to API
- **Output Sanitization**: Sanitize responses from LLMs
- **Rate Limiting**: Implement client-side rate limiting

### Testing Strategy

```python
class MockLLMInterface(LLMInterface):
    """Mock implementation for testing."""
    
    def __init__(self, responses: dict = None):
        self.responses = responses or {}
        self.call_count = 0
    
    def generate(self, prompt: str) -> str:
        self.call_count += 1
        return self.responses.get(prompt, "Mock response")
```

### Extension Points

The interface provides several extension points:

1. **Custom Providers**: Add support for new LLM providers
2. **Middleware**: Add middleware for logging, caching, etc.
3. **Response Processing**: Custom response parsing and validation
4. **Error Handling**: Custom error handling strategies

## Implementation Details

### Current Implementation

The current implementation is minimal:

```python
class LLMInterface:
    """
    Abstracts LLM API calls.

    Methods:
        generate(prompt: str) -> str
            Sends prompt to the LLM and returns the generated response.
    """
    pass
```

### Planned Enhancements

1. **Abstract Base Class**: Convert to proper ABC with abstract methods
2. **Standard Interface**: Define standard methods for all implementations
3. **Configuration Support**: Built-in configuration management
4. **Error Handling**: Comprehensive error handling framework
5. **Async Support**: Asynchronous API support
6. **Streaming**: Support for streaming responses
7. **Batch Processing**: Batch API call support

### Integration with Existing Plugins

Currently, plugins implement their own LLM interactions. Future enhancements will:

1. **Refactor Plugins**: Update plugins to use LLM interfaces
2. **Dependency Injection**: Inject LLM interfaces into plugins
3. **Configuration Binding**: Bind plugin configurations to LLM interfaces
4. **Testing Support**: Provide mock interfaces for testing

### Backward Compatibility

The interface is designed to maintain backward compatibility:

- **Optional Migration**: Plugins can gradually migrate to use interfaces
- **Fallback Support**: Support for direct API calls where needed
- **Configuration Compatibility**: Maintain compatibility with existing configs

## Future Roadmap

### Short-term Goals

1. **Convert to ABC**: Make LLMInterface a proper abstract base class
2. **OpenAI Implementation**: Create OpenAIInterface implementation
3. **Plugin Integration**: Integrate with LogiTranslate and LogiAttack plugins

### Medium-term Goals

1. **Multiple Providers**: Support for Anthropic, HuggingFace, etc.
2. **Async Support**: Asynchronous API support
3. **Caching Layer**: Response caching for improved performance
4. **Monitoring**: Built-in monitoring and metrics collection

### Long-term Goals

1. **Local Models**: Support for locally hosted models
2. **Custom Endpoints**: Support for custom API endpoints
3. **Model Switching**: Dynamic model switching based on context
4. **Fine-tuning**: Support for fine-tuned models

The LLM interface module represents a foundational component that will enable the PromptMatryoshka framework to support multiple LLM providers and deployment scenarios while maintaining a consistent, testable interface.