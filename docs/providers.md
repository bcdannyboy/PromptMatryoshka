# LLM Providers Documentation

## Overview

PromptMatryoshka supports multiple Large Language Model (LLM) providers through a unified interface, enabling researchers to seamlessly switch between different models and providers based on their specific needs. This multi-provider architecture allows for comparative studies, fallback scenarios, and optimization across different model capabilities.

## Supported Providers

### OpenAI
**Status:** ✅ Fully Supported  
**Models:** GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo  
**Use Cases:** Research, production, high-quality text generation

```python
# Configuration
{
    "provider": "openai",
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
}
```

### Anthropic
**Status:** ✅ Fully Supported  
**Models:** Claude-3 Sonnet, Claude-3 Haiku, Claude-3 Opus  
**Use Cases:** Safety-focused research, constitutional AI, content generation

```python
# Configuration
{
    "provider": "anthropic",
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
}
```

### Ollama
**Status:** ✅ Fully Supported  
**Models:** Llama2, Mistral, CodeLlama, custom models  
**Use Cases:** Local development, privacy-sensitive research, offline operations

```python
# Configuration
{
    "provider": "ollama",
    "base_url": "${OLLAMA_BASE_URL}",
    "models": {
        "llama2:7b": {
            "context_window": 4096,
            "max_tokens": 2048
        },
        "mistral:7b": {
            "context_window": 8192,
            "max_tokens": 4096
        }
    }
}
```

### HuggingFace
**Status:** ✅ Fully Supported  
**Models:** Open-source models via Inference API  
**Use Cases:** Research with open models, cost-effective solutions

```python
# Configuration
{
    "provider": "huggingface",
    "api_key": "${HUGGINGFACE_API_KEY}",
    "base_url": "https://api-inference.huggingface.co",
    "models": {
        "microsoft/DialoGPT-large": {
            "context_window": 1024,
            "max_tokens": 512
        }
    }
}
```

## Provider-Specific Features

### OpenAI Features
- **Function Calling**: Advanced function calling capabilities
- **Vision Models**: GPT-4V for image understanding
- **Fine-tuning**: Custom model fine-tuning support
- **Embeddings**: Text embedding generation

### Anthropic Features
- **Constitutional AI**: Built-in safety mechanisms
- **Long Context**: Exceptional long-context capabilities
- **Tool Use**: Advanced tool integration
- **Safety Filtering**: Enhanced content filtering

### Ollama Features
- **Local Execution**: Complete offline operation
- **Custom Models**: Support for custom fine-tuned models
- **GPU Acceleration**: CUDA and Metal support
- **Model Management**: Easy model installation and updates

### HuggingFace Features
- **Open Models**: Access to thousands of open-source models
- **Research Models**: Latest research models and architectures
- **Custom Deployments**: Deploy custom models via Inference API
- **Community Models**: Community-contributed models

## Provider Selection Guide

### Research Use Cases

#### Academic Research
- **Primary**: OpenAI (GPT-4o-mini) for consistency
- **Secondary**: Anthropic (Claude-3-Sonnet) for safety comparison
- **Tertiary**: Ollama (Llama2) for reproducibility

#### Industry Research
- **Primary**: Anthropic (Claude-3-Sonnet) for safety
- **Secondary**: OpenAI (GPT-4o) for performance
- **Tertiary**: HuggingFace for cost optimization

#### Privacy-Sensitive Research
- **Primary**: Ollama (local models) for privacy
- **Secondary**: HuggingFace (on-premises) for flexibility

### Production Use Cases

#### High-Volume Production
- **Primary**: OpenAI (GPT-4o-mini) for cost-effectiveness
- **Fallback**: Anthropic (Claude-3-Haiku) for reliability

#### Safety-Critical Applications
- **Primary**: Anthropic (Claude-3-Sonnet) for safety
- **Fallback**: OpenAI (GPT-4o) with additional filtering

#### Cost-Optimized Production
- **Primary**: HuggingFace (open models) for cost
- **Fallback**: OpenAI (GPT-4o-mini) for quality

## Environment Variables

Each provider requires specific environment variables for authentication:

### OpenAI
```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_ORG_ID="your-organization-id"  # Optional
```

### Anthropic
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Ollama
```bash
export OLLAMA_BASE_URL="http://localhost:11434"  # Default
```

### HuggingFace
```bash
export HUGGINGFACE_API_KEY="your-huggingface-token"
```

## Provider Configuration Examples

### Research Configuration
```json
{
    "active_profile": "research-mixed",
    "profiles": {
        "research-mixed": {
            "plugin_settings": {
                "logitranslate": {
                    "provider": "openai",
                    "model": "gpt-4o-mini"
                },
                "logiattack": {
                    "provider": "anthropic",
                    "model": "claude-3-sonnet-20240229"
                },
                "judge": {
                    "provider": "openai",
                    "model": "gpt-4o"
                }
            }
        }
    }
}
```

### Local Development Configuration
```json
{
    "active_profile": "local-development",
    "profiles": {
        "local-development": {
            "plugin_settings": {
                "logitranslate": {
                    "provider": "ollama",
                    "model": "llama2:7b"
                },
                "logiattack": {
                    "provider": "ollama",
                    "model": "mistral:7b"
                }
            }
        }
    }
}
```

## API Rate Limits and Considerations

### OpenAI
- **Rate Limits**: Varies by tier and model
- **Best Practices**: Implement exponential backoff
- **Cost**: Pay-per-token model
- **Latency**: Generally low latency

### Anthropic
- **Rate Limits**: Conservative rate limiting
- **Best Practices**: Batch requests when possible
- **Cost**: Competitive pricing
- **Latency**: Moderate latency

### Ollama
- **Rate Limits**: None (local execution)
- **Best Practices**: Monitor GPU memory usage
- **Cost**: Free (local compute costs)
- **Latency**: Depends on hardware

### HuggingFace
- **Rate Limits**: Varies by model and plan
- **Best Practices**: Use caching for repeated requests
- **Cost**: Free tier available
- **Latency**: Variable by model size

## Provider Health Monitoring

### Status Checking
```python
from promptmatryoshka.llm_factory import LLMFactory

# Check provider availability
providers = LLMFactory.get_available_providers()
for provider in providers:
    status = LLMFactory.check_provider_health(provider)
    print(f"{provider}: {status}")
```

### Fallback Configuration
```json
{
    "provider_fallbacks": {
        "openai": ["anthropic", "ollama"],
        "anthropic": ["openai", "huggingface"],
        "ollama": ["openai", "anthropic"],
        "huggingface": ["openai", "anthropic"]
    }
}
```

## Migration Between Providers

### Configuration Migration
```python
# Migrate from single-provider to multi-provider
from promptmatryoshka.config import get_config

config = get_config()
config.migrate_to_multi_provider()
```

### Data Format Compatibility
All providers use the same input/output format, ensuring seamless migration:

```python
# Standard interface across all providers
llm = LLMFactory.create_llm(provider="any_provider", model="any_model")
response = llm.generate("Your prompt here")
```

## Troubleshooting

### Common Issues

#### Authentication Errors
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Verify API key format
python -c "import os; print(len(os.getenv('OPENAI_API_KEY', '')))"
```

#### Network Connectivity
```python
# Test provider connectivity
from promptmatryoshka.llm_factory import LLMFactory

try:
    llm = LLMFactory.create_llm("openai", "gpt-4o-mini")
    response = llm.generate("Hello")
    print("Provider available")
except Exception as e:
    print(f"Provider unavailable: {e}")
```

#### Model Availability
```python
# Check model availability
provider_config = config.get_provider_settings("openai")
available_models = provider_config.get("models", {})
print(f"Available models: {list(available_models.keys())}")
```

### Performance Optimization

#### Caching
```python
# Enable response caching
{
    "cache_settings": {
        "enabled": true,
        "ttl": 3600,
        "max_size": 1000
    }
}
```

#### Connection Pooling
```python
# Configure connection pooling
{
    "connection_settings": {
        "pool_size": 10,
        "max_retries": 3,
        "timeout": 30
    }
}
```

## Best Practices

### Provider Selection
1. **Start with OpenAI** for consistency and reliability
2. **Use Anthropic** for safety-critical applications
3. **Consider Ollama** for privacy and offline requirements
4. **Explore HuggingFace** for cost optimization and research

### Configuration Management
1. **Use profiles** for different environments
2. **Set up fallbacks** for reliability
3. **Monitor costs** across providers
4. **Validate configurations** before deployment

### Security Considerations
1. **Secure API keys** using environment variables
2. **Rotate credentials** regularly
3. **Monitor usage** for anomalies
4. **Use least privilege** access patterns

This multi-provider architecture enables PromptMatryoshka to be flexible, reliable, and adaptable to various research and production needs while maintaining a consistent interface across all supported providers.