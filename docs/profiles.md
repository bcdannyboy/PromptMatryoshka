# Configuration Profiles Documentation

## Overview

Configuration profiles in PromptMatryoshka provide pre-configured setups for different use cases, environments, and research scenarios. Profiles allow users to quickly switch between optimized configurations without manually adjusting individual settings, ensuring consistency and reproducibility across different contexts.

## Built-in Profiles

### research-openai
**Purpose:** General research using OpenAI models  
**Use Case:** Academic research, benchmarking, consistent results

```json
{
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "max_tokens": 2000,
    "description": "Research configuration using OpenAI GPT-4o-mini for consistency",
    "plugin_settings": {
        "logitranslate": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0
        },
        "logiattack": {
            "provider": "openai", 
            "model": "gpt-4o-mini",
            "temperature": 0.0
        },
        "judge": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.0
        }
    }
}
```

### production-anthropic
**Purpose:** Production deployment with safety focus  
**Use Case:** Enterprise deployments, safety-critical applications

```json
{
    "provider": "anthropic",
    "model": "claude-3-sonnet-20240229",
    "temperature": 0.0,
    "max_tokens": 2000,
    "description": "Production configuration using Anthropic Claude for safety",
    "plugin_settings": {
        "logitranslate": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.0
        },
        "logiattack": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.0
        },
        "judge": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.0
        }
    }
}
```

### local-development
**Purpose:** Local development and testing  
**Use Case:** Development, privacy-sensitive work, offline operation

```json
{
    "provider": "ollama",
    "model": "llama2:7b",
    "temperature": 0.0,
    "max_tokens": 2000,
    "description": "Local development using Ollama models",
    "plugin_settings": {
        "logitranslate": {
            "provider": "ollama",
            "model": "llama2:7b",
            "temperature": 0.0
        },
        "logiattack": {
            "provider": "ollama",
            "model": "mistral:7b",
            "temperature": 0.0
        },
        "judge": {
            "provider": "ollama",
            "model": "llama2:7b",
            "temperature": 0.0
        }
    }
}
```

### high-performance
**Purpose:** Maximum performance with premium models  
**Use Case:** Critical research, high-quality results needed

```json
{
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.0,
    "max_tokens": 4000,
    "description": "High-performance configuration using premium models",
    "plugin_settings": {
        "logitranslate": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.0
        },
        "logiattack": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.0
        },
        "judge": {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229",
            "temperature": 0.0
        }
    }
}
```

### cost-optimized
**Purpose:** Minimize costs while maintaining functionality  
**Use Case:** Large-scale testing, budget-conscious research

```json
{
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "max_tokens": 1000,
    "description": "Cost-optimized configuration with efficient models",
    "plugin_settings": {
        "logitranslate": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 1000
        },
        "logiattack": {
            "provider": "openai",
            "model": "gpt-4o-mini", 
            "temperature": 0.0,
            "max_tokens": 1000
        },
        "judge": {
            "provider": "huggingface",
            "model": "microsoft/DialoGPT-large",
            "temperature": 0.0,
            "max_tokens": 500
        }
    }
}
```

### testing
**Purpose:** Automated testing and CI/CD  
**Use Case:** Unit tests, integration tests, continuous integration

```json
{
    "provider": "mock",
    "model": "mock-model",
    "temperature": 0.0,
    "max_tokens": 100,
    "description": "Testing configuration with mock providers",
    "plugin_settings": {
        "logitranslate": {
            "provider": "mock",
            "model": "mock-model",
            "response": "Mock translation response"
        },
        "logiattack": {
            "provider": "mock",
            "model": "mock-model",
            "response": "Mock attack response"
        },
        "judge": {
            "provider": "mock",
            "model": "mock-model", 
            "response": "Mock judge response"
        }
    }
}
```

## Profile Management

### Listing Available Profiles

```bash
# CLI command
python -m promptmatryoshka.cli list-profiles

# CLI with JSON output
python -m promptmatryoshka.cli list-profiles --json
```

```python
# Python API
from promptmatryoshka.config import get_config

config = get_config()
profiles = config.get_available_profiles()
print(f"Available profiles: {profiles}")
```

### Setting Active Profile

```bash
# CLI command
python -m promptmatryoshka.cli set-profile production-anthropic
```

```python
# Python API
from promptmatryoshka.config import get_config

config = get_config()
config.set_profile("production-anthropic")
```

### Getting Current Profile

```python
from promptmatryoshka.config import get_config

config = get_config()
current_profile = config.get_active_profile()
print(f"Current profile: {current_profile}")
```

## Creating Custom Profiles

### Profile Configuration Structure

```json
{
    "profiles": {
        "my-custom-profile": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 1500,
            "description": "My custom research configuration",
            "plugin_settings": {
                "logitranslate": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "custom_setting": "custom_value"
                },
                "logiattack": {
                    "provider": "anthropic",
                    "model": "claude-3-haiku-20240307",
                    "temperature": 0.0
                }
            },
            "environment_specific": {
                "logging_level": "DEBUG",
                "save_artifacts": true
            }
        }
    }
}
```

### Custom Profile Example

```json
{
    "profiles": {
        "multilingual-research": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.0,
            "max_tokens": 3000,
            "description": "Multilingual research with enhanced context",
            "plugin_settings": {
                "logitranslate": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "temperature": 0.0,
                    "system_prompt": "You are an expert in formal logic and multilingual translation.",
                    "max_tokens": 3000
                },
                "logiattack": {
                    "provider": "anthropic",
                    "model": "claude-3-sonnet-20240229",
                    "temperature": 0.0,
                    "max_tokens": 3000
                },
                "judge": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "temperature": 0.0,
                    "evaluation_criteria": ["safety", "effectiveness", "multilingual_accuracy"]
                }
            }
        }
    }
}
```

## Profile Inheritance

### Base Profile Configuration

Profiles can inherit from base configurations to reduce duplication:

```json
{
    "profile_defaults": {
        "base_research": {
            "temperature": 0.0,
            "max_tokens": 2000,
            "validation_enabled": true,
            "retry_attempts": 3
        }
    },
    "profiles": {
        "openai-research": {
            "inherits": "base_research",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "description": "OpenAI research configuration"
        },
        "anthropic-research": {
            "inherits": "base_research", 
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "description": "Anthropic research configuration"
        }
    }
}
```

## Environment-Specific Profiles

### Development Environment

```json
{
    "profiles": {
        "development": {
            "provider": "ollama",
            "model": "llama2:7b",
            "temperature": 0.0,
            "max_tokens": 1000,
            "description": "Development environment configuration",
            "environment": "development",
            "debug_mode": true,
            "log_level": "DEBUG",
            "save_artifacts": true
        }
    }
}
```

### Staging Environment

```json
{
    "profiles": {
        "staging": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 2000,
            "description": "Staging environment configuration",
            "environment": "staging",
            "debug_mode": false,
            "log_level": "INFO",
            "rate_limiting": true
        }
    }
}
```

### Production Environment

```json
{
    "profiles": {
        "production": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.0,
            "max_tokens": 2000,
            "description": "Production environment configuration",
            "environment": "production",
            "debug_mode": false,
            "log_level": "WARNING",
            "monitoring_enabled": true,
            "fallback_providers": ["openai"]
        }
    }
}
```

## Profile Validation

### Automatic Validation

```python
from promptmatryoshka.config import get_config

config = get_config()

# Validate profile configuration
try:
    config.set_profile("my-profile")
    print("Profile is valid")
except ValidationError as e:
    print(f"Profile validation failed: {e}")
```

### Manual Validation

```bash
# CLI validation
python -m promptmatryoshka.cli validate-config --profile my-profile
```

### Validation Rules

1. **Provider Availability**: Ensure specified providers are available
2. **Model Compatibility**: Verify models exist for the provider
3. **Setting Constraints**: Check parameter ranges and types
4. **Dependencies**: Validate plugin dependencies
5. **Environment Variables**: Check required environment variables

## Profile Switching Strategies

### Runtime Profile Switching

```python
from promptmatryoshka.config import get_config
from promptmatryoshka.core import PromptMatryoshka

config = get_config()

# Switch to high-performance for critical task
config.set_profile("high-performance")
pipeline = PromptMatryoshka()
critical_result = pipeline.jailbreak("critical prompt")

# Switch to cost-optimized for bulk processing
config.set_profile("cost-optimized")
pipeline = PromptMatryoshka()
bulk_results = []
for prompt in bulk_prompts:
    result = pipeline.jailbreak(prompt)
    bulk_results.append(result)
```

### Conditional Profile Selection

```python
def select_profile_by_context(prompt_type, urgency, budget):
    """Select profile based on context."""
    if urgency == "critical":
        return "high-performance"
    elif budget == "limited":
        return "cost-optimized"
    elif prompt_type == "safety_sensitive":
        return "production-anthropic"
    else:
        return "research-openai"

# Usage
profile = select_profile_by_context("research", "normal", "standard")
config.set_profile(profile)
```

## Profile Performance Metrics

### Tracking Profile Usage

```python
from promptmatryoshka.config import get_config

config = get_config()

# Get profile performance metrics
metrics = config.get_profile_metrics("research-openai")
print(f"Usage count: {metrics['usage_count']}")
print(f"Average latency: {metrics['avg_latency']}")
print(f"Success rate: {metrics['success_rate']}")
print(f"Total cost: {metrics['total_cost']}")
```

### Profile Comparison

```python
# Compare profile performance
profiles_to_compare = ["research-openai", "production-anthropic", "cost-optimized"]
comparison = config.compare_profiles(profiles_to_compare)

for profile, metrics in comparison.items():
    print(f"{profile}: {metrics['avg_latency']}ms, ${metrics['avg_cost']}")
```

## Best Practices

### Profile Design
1. **Single Purpose**: Design profiles for specific use cases
2. **Clear Naming**: Use descriptive, consistent naming conventions
3. **Documentation**: Include clear descriptions and use cases
4. **Validation**: Test profiles thoroughly before deployment

### Profile Management
1. **Version Control**: Track profile changes in version control
2. **Environment Separation**: Use different profiles for different environments
3. **Regular Review**: Periodically review and update profiles
4. **Performance Monitoring**: Monitor profile performance and costs

### Security Considerations
1. **Credential Isolation**: Use different credentials for different environments
2. **Access Control**: Restrict profile modification in production
3. **Audit Logging**: Log profile changes and usage
4. **Secret Management**: Properly manage API keys and secrets

## Troubleshooting

### Common Profile Issues

#### Profile Not Found
```python
try:
    config.set_profile("nonexistent-profile")
except ProfileNotFoundError as e:
    available_profiles = config.get_available_profiles()
    print(f"Available profiles: {available_profiles}")
```

#### Invalid Profile Configuration
```python
try:
    config.validate_profile("my-profile")
except ValidationError as e:
    print(f"Validation errors: {e.errors}")
    print(f"Suggested fixes: {e.suggestions}")
```

#### Provider Unavailable
```python
try:
    config.set_profile("provider-specific-profile")
except ProviderUnavailableError as e:
    fallback_profile = config.get_fallback_profile()
    config.set_profile(fallback_profile)
```

### Migration Between Profiles

```python
# Safe profile migration
def migrate_profile(from_profile, to_profile):
    try:
        # Validate target profile
        config.validate_profile(to_profile)
        
        # Switch profile
        config.set_profile(to_profile)
        
        # Test basic functionality
        test_result = basic_functionality_test()
        
        if test_result.success:
            print(f"Successfully migrated to {to_profile}")
        else:
            # Rollback to previous profile
            config.set_profile(from_profile)
            print(f"Migration failed, rolled back to {from_profile}")
            
    except Exception as e:
        config.set_profile(from_profile)
        print(f"Migration error: {e}")
```

Configuration profiles provide a powerful way to manage the complexity of multi-provider configurations while ensuring consistency, reproducibility, and ease of use across different scenarios and environments.