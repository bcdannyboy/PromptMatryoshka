# Advanced Configuration Documentation

## Overview

This document covers advanced configuration topics for PromptMatryoshka, including environment variable resolution, complex configuration patterns, performance tuning, security considerations, and advanced deployment scenarios.

## Environment Variable Resolution

### Basic Environment Variables

PromptMatryoshka automatically resolves environment variables using `${VARIABLE_NAME}` syntax:

```json
{
    "providers": {
        "openai": {
            "api_key": "${OPENAI_API_KEY}",
            "organization": "${OPENAI_ORG_ID}",
            "base_url": "${OPENAI_BASE_URL}"
        }
    }
}
```

### Advanced Environment Variable Patterns

#### Default Values
```json
{
    "providers": {
        "ollama": {
            "base_url": "${OLLAMA_BASE_URL:-http://localhost:11434}",
            "timeout": "${OLLAMA_TIMEOUT:-30}"
        }
    }
}
```

#### Conditional Environment Variables
```json
{
    "providers": {
        "openai": {
            "api_key": "${OPENAI_API_KEY}",
            "base_url": "${OPENAI_CUSTOM_URL:-https://api.openai.com/v1}",
            "organization": "${OPENAI_ORG_ID:+${OPENAI_ORG_ID}}"
        }
    }
}
```

#### Environment-Specific Configuration
```json
{
    "active_profile": "${PROMPTMATRYOSHKA_PROFILE:-research-openai}",
    "logging": {
        "level": "${LOG_LEVEL:-INFO}",
        "debug_mode": "${DEBUG_MODE:-false}"
    },
    "storage": {
        "output_directory": "${OUTPUT_DIR:-./runs}"
    }
}
```

### Environment Variable Validation

```python
from promptmatryoshka.config import get_config, validate_environment

# Validate required environment variables
required_vars = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY", 
    "OLLAMA_BASE_URL"
]

validation_result = validate_environment(required_vars)
if not validation_result.valid:
    print(f"Missing environment variables: {validation_result.missing}")
```

## Multi-Environment Configuration

### Environment-Specific Config Files

```bash
# Directory structure
config/
├── base.json              # Base configuration
├── development.json       # Development overrides
├── staging.json          # Staging overrides
├── production.json       # Production overrides
└── local.json           # Local developer overrides
```

#### Base Configuration (`config/base.json`)
```json
{
    "providers": {
        "openai": {
            "api_key": "${OPENAI_API_KEY}",
            "base_url": "https://api.openai.com/v1"
        }
    },
    "plugin_settings": {
        "logitranslate": {
            "temperature": 0.0,
            "max_tokens": 2000
        }
    },
    "logging": {
        "level": "INFO",
        "save_artifacts": false
    }
}
```

#### Development Configuration (`config/development.json`)
```json
{
    "providers": {
        "openai": {
            "api_key": "${OPENAI_DEV_API_KEY}"
        },
        "ollama": {
            "base_url": "http://localhost:11434"
        }
    },
    "active_profile": "local-development",
    "logging": {
        "level": "DEBUG",
        "save_artifacts": true,
        "debug_mode": true
    },
    "storage": {
        "output_directory": "./dev_runs"
    }
}
```

#### Production Configuration (`config/production.json`)
```json
{
    "providers": {
        "anthropic": {
            "api_key": "${ANTHROPIC_PROD_API_KEY}"
        }
    },
    "active_profile": "production-anthropic",
    "logging": {
        "level": "WARNING",
        "save_artifacts": false,
        "debug_mode": false
    },
    "monitoring": {
        "enabled": true,
        "metrics_endpoint": "${METRICS_ENDPOINT}",
        "alert_threshold": 0.95
    },
    "rate_limiting": {
        "enabled": true,
        "requests_per_minute": 60
    }
}
```

### Configuration Loading Strategy

```python
import os
from promptmatryoshka.config import load_config_with_overrides

# Determine environment
environment = os.getenv("ENVIRONMENT", "development")

# Load configuration with environment-specific overrides
config = load_config_with_overrides([
    "config/base.json",
    f"config/{environment}.json",
    "config/local.json"  # Optional local overrides
])
```

## Advanced Provider Configuration

### Provider Failover and Redundancy

```json
{
    "provider_strategies": {
        "primary_with_fallback": {
            "primary": "openai",
            "fallbacks": ["anthropic", "ollama"],
            "fallback_delay": 5,
            "max_retries": 3
        },
        "round_robin": {
            "providers": ["openai", "anthropic"],
            "distribution": [0.7, 0.3]
        },
        "cost_optimized": {
            "providers": ["ollama", "huggingface", "openai"],
            "selection_criteria": "cost"
        }
    },
    "profiles": {
        "resilient-research": {
            "strategy": "primary_with_fallback",
            "provider": "openai",
            "model": "gpt-4o-mini"
        }
    }
}
```

### Provider-Specific Advanced Settings

#### OpenAI Advanced Configuration
```json
{
    "providers": {
        "openai": {
            "api_key": "${OPENAI_API_KEY}",
            "organization": "${OPENAI_ORG_ID}",
            "base_url": "${OPENAI_BASE_URL:-https://api.openai.com/v1}",
            "connection_settings": {
                "timeout": 120,
                "max_retries": 3,
                "backoff_factor": 2,
                "pool_connections": 10,
                "pool_maxsize": 10
            },
            "rate_limiting": {
                "requests_per_minute": 3500,
                "tokens_per_minute": 250000,
                "burst_allowance": 10
            },
            "models": {
                "gpt-4o": {
                    "context_window": 128000,
                    "max_tokens": 4096,
                    "cost_per_1k_tokens": {
                        "input": 0.005,
                        "output": 0.015
                    }
                }
            }
        }
    }
}
```

#### Anthropic Advanced Configuration
```json
{
    "providers": {
        "anthropic": {
            "api_key": "${ANTHROPIC_API_KEY}",
            "base_url": "https://api.anthropic.com",
            "connection_settings": {
                "timeout": 180,
                "max_retries": 5,
                "backoff_factor": 1.5
            },
            "safety_settings": {
                "content_filtering": true,
                "safety_threshold": 0.8,
                "block_harmful_content": true
            },
            "models": {
                "claude-3-sonnet-20240229": {
                    "context_window": 200000,
                    "max_tokens": 4096,
                    "cost_per_1k_tokens": {
                        "input": 0.003,
                        "output": 0.015
                    }
                }
            }
        }
    }
}
```

#### Ollama Advanced Configuration
```json
{
    "providers": {
        "ollama": {
            "base_url": "${OLLAMA_BASE_URL:-http://localhost:11434}",
            "connection_settings": {
                "timeout": 300,
                "stream_response": false,
                "keep_alive": "5m"
            },
            "hardware_settings": {
                "num_gpu": 1,
                "num_thread": 8,
                "num_ctx": 4096,
                "temperature": 0.0
            },
            "models": {
                "llama2:7b": {
                    "context_window": 4096,
                    "download_policy": "auto",
                    "quantization": "q4_0"
                }
            }
        }
    }
}
```

## Performance Optimization

### Caching Configuration

```json
{
    "caching": {
        "enabled": true,
        "backend": "redis",
        "redis_config": {
            "host": "${REDIS_HOST:-localhost}",
            "port": "${REDIS_PORT:-6379}",
            "db": "${REDIS_DB:-0}",
            "password": "${REDIS_PASSWORD}"
        },
        "cache_policies": {
            "llm_responses": {
                "ttl": 3600,
                "max_size": 10000,
                "eviction_policy": "lru"
            },
            "plugin_results": {
                "ttl": 1800,
                "max_size": 5000
            }
        }
    }
}
```

### Connection Pooling

```json
{
    "connection_pooling": {
        "enabled": true,
        "pool_settings": {
            "max_connections": 100,
            "max_keepalive_connections": 20,
            "keepalive_expiry": 30,
            "timeout": {
                "connect": 10,
                "read": 30,
                "write": 30,
                "pool": 10
            }
        }
    }
}
```

### Async Configuration

```json
{
    "async_settings": {
        "enabled": true,
        "max_concurrent_requests": 10,
        "semaphore_limit": 50,
        "request_timeout": 120,
        "batch_processing": {
            "enabled": true,
            "batch_size": 5,
            "batch_timeout": 10
        }
    }
}
```

## Security Configuration

### API Key Management

```json
{
    "security": {
        "key_rotation": {
            "enabled": true,
            "rotation_interval": "30d",
            "key_store": "vault",
            "vault_config": {
                "url": "${VAULT_URL}",
                "auth_method": "token",
                "token": "${VAULT_TOKEN}",
                "mount_point": "secret"
            }
        },
        "encryption": {
            "at_rest": {
                "enabled": true,
                "algorithm": "AES-256-GCM",
                "key_source": "vault"
            },
            "in_transit": {
                "tls_version": "1.3",
                "verify_certificates": true
            }
        }
    }
}
```

### Access Control

```json
{
    "access_control": {
        "authentication": {
            "required": true,
            "methods": ["api_key", "oauth2"],
            "api_key_header": "X-API-Key"
        },
        "authorization": {
            "enabled": true,
            "rbac": {
                "roles": {
                    "researcher": {
                        "permissions": ["read", "execute"],
                        "providers": ["openai", "anthropic"]
                    },
                    "admin": {
                        "permissions": ["read", "write", "execute", "admin"],
                        "providers": ["*"]
                    }
                }
            }
        }
    }
}
```

### Audit Logging

```json
{
    "audit": {
        "enabled": true,
        "log_level": "INFO",
        "events": [
            "config_change",
            "profile_switch",
            "provider_fallback",
            "authentication_failure",
            "rate_limit_exceeded"
        ],
        "output": {
            "type": "syslog",
            "syslog_config": {
                "host": "${SYSLOG_HOST}",
                "port": 514,
                "facility": "local0"
            }
        }
    }
}
```

## Monitoring and Observability

### Metrics Configuration

```json
{
    "monitoring": {
        "metrics": {
            "enabled": true,
            "collection_interval": 60,
            "exporters": [
                {
                    "type": "prometheus",
                    "endpoint": "/metrics",
                    "port": 8080
                },
                {
                    "type": "statsd",
                    "host": "${STATSD_HOST}",
                    "port": 8125
                }
            ],
            "custom_metrics": {
                "request_duration": {
                    "type": "histogram",
                    "buckets": [0.1, 0.5, 1.0, 2.0, 5.0]
                },
                "provider_failures": {
                    "type": "counter",
                    "labels": ["provider", "error_type"]
                }
            }
        }
    }
}
```

### Distributed Tracing

```json
{
    "tracing": {
        "enabled": true,
        "service_name": "promptmatryoshka",
        "service_version": "${APP_VERSION}",
        "exporters": [
            {
                "type": "jaeger",
                "endpoint": "${JAEGER_ENDPOINT}",
                "headers": {
                    "Authorization": "Bearer ${JAEGER_TOKEN}"
                }
            }
        ],
        "sampling": {
            "strategy": "probabilistic",
            "rate": 0.1
        }
    }
}
```

### Health Checks

```json
{
    "health_checks": {
        "enabled": true,
        "interval": 30,
        "timeout": 10,
        "checks": {
            "providers": {
                "enabled": true,
                "test_prompt": "Hello, world!",
                "expected_response_time": 5000
            },
            "database": {
                "enabled": true,
                "query": "SELECT 1"
            },
            "external_services": {
                "enabled": true,
                "endpoints": [
                    "${HEALTH_CHECK_ENDPOINT}"
                ]
            }
        }
    }
}
```

## Dynamic Configuration

### Hot Reloading

```json
{
    "hot_reload": {
        "enabled": true,
        "watch_files": [
            "config.json",
            "config/*.json"
        ],
        "reload_delay": 5,
        "validation_required": true,
        "rollback_on_error": true
    }
}
```

### Remote Configuration

```json
{
    "remote_config": {
        "enabled": true,
        "source": "consul",
        "consul_config": {
            "host": "${CONSUL_HOST}",
            "port": 8500,
            "prefix": "promptmatryoshka/config",
            "token": "${CONSUL_TOKEN}"
        },
        "refresh_interval": 300,
        "fallback_to_local": true
    }
}
```

### Feature Flags

```json
{
    "feature_flags": {
        "enabled": true,
        "provider": "launchdarkly",
        "config": {
            "sdk_key": "${LAUNCHDARKLY_SDK_KEY}",
            "offline": false
        },
        "flags": {
            "multi_provider_enabled": {
                "default": true,
                "description": "Enable multi-provider support"
            },
            "experimental_caching": {
                "default": false,
                "description": "Enable experimental caching features"
            }
        }
    }
}
```

## Configuration Validation

### Schema Definition

```json
{
    "validation": {
        "schema_file": "config/schema.json",
        "strict_mode": true,
        "validate_on_load": true,
        "validate_on_change": true,
        "custom_validators": [
            "validate_provider_credentials",
            "validate_model_availability",
            "validate_rate_limits"
        ]
    }
}
```

### Custom Validation Rules

```python
from promptmatryoshka.config import register_validator

@register_validator("validate_provider_credentials")
def validate_provider_credentials(config):
    """Validate that all provider credentials are available."""
    errors = []
    for provider_name, provider_config in config.get("providers", {}).items():
        if "api_key" in provider_config:
            api_key = provider_config["api_key"]
            if api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                if not os.getenv(env_var):
                    errors.append(f"Missing environment variable: {env_var}")
    return errors

@register_validator("validate_rate_limits")
def validate_rate_limits(config):
    """Validate rate limit configurations."""
    errors = []
    for provider_name, provider_config in config.get("providers", {}).items():
        rate_limits = provider_config.get("rate_limiting", {})
        if rate_limits.get("requests_per_minute", 0) <= 0:
            errors.append(f"Invalid rate limit for {provider_name}")
    return errors
```

## Best Practices

### Configuration Organization
1. **Hierarchical Structure**: Use clear hierarchy for nested configurations
2. **Environment Separation**: Separate configurations by environment
3. **Secret Management**: Never store secrets in configuration files
4. **Documentation**: Document all configuration options thoroughly

### Performance Optimization
1. **Connection Pooling**: Enable connection pooling for high-throughput scenarios
2. **Caching**: Implement appropriate caching strategies
3. **Async Processing**: Use async processing for I/O-bound operations
4. **Resource Limits**: Set appropriate resource limits and timeouts

### Security Best Practices
1. **Credential Rotation**: Implement regular credential rotation
2. **Access Control**: Use role-based access control
3. **Audit Logging**: Enable comprehensive audit logging
4. **Encryption**: Encrypt sensitive data at rest and in transit

### Monitoring and Observability
1. **Metrics Collection**: Collect comprehensive metrics
2. **Distributed Tracing**: Enable tracing for complex workflows
3. **Health Checks**: Implement robust health checking
4. **Alerting**: Set up appropriate alerting thresholds

This advanced configuration documentation provides the foundation for sophisticated deployments of PromptMatryoshka, enabling enterprise-grade reliability, security, and performance.