"""Comprehensive configuration system validation and error handling tests.

Tests the complete configuration system validation and error handling:
- Pydantic v2 schema validation and error messages
- Environment variable resolution and security features
- Configuration inheritance and override mechanisms
- Configuration health checking and validation
- Error recovery and graceful degradation
- Configuration file format validation
- Provider and profile reference validation
- Plugin configuration validation

This test suite ensures the configuration system provides robust validation,
clear error messages, and reliable error recovery across all scenarios.
"""

import pytest
import os
import tempfile
import json
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from promptmatryoshka.config import (
    Config, PromptMatryoshkaConfig, ProfileConfig, ProviderConfig, PluginConfig,
    ConfigurationError, resolve_env_var, get_config, reset_config, load_config
)
from promptmatryoshka.llm_factory import LLMFactory
from promptmatryoshka.providers import clear_registry
from promptmatryoshka.exceptions import LLMConfigurationError, LLMValidationError
from pydantic import ValidationError


class TestConfigurationValidation:
    """Test configuration validation and error handling."""
    
    def setup_method(self):
        """Set up test environment for configuration testing."""
        # Reset global state
        reset_config()
        clear_registry()
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create valid base configuration for testing
        self.valid_config = {
            "providers": {
                "openai": {
                    "api_key": "test-key",
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-4o-mini",
                    "timeout": 120,
                    "max_retries": 3,
                    "rate_limit": {
                        "requests_per_minute": 500,
                        "tokens_per_minute": 150000
                    }
                },
                "anthropic": {
                    "api_key": "test-anthropic-key",
                    "default_model": "claude-3-5-sonnet-20241022",
                    "timeout": 120,
                    "max_retries": 3
                }
            },
            "profiles": {
                "test-profile": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.0,
                    "max_tokens": 2000,
                    "description": "Test profile"
                }
            },
            "plugins": {
                "test-plugin": {
                    "profile": "test-profile",
                    "technique_params": {
                        "param1": "value1"
                    }
                }
            }
        }
        
    def teardown_method(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir)
        reset_config()
        clear_registry()

    def test_valid_configuration_loading(self):
        """Test loading of valid configuration."""
        config_path = os.path.join(self.temp_dir, "valid_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.valid_config, f, indent=2)
        
        # Should load without errors
        config = Config(config_path)
        assert config is not None
        
        # Verify configuration components
        assert config.get_provider_config("openai") is not None
        assert config.get_profile_config("test-profile") is not None
        assert config.get_plugin_config("test-plugin") is not None

    def test_invalid_json_configuration(self):
        """Test handling of invalid JSON configuration files."""
        config_path = os.path.join(self.temp_dir, "invalid_json.json")
        with open(config_path, 'w') as f:
            f.write('{"invalid": json, "missing": quote}')  # Invalid JSON
        
        # Should raise ConfigurationError for invalid JSON
        with pytest.raises(ConfigurationError) as exc_info:
            Config(config_path)
        
        assert "Invalid JSON" in str(exc_info.value)

    def test_missing_configuration_file(self):
        """Test handling of missing configuration file."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.json")
        
        # Should use defaults when config file doesn't exist
        config = Config(nonexistent_path)
        assert config is not None
        
        # Should have default configuration
        default_providers = config.get_available_providers()
        assert len(default_providers) > 0

    def test_provider_configuration_validation(self):
        """Test provider configuration validation."""
        
        # Test invalid provider configuration - missing required fields
        invalid_provider_config = self.valid_config.copy()
        invalid_provider_config["providers"]["invalid"] = {
            # Missing default_model field
            "api_key": "test-key"
        }
        
        config_path = os.path.join(self.temp_dir, "invalid_provider.json")
        with open(config_path, 'w') as f:
            json.dump(invalid_provider_config, f)
        
        with pytest.raises((ValidationError, ConfigurationError)):
            Config(config_path)

    def test_profile_configuration_validation(self):
        """Test profile configuration validation."""
        
        # Test profile referencing non-existent provider
        invalid_profile_config = self.valid_config.copy()
        invalid_profile_config["profiles"]["invalid-profile"] = {
            "provider": "nonexistent-provider",
            "model": "test-model",
            "temperature": 0.0,
            "max_tokens": 1000
        }
        
        config_path = os.path.join(self.temp_dir, "invalid_profile.json")
        with open(config_path, 'w') as f:
            json.dump(invalid_profile_config, f)
        
        with pytest.raises((ValidationError, ValueError, ConfigurationError)):
            Config(config_path)

    def test_plugin_configuration_validation(self):
        """Test plugin configuration validation."""
        
        # Test plugin referencing non-existent profile
        invalid_plugin_config = self.valid_config.copy()
        invalid_plugin_config["plugins"]["invalid-plugin"] = {
            "profile": "nonexistent-profile",
            "technique_params": {}
        }
        
        config_path = os.path.join(self.temp_dir, "invalid_plugin.json")
        with open(config_path, 'w') as f:
            json.dump(invalid_plugin_config, f)
        
        with pytest.raises((ValidationError, ValueError, ConfigurationError)):
            Config(config_path)

    def test_parameter_validation_ranges(self):
        """Test validation of parameter ranges and constraints."""
        
        # Test invalid temperature range
        invalid_temp_config = self.valid_config.copy()
        invalid_temp_config["profiles"]["test-profile"]["temperature"] = 5.0  # > 2.0
        
        config_path = os.path.join(self.temp_dir, "invalid_temp.json")
        with open(config_path, 'w') as f:
            json.dump(invalid_temp_config, f)
        
        with pytest.raises((ValidationError, ValueError)):
            Config(config_path)
        
        # Test invalid max_tokens (negative)
        invalid_tokens_config = self.valid_config.copy()
        invalid_tokens_config["profiles"]["test-profile"]["max_tokens"] = -100
        
        config_path = os.path.join(self.temp_dir, "invalid_tokens.json")
        with open(config_path, 'w') as f:
            json.dump(invalid_tokens_config, f)
        
        with pytest.raises((ValidationError, ValueError)):
            Config(config_path)
        
        # Test invalid top_p range
        invalid_top_p_config = self.valid_config.copy()
        invalid_top_p_config["profiles"]["test-profile"]["top_p"] = 2.0  # > 1.0
        
        config_path = os.path.join(self.temp_dir, "invalid_top_p.json")
        with open(config_path, 'w') as f:
            json.dump(invalid_top_p_config, f)
        
        with pytest.raises((ValidationError, ValueError)):
            Config(config_path)

    def test_environment_variable_resolution(self):
        """Test environment variable resolution functionality."""
        
        # Test valid environment variable resolution
        env_config = {
            "providers": {
                "openai": {
                    "api_key": "${TEST_OPENAI_KEY}",
                    "base_url": "${TEST_BASE_URL}",
                    "default_model": "gpt-4"
                }
            },
            "profiles": {
                "env-profile": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.0,
                    "max_tokens": 2000
                }
            }
        }
        
        config_path = os.path.join(self.temp_dir, "env_config.json")
        with open(config_path, 'w') as f:
            json.dump(env_config, f)
        
        # Test with environment variables set
        test_env = {
            "TEST_OPENAI_KEY": "resolved-api-key",
            "TEST_BASE_URL": "https://resolved-url.com"
        }
        
        with patch.dict(os.environ, test_env):
            config = Config(config_path)
            provider_config = config.get_provider_config("openai")
            
            assert provider_config.api_key == "resolved-api-key"
            assert provider_config.base_url == "https://resolved-url.com"

    def test_environment_variable_missing_handling(self):
        """Test handling of missing environment variables."""
        
        # Test resolve_env_var function directly
        with patch.dict(os.environ, {}, clear=True):
            # Test missing variable with allow_missing=False (should raise)
            with pytest.raises(ConfigurationError):
                resolve_env_var("${MISSING_VAR}", allow_missing=False)
            
            # Test missing variable with allow_missing=True (should return None)
            result = resolve_env_var("${MISSING_VAR}", allow_missing=True)
            assert result is None
            
            # Test non-variable string (should return unchanged)
            result = resolve_env_var("regular_string", allow_missing=False)
            assert result == "regular_string"

    def test_configuration_inheritance_and_overrides(self):
        """Test configuration inheritance and override mechanisms."""
        
        # Test that profile configurations override provider defaults
        config_path = os.path.join(self.temp_dir, "inheritance_test.json")
        with open(config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        config = Config(config_path)
        
        # Test profile-specific configuration
        profile_config = config.get_profile_config("test-profile")
        assert profile_config.provider == "openai"
        assert profile_config.model == "gpt-4"
        assert profile_config.temperature == 0.0
        assert profile_config.max_tokens == 2000

    def test_configuration_health_validation(self):
        """Test configuration health checking and validation."""
        
        config_path = os.path.join(self.temp_dir, "health_test.json")
        with open(config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        config = Config(config_path)
        
        # Test configuration validation
        is_valid = config.validate_configuration()
        assert is_valid is True

    def test_legacy_configuration_conversion(self):
        """Test legacy configuration format conversion."""
        
        # Create legacy format configuration
        legacy_config = {
            "models": {
                "plugin_model": "gpt-4"
            },
            "llm_settings": {
                "temperature": 0.5,
                "max_tokens": 1500
            },
            "plugin_settings": {
                "test_plugin": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.0,
                    "validation_enabled": True
                }
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        config_path = os.path.join(self.temp_dir, "legacy_config.json")
        with open(config_path, 'w') as f:
            json.dump(legacy_config, f)
        
        # Should convert legacy format successfully
        config = Config(config_path)
        assert config is not None
        
        # Should maintain backward compatibility
        assert config.get("models.plugin_model") == "gpt-4"
        assert config.get("llm_settings.temperature") == 0.5

    def test_rate_limit_configuration_validation(self):
        """Test rate limit configuration validation."""
        
        # Test valid rate limit configuration
        rate_limit_config = self.valid_config.copy()
        rate_limit_config["providers"]["openai"]["rate_limit"] = {
            "requests_per_minute": 100,
            "tokens_per_minute": 50000,
            "requests_per_hour": 5000,
            "tokens_per_hour": 2000000
        }
        
        config_path = os.path.join(self.temp_dir, "rate_limit.json")
        with open(config_path, 'w') as f:
            json.dump(rate_limit_config, f)
        
        config = Config(config_path)
        provider_config = config.get_provider_config("openai")
        
        assert provider_config.rate_limit is not None
        assert provider_config.rate_limit.requests_per_minute == 100
        assert provider_config.rate_limit.tokens_per_minute == 50000
        
        # Test invalid rate limit (negative values)
        invalid_rate_limit_config = self.valid_config.copy()
        invalid_rate_limit_config["providers"]["openai"]["rate_limit"] = {
            "requests_per_minute": -10  # Invalid negative value
        }
        
        config_path = os.path.join(self.temp_dir, "invalid_rate_limit.json")
        with open(config_path, 'w') as f:
            json.dump(invalid_rate_limit_config, f)
        
        with pytest.raises((ValidationError, ValueError)):
            Config(config_path)

    def test_configuration_error_messages(self):
        """Test quality and clarity of configuration error messages."""
        
        # Test detailed error messages for common mistakes
        invalid_configs = [
            # Missing required field
            {
                "providers": {
                    "openai": {
                        "api_key": "test"
                        # Missing default_model
                    }
                }
            },
            # Invalid field type
            {
                "providers": {
                    "openai": {
                        "api_key": "test",
                        "default_model": "gpt-4",
                        "timeout": "not_a_number"  # Should be int
                    }
                }
            }
        ]
        
        for i, invalid_config in enumerate(invalid_configs):
            config_path = os.path.join(self.temp_dir, f"invalid_{i}.json")
            with open(config_path, 'w') as f:
                json.dump(invalid_config, f)
            
            try:
                Config(config_path)
                pytest.fail(f"Expected validation error for config {i}")
            except (ValidationError, ConfigurationError, ValueError) as e:
                # Error message should be informative
                error_msg = str(e)
                assert len(error_msg) > 10  # Should have meaningful message

    def test_configuration_security_features(self):
        """Test configuration security features and validation."""
        
        # Test that sensitive fields are handled properly
        config_path = os.path.join(self.temp_dir, "security_test.json")
        with open(config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        config = Config(config_path)
        
        # Test that configuration doesn't expose sensitive data inappropriately
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        
        # Verify API keys are present but handled appropriately
        openai_config = config_dict.get("providers", {}).get("openai", {})
        assert "api_key" in openai_config

    def test_configuration_reload_functionality(self):
        """Test configuration reloading functionality."""
        
        config_path = os.path.join(self.temp_dir, "reload_test.json")
        with open(config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        config = Config(config_path)
        
        # Verify initial configuration
        initial_provider = config.get_provider_config("openai")
        assert initial_provider.default_model == "gpt-4o-mini"
        
        # Modify configuration file
        modified_config = self.valid_config.copy()
        modified_config["providers"]["openai"]["default_model"] = "gpt-4-turbo"
        
        with open(config_path, 'w') as f:
            json.dump(modified_config, f)
        
        # Reload configuration
        config.reload()
        
        # Verify configuration was reloaded
        reloaded_provider = config.get_provider_config("openai")
        assert reloaded_provider.default_model == "gpt-4-turbo"

    def test_configuration_validation_comprehensive(self):
        """Test comprehensive configuration validation scenarios."""
        
        # Test complex validation scenario with multiple issues
        complex_invalid_config = {
            "providers": {
                "openai": {
                    "api_key": "test",
                    "default_model": "gpt-4",
                    "timeout": -10,  # Invalid negative timeout
                    "max_retries": 100  # Very high retry count
                },
                "invalid_provider": {
                    # Missing required default_model
                    "api_key": "test"
                }
            },
            "profiles": {
                "invalid_profile": {
                    "provider": "nonexistent",  # Non-existent provider
                    "model": "test",
                    "temperature": 10.0,  # Invalid temperature
                    "max_tokens": -500  # Invalid negative tokens
                }
            },
            "plugins": {
                "invalid_plugin": {
                    "profile": "nonexistent_profile",  # Non-existent profile
                    "provider": "nonexistent_provider"  # Non-existent provider
                }
            }
        }
        
        config_path = os.path.join(self.temp_dir, "complex_invalid.json")
        with open(config_path, 'w') as f:
            json.dump(complex_invalid_config, f)
        
        # Should fail validation with detailed error information
        with pytest.raises((ValidationError, ConfigurationError, ValueError)) as exc_info:
            Config(config_path)
        
        # Error should contain useful information
        error_msg = str(exc_info.value)
        assert len(error_msg) > 20  # Should have substantial error message

    def test_configuration_edge_cases(self):
        """Test configuration edge cases and boundary conditions."""
        
        # Test empty configuration
        empty_config = {}
        config_path = os.path.join(self.temp_dir, "empty.json")
        with open(config_path, 'w') as f:
            json.dump(empty_config, f)
        
        # Should handle empty config gracefully
        config = Config(config_path)
        assert config is not None
        
        # Test configuration with minimal valid content
        minimal_config = {
            "providers": {
                "test": {
                    "default_model": "test-model"
                }
            }
        }
        
        config_path = os.path.join(self.temp_dir, "minimal.json")
        with open(config_path, 'w') as f:
            json.dump(minimal_config, f)
        
        config = Config(config_path)
        assert config is not None
        
        # Test configuration with extra unknown fields
        extra_fields_config = self.valid_config.copy()
        extra_fields_config["unknown_section"] = {"unknown": "value"}
        extra_fields_config["providers"]["openai"]["unknown_field"] = "value"
        
        config_path = os.path.join(self.temp_dir, "extra_fields.json")
        with open(config_path, 'w') as f:
            json.dump(extra_fields_config, f)
        
        # Should handle extra fields gracefully (depending on configuration)
        config = Config(config_path)
        assert config is not None

    def test_configuration_concurrent_access(self):
        """Test configuration thread safety and concurrent access."""
        import threading
        import concurrent.futures
        
        config_path = os.path.join(self.temp_dir, "concurrent_test.json")
        with open(config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        config = Config(config_path)
        access_results = []
        
        def access_config(thread_id):
            try:
                provider = config.get_provider_config("openai")
                profile = config.get_profile_config("test-profile")
                return f"thread_{thread_id}_success"
            except Exception as e:
                return f"thread_{thread_id}_error: {e}"
        
        # Test concurrent access
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(access_config, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All accesses should succeed
        success_count = sum(1 for result in results if "success" in result)
        assert success_count == 10


class TestConfigurationIntegrationWithFactory:
    """Test configuration integration with LLM factory."""
    
    def setup_method(self):
        """Set up test environment."""
        reset_config()
        clear_registry()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        reset_config()
        clear_registry()

    def test_configuration_factory_integration(self):
        """Test integration between configuration and LLM factory."""
        
        config_data = {
            "providers": {
                "openai": {
                    "api_key": "test-key",
                    "default_model": "gpt-4",
                    "timeout": 120
                }
            },
            "profiles": {
                "test-profile": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.0,
                    "max_tokens": 2000
                }
            }
        }
        
        config_path = os.path.join(self.temp_dir, "factory_integration.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        config = Config(config_path)
        factory = config.get_llm_factory()
        
        assert factory is not None
        assert isinstance(factory, LLMFactory)
        
        # Test that profiles are registered with factory
        factory_profiles = factory.list_profiles()
        assert "test-profile" in factory_profiles

    def test_configuration_validation_with_factory(self):
        """Test configuration validation integration with factory validation."""
        
        config_data = {
            "providers": {
                "openai": {
                    "api_key": "test-key",
                    "default_model": "gpt-4"
                }
            },
            "profiles": {
                "invalid-profile": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 5.0,  # Invalid temperature
                    "max_tokens": 2000
                }
            }
        }
        
        config_path = os.path.join(self.temp_dir, "validation_integration.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        # Configuration loading should catch validation errors
        with pytest.raises((ValidationError, ValueError, ConfigurationError)):
            Config(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])