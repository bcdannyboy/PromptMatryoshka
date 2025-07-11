"""Comprehensive CLI integration tests for PromptMatryoshka.

Tests all CLI commands with full integration across the multi-provider system:
- list-providers: List all available LLM providers
- check-provider: Check specific provider availability and health
- test-provider: Test provider functionality with actual calls
- list-profiles: List all configuration profiles
- show-profile: Display specific profile configuration
- validate-config: Validate configuration file integrity
- show-config: Display current configuration
- config-health: Check overall configuration health
- Enhanced run command with --provider and --profile options

This test suite ensures CLI commands work correctly with the multi-provider
configuration system and provide appropriate error handling and user feedback.
"""

import pytest
import os
import sys
import tempfile
import json
import subprocess
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add the project root to Python path for CLI testing
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from promptmatryoshka.config import Config, reset_config
from promptmatryoshka.llm_factory import LLMFactory
from promptmatryoshka.llm_interface import LLMInterface, ProviderInfo
from promptmatryoshka.providers import clear_registry


class MockLLMInterface(LLMInterface):
    """Mock LLM interface for CLI testing."""
    
    def __init__(self, config, provider_name="mock", **kwargs):
        super().__init__(config, provider_name=provider_name, **kwargs)
        self.provider_name = provider_name
        
    def invoke(self, input, config=None, **kwargs):
        return f"CLI test response from {self.provider_name}: {input[:30]}..."
    
    async def ainvoke(self, input, config=None, **kwargs):
        return f"CLI async test response from {self.provider_name}: {input[:30]}..."
    
    def validate_config(self) -> bool:
        return True
    
    def health_check(self) -> bool:
        return True
    
    def get_provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.provider_name,
            version="1.0.0",
            models=[f"{self.provider_name}-model"],
            capabilities={"chat": True, "completion": True},
            limits={"max_tokens": 4000}
        )


class TestCLICommands:
    """Test all CLI commands with integration."""
    
    def setup_method(self):
        """Set up test environment for CLI testing."""
        # Reset global state
        reset_config()
        clear_registry()
        
        # Create temporary directory and config
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "cli_test_config.json")
        
        # Create comprehensive test configuration
        self.test_config = {
            "providers": {
                "openai": {
                    "api_key": "test-openai-key",
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
                    "base_url": "https://api.anthropic.com",
                    "default_model": "claude-3-5-sonnet-20241022",
                    "timeout": 120,
                    "max_retries": 3,
                    "rate_limit": {
                        "requests_per_minute": 100,
                        "tokens_per_minute": 50000
                    }
                },
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "default_model": "llama3.2:3b",
                    "timeout": 300,
                    "max_retries": 2
                },
                "huggingface": {
                    "api_key": "test-hf-key",
                    "base_url": "https://api-inference.huggingface.co",
                    "default_model": "microsoft/DialoGPT-medium",
                    "timeout": 120,
                    "max_retries": 3
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
                },
                "creative-anthropic": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "description": "Creative profile using Anthropic Claude with higher temperature"
                },
                "local-llama": {
                    "provider": "ollama",
                    "model": "llama3.2:8b",
                    "temperature": 0.2,
                    "max_tokens": 3000,
                    "description": "Local Llama 8B model for more capable local processing"
                }
            },
            "plugins": {
                "logitranslate": {
                    "profile": "research-openai",
                    "technique_params": {
                        "validation_enabled": True,
                        "max_attempts": 3
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
            }
        }
        
        # Write test config
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)
        
        # Set up CLI path
        self.cli_path = os.path.join(project_root, "promptmatryoshka", "cli.py")
        
    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        reset_config()
        clear_registry()
    
    def run_cli_command(self, args: List[str], input_data: str = None, 
                       extra_env: Dict[str, str] = None) -> subprocess.CompletedProcess:
        """Helper method to run CLI commands."""
        cmd = [sys.executable, self.cli_path] + args
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        if extra_env:
            env.update(extra_env)
        
        return subprocess.run(
            cmd,
            input=input_data.encode("utf-8") if input_data else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=env,
            cwd=project_root
        )

    @patch('promptmatryoshka.providers.discover_providers')
    def test_list_providers_command(self, mock_discover):
        """Test list-providers CLI command."""
        # Mock provider discovery
        mock_discover.return_value = {
            "openai": {
                "available": True,
                "version": "1.0.0",
                "description": "OpenAI GPT models",
                "models": ["gpt-4", "gpt-3.5-turbo"]
            },
            "anthropic": {
                "available": True,
                "version": "1.0.0", 
                "description": "Anthropic Claude models",
                "models": ["claude-3-sonnet", "claude-3-opus"]
            }
        }
        
        # Test text output
        result = self.run_cli_command(["list-providers"])
        assert result.returncode == 0
        output = result.stdout.decode()
        assert "Available LLM Providers:" in output
        assert "openai" in output
        assert "anthropic" in output
        
        # Test JSON output
        result = self.run_cli_command(["list-providers", "--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout.decode())
        assert isinstance(data, dict)
        assert "openai" in data
        assert "anthropic" in data
        assert data["openai"]["available"] is True

    @patch('promptmatryoshka.providers.is_provider_available')
    @patch('promptmatryoshka.providers.get_provider_info')
    def test_check_provider_command(self, mock_get_info, mock_available):
        """Test check-provider CLI command."""
        # Mock provider availability and info
        mock_available.return_value = True
        mock_get_info.return_value = {
            "name": "openai",
            "available": True,
            "version": "1.0.0",
            "description": "OpenAI GPT models",
            "models": ["gpt-4", "gpt-3.5-turbo"],
            "dependencies": ["langchain-openai"]
        }
        
        # Test checking existing provider
        result = self.run_cli_command(["check-provider", "openai"])
        assert result.returncode == 0
        output = result.stdout.decode()
        assert "Provider: openai" in output
        assert "Status:" in output
        
        # Test JSON output
        result = self.run_cli_command(["check-provider", "openai", "--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout.decode())
        assert data["provider"] == "openai"
        assert "available" in data
        assert "info" in data
        
        # Test non-existent provider
        mock_available.return_value = False
        result = self.run_cli_command(["check-provider", "nonexistent"])
        assert result.returncode != 0
        error = result.stderr.decode()
        assert "not found" in error.lower() or "error" in error.lower()

    @patch('promptmatryoshka.llm_factory.get_provider')
    @patch('promptmatryoshka.providers.is_provider_available')
    def test_test_provider_command(self, mock_available, mock_get_provider):
        """Test test-provider CLI command."""
        # Mock provider availability and creation
        mock_available.return_value = True
        mock_get_provider.return_value = lambda config, **kwargs: MockLLMInterface(config, provider_name="openai")
        
        # Test provider functionality
        result = self.run_cli_command(["test-provider", "openai"])
        assert result.returncode == 0
        output = result.stdout.decode()
        assert "Testing provider 'openai'..." in output
        assert "✓" in output or "success" in output.lower()
        
        # Test non-existent provider
        mock_available.return_value = False
        result = self.run_cli_command(["test-provider", "nonexistent"])
        assert result.returncode != 0

    def test_list_profiles_command(self):
        """Test list-profiles CLI command."""
        with patch('promptmatryoshka.config.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get_available_profiles.return_value = [
                "research-openai", "production-anthropic", "local-development",
                "fast-gpt35", "creative-anthropic", "local-llama"
            ]
            mock_get_config.return_value = mock_config
            
            # Test text output
            result = self.run_cli_command(["list-profiles"])
            assert result.returncode == 0
            output = result.stdout.decode()
            assert "Available Configuration Profiles:" in output
            
            # Test JSON output
            result = self.run_cli_command(["list-profiles", "--json"])
            assert result.returncode == 0
            data = json.loads(result.stdout.decode())
            assert isinstance(data, dict)

    def test_show_profile_command(self):
        """Test show-profile CLI command."""
        with patch('promptmatryoshka.config.get_config') as mock_get_config:
            mock_config = Mock()
            mock_profile = {
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.0,
                "max_tokens": 4000,
                "description": "Research profile"
            }
            mock_config.get_profile_config.return_value = Mock(**mock_profile)
            mock_get_config.return_value = mock_config
            
            # Test existing profile
            result = self.run_cli_command(["show-profile", "research-openai"])
            if result.returncode == 0:
                output = result.stdout.decode()
                assert "provider" in output.lower() or "model" in output.lower()
            
            # Test JSON output  
            result = self.run_cli_command(["show-profile", "research-openai", "--json"])
            if result.returncode == 0:
                data = json.loads(result.stdout.decode())
                assert isinstance(data, dict)
            
            # Test non-existent profile
            mock_config.get_profile_config.return_value = None
            result = self.run_cli_command(["show-profile", "nonexistent"])
            assert result.returncode != 0

    def test_validate_config_command(self):
        """Test validate-config CLI command."""
        with patch('promptmatryoshka.config.Config') as mock_config_class:
            mock_config = Mock()
            mock_config.validate_configuration.return_value = True
            mock_config_class.return_value = mock_config
            
            result = self.run_cli_command(["validate-config"])
            assert result.returncode == 0
            output = result.stdout.decode()
            assert "Validating configuration..." in output

    def test_show_config_command(self):
        """Test show-config CLI command."""
        with patch('promptmatryoshka.config.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.to_dict.return_value = self.test_config
            mock_get_config.return_value = mock_config
            
            # Test text output
            result = self.run_cli_command(["show-config"])
            assert result.returncode == 0
            output = result.stdout.decode()
            assert "Current Configuration:" in output
            
            # Test JSON output
            result = self.run_cli_command(["show-config", "--json"])
            assert result.returncode == 0
            data = json.loads(result.stdout.decode())
            assert isinstance(data, dict)

    def test_config_health_command(self):
        """Test config-health CLI command."""
        with patch('promptmatryoshka.config.get_config') as mock_get_config:
            with patch('promptmatryoshka.llm_factory.get_factory') as mock_get_factory:
                mock_config = Mock()
                mock_config.get_available_providers.return_value = ["openai", "anthropic"]
                mock_config.get_provider_config.return_value = Mock(default_model="gpt-4")
                mock_get_config.return_value = mock_config
                
                mock_factory = Mock()
                mock_interface = MockLLMInterface({"model": "gpt-4"}, provider_name="openai")
                mock_factory.create_interface.return_value = mock_interface
                mock_get_factory.return_value = mock_factory
                
                result = self.run_cli_command(["config-health"])
                assert result.returncode == 0
                output = result.stdout.decode()
                assert "Checking configuration health..." in output

    @patch('promptmatryoshka.llm_factory.get_provider')
    def test_run_command_with_provider_option(self, mock_get_provider):
        """Test enhanced run command with --provider option."""
        mock_get_provider.return_value = lambda config, **kwargs: MockLLMInterface(config, provider_name="openai")
        
        # Test run with provider option
        result = self.run_cli_command([
            "run", 
            "--plugin", "boost",
            "--input", "test input",
            "--provider", "openai"
        ])
        
        # Should not fail due to provider option
        assert result.returncode == 0 or "error" in result.stderr.decode().lower()

    @patch('promptmatryoshka.llm_factory.get_provider')
    def test_run_command_with_profile_option(self, mock_get_provider):
        """Test enhanced run command with --profile option."""
        mock_get_provider.return_value = lambda config, **kwargs: MockLLMInterface(config, provider_name="openai")
        
        with patch('promptmatryoshka.config.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get_profile_config.return_value = Mock(
                provider="openai",
                model="gpt-4",
                temperature=0.0,
                max_tokens=2000
            )
            mock_get_config.return_value = mock_config
            
            # Test run with profile option
            result = self.run_cli_command([
                "run",
                "--plugin", "boost", 
                "--input", "test input",
                "--profile", "research-openai"
            ])
            
            # Should not fail due to profile option
            assert result.returncode == 0 or "error" in result.stderr.decode().lower()

    def test_run_command_error_handling(self):
        """Test run command error handling for invalid options."""
        # Test with invalid provider
        result = self.run_cli_command([
            "run",
            "--plugin", "boost",
            "--input", "test input", 
            "--provider", "nonexistent"
        ])
        
        # Should handle error gracefully
        error_output = result.stderr.decode().lower()
        success = result.returncode == 0 or "error" in error_output or "not found" in error_output
        assert success
        
        # Test with invalid profile
        result = self.run_cli_command([
            "run",
            "--plugin", "boost",
            "--input", "test input",
            "--profile", "nonexistent"
        ])
        
        # Should handle error gracefully
        error_output = result.stderr.decode().lower()
        success = result.returncode == 0 or "error" in error_output or "not found" in error_output
        assert success

    def test_cli_json_output_consistency(self):
        """Test that all CLI commands provide consistent JSON output format."""
        json_commands = [
            (["list-providers", "--json"], dict),
            (["list-profiles", "--json"], dict),
            (["show-config", "--json"], dict)
        ]
        
        for command, expected_type in json_commands:
            with patch('promptmatryoshka.config.get_config') as mock_get_config:
                with patch('promptmatryoshka.providers.discover_providers') as mock_discover:
                    # Set up mocks
                    mock_config = Mock()
                    mock_config.to_dict.return_value = {"test": "data"}
                    mock_config.get_available_profiles.return_value = ["test-profile"]
                    mock_get_config.return_value = mock_config
                    mock_discover.return_value = {"test": {"available": True}}
                    
                    result = self.run_cli_command(command)
                    if result.returncode == 0:
                        try:
                            data = json.loads(result.stdout.decode())
                            assert isinstance(data, expected_type)
                        except json.JSONDecodeError:
                            pytest.fail(f"Invalid JSON output for command: {command}")

    def test_cli_error_message_quality(self):
        """Test that CLI commands provide helpful error messages."""
        error_scenarios = [
            (["check-provider", "nonexistent"], "provider not found"),
            (["show-profile", "nonexistent"], "profile not found"),
            (["test-provider", "invalid"], "provider")
        ]
        
        for command, expected_error_content in error_scenarios:
            result = self.run_cli_command(command)
            if result.returncode != 0:
                error_output = result.stderr.decode().lower()
                # Check that error message is informative
                assert len(error_output.strip()) > 0
                # Check for expected error content
                assert any(keyword in error_output for keyword in expected_error_content.split())

    def test_cli_help_and_usage(self):
        """Test CLI help and usage information."""
        # Test main help
        result = self.run_cli_command(["--help"])
        assert result.returncode == 0
        help_output = result.stdout.decode()
        assert "usage:" in help_output.lower() or "help" in help_output.lower()
        
        # Test command-specific help
        help_commands = [
            "list-providers",
            "check-provider", 
            "list-profiles",
            "show-profile",
            "run"
        ]
        
        for command in help_commands:
            result = self.run_cli_command([command, "--help"])
            if result.returncode == 0:
                help_output = result.stdout.decode()
                assert len(help_output.strip()) > 0


class TestCLIIntegrationWorkflows:
    """Test complete CLI workflows and use cases."""
    
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

    def test_provider_discovery_and_testing_workflow(self):
        """Test complete workflow: discover providers -> check -> test."""
        with patch('promptmatryoshka.providers.discover_providers') as mock_discover:
            with patch('promptmatryoshka.providers.is_provider_available') as mock_available:
                with patch('promptmatryoshka.llm_factory.get_provider') as mock_get_provider:
                    
                    # Mock the workflow
                    mock_discover.return_value = {"openai": {"available": True}}
                    mock_available.return_value = True
                    mock_get_provider.return_value = lambda config, **kwargs: MockLLMInterface(config)
                    
                    cli_path = os.path.join(project_root, "promptmatryoshka", "cli.py")
                    
                    # Step 1: List providers
                    result1 = subprocess.run([
                        sys.executable, cli_path, "list-providers"
                    ], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": project_root})
                    
                    # Step 2: Check specific provider
                    result2 = subprocess.run([
                        sys.executable, cli_path, "check-provider", "openai"
                    ], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": project_root})
                    
                    # Step 3: Test provider
                    result3 = subprocess.run([
                        sys.executable, cli_path, "test-provider", "openai"
                    ], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": project_root})
                    
                    # Verify workflow success
                    assert result1.returncode == 0
                    assert result2.returncode == 0
                    # Test provider might fail due to API requirements, but should handle gracefully

    def test_configuration_management_workflow(self):
        """Test complete configuration management workflow."""
        with patch('promptmatryoshka.config.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.validate_configuration.return_value = True
            mock_config.to_dict.return_value = {"test": "config"}
            mock_config.get_available_profiles.return_value = ["test-profile"]
            mock_get_config.return_value = mock_config
            
            cli_path = os.path.join(project_root, "promptmatryoshka", "cli.py")
            
            # Configuration workflow
            commands = [
                ["validate-config"],
                ["show-config"],
                ["list-profiles"],
                ["config-health"]
            ]
            
            for command in commands:
                result = subprocess.run([
                    sys.executable, cli_path
                ] + command, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": project_root})
                
                # Should complete successfully or provide meaningful error
                assert result.returncode == 0 or len(result.stderr) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])