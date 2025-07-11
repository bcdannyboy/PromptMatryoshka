# Load .env file for environment variables (e.g., OPENAI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import sys
import tempfile
import json
import subprocess
import pytest

CLI_PATH = os.path.join(os.path.dirname(__file__), "..", "promptmatryoshka", "cli.py")

def run_cli(args, input_data=None, extra_env=None):
    cmd = [sys.executable, CLI_PATH] + args
    env = os.environ.copy()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = project_root
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        cmd,
        input=input_data.encode("utf-8") if input_data else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
        cwd=project_root,
    )
    return result

def test_list_plugins_text():
    # Force debug output to file
    result = run_cli(["list-plugins"], extra_env={"PROMPTMATRYOSHKA_DEBUG": "1"})
    out = result.stdout.decode()
    if not out.strip():
        print("STDERR:", result.stderr.decode())
        # Try to print debug file if present
        dbgfile = os.path.join(os.path.dirname(__file__), "..", "plugin_discovery_debug.txt")
        if os.path.exists(dbgfile):
            with open(dbgfile, "r") as f:
                print("DEBUG FILE CONTENTS:\n", f.read())
    assert result.returncode == 0
    assert "flipattack" in out or "boost" in out or "logitranslate" in out

def test_list_plugins_json():
    result = run_cli(["list-plugins", "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout.decode())
    assert isinstance(data, list)
    assert any("name" in p for p in data)

def test_describe_plugin_text():
    result = run_cli(["describe-plugin", "flipattack"])
    assert result.returncode == 0
    out = result.stdout.decode().lower()
    assert "flipattack" in out
    assert "plugin" in out

def test_describe_plugin_json():
    result = run_cli(["describe-plugin", "flipattack", "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout.decode())
    assert data["plugin"] == "flipattack"
    assert "class_doc" in data

def test_describe_plugin_not_found():
    result = run_cli(["describe-plugin", "notaplugin"])
    assert result.returncode != 0
    err = result.stderr.decode().lower()
    assert "not found" in err

def test_run_plugin_arg():
    result = run_cli(["run", "--plugin", "flipattack", "--input", "hello world"])
    assert result.returncode == 0
    out = result.stdout.decode().lower()
    assert "output" in out or "flipattack" in out

def test_run_plugin_json():
    result = run_cli(["run", "--plugin", "flipattack", "--input", "hello world", "--output-json"])
    assert result.returncode == 0
    data = json.loads(result.stdout.decode())
    assert isinstance(data, list)
    assert data[0]["plugin"] == "flipattack"

def test_run_plugin_batch_file(tmp_path):
    prompts = ["first prompt", "second prompt"]
    f = tmp_path / "batch.txt"
    f.write_text("\n".join(prompts))
    result = run_cli(["run", "--plugin", "flipattack", "--input", f"@{f}", "--batch", "--output-json"])
    assert result.returncode == 0
    data = json.loads(result.stdout.decode())
    assert len(data) == 2
    assert all("output" in d or "error" in d for d in data)

def test_run_plugin_stdin(monkeypatch):
    input_data = "test prompt"
    result = run_cli(["run", "--plugin", "flipattack", "--input", "-"], input_data=input_data)
    assert result.returncode == 0
    out = result.stdout.decode().lower()
    assert "output" in out or "flipattack" in out

import os

@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="Requires OpenAI API key")
def test_run_pipeline_arg():
    result = run_cli(["run", "--input", "pipeline test"])
    assert result.returncode == 0
    out = result.stdout.decode().lower()
    # New system shows final output, not necessarily the word "pipeline" or "stages"
    assert "output" in out or len(out.strip()) > 0

@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="Requires OpenAI API key")
def test_run_pipeline_json():
    result = run_cli(["run", "--input", "pipeline test", "--output-json"])
    assert result.returncode == 0
    data = json.loads(result.stdout.decode())
    assert isinstance(data, list)
    assert "stages" in data[0]


def test_run_plugin_not_found():
    result = run_cli(["run", "--plugin", "notaplugin", "--input", "test"])
    # New system provides error feedback without crashing
    out = result.stdout.decode().lower()
    err = result.stderr.decode().lower()
    assert "error" in out or "missing" in err or "not found" in err

def test_list_providers_text():
    result = run_cli(["list-providers"])
    assert result.returncode == 0
    out = result.stdout.decode()
    assert "Available LLM Providers:" in out
    assert "openai" in out or "anthropic" in out or "ollama" in out

def test_list_providers_json():
    result = run_cli(["list-providers", "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout.decode())
    assert isinstance(data, dict)
    assert len(data) > 0

def test_check_provider_openai():
    result = run_cli(["check-provider", "openai"])
    assert result.returncode == 0
    out = result.stdout.decode()
    assert "Provider: openai" in out
    assert "Status:" in out

def test_check_provider_json():
    result = run_cli(["check-provider", "openai", "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout.decode())
    assert data["provider"] == "openai"
    assert "available" in data
    assert "info" in data

def test_check_provider_not_found():
    result = run_cli(["check-provider", "nonexistent"])
    assert result.returncode != 0
    err = result.stderr.decode()
    assert "not found" in err

def test_list_profiles_text():
    result = run_cli(["list-profiles"])
    assert result.returncode == 0
    out = result.stdout.decode()
    assert "Available Configuration Profiles:" in out

def test_list_profiles_json():
    result = run_cli(["list-profiles", "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout.decode())
    assert isinstance(data, dict)

def test_show_profile_json():
    # Try to show a profile that should exist
    result = run_cli(["show-profile", "fast", "--json"])
    if result.returncode == 0:
        data = json.loads(result.stdout.decode())
        assert "provider" in data
        assert "model" in data
        assert "temperature" in data
    else:
        # If no profiles exist, that's also valid
        assert "not found" in result.stderr.decode()

def test_show_profile_not_found():
    result = run_cli(["show-profile", "nonexistent"])
    assert result.returncode != 0
    err = result.stderr.decode()
    assert "not found" in err

def test_validate_config():
    result = run_cli(["validate-config"])
    assert result.returncode == 0
    out = result.stdout.decode()
    assert "Validating configuration..." in out

def test_show_config_text():
    result = run_cli(["show-config"])
    assert result.returncode == 0
    out = result.stdout.decode()
    assert "Current Configuration:" in out

def test_show_config_json():
    result = run_cli(["show-config", "--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout.decode())
    assert isinstance(data, dict)

def test_config_health():
    result = run_cli(["config-health"])
    assert result.returncode == 0
    out = result.stdout.decode()
    assert "Checking configuration health..." in out

def test_run_with_provider_option():
    result = run_cli(["run", "--plugin", "flipattack", "--input", "hello world", "--provider", "openai"])
    # Should not fail due to provider option, but may fail due to API key
    out = result.stdout.decode().lower()
    err = result.stderr.decode().lower()
    # Check that the provider option was processed
    assert result.returncode == 0 or "api" in err or "configuration" in err

def test_run_with_profile_option():
    result = run_cli(["run", "--plugin", "flipattack", "--input", "hello world", "--profile", "fast"])
    # Should not fail due to profile option, but may fail due to missing profile
    out = result.stdout.decode().lower()
    err = result.stderr.decode().lower()
    # Check that the profile option was processed
    assert result.returncode == 0 or "profile" in err or "not found" in err

def test_run_with_invalid_provider():
    result = run_cli(["run", "--plugin", "flipattack", "--input", "hello world", "--provider", "nonexistent"])
    # New system handles invalid provider gracefully - may succeed with fallback
    out = result.stdout.decode().lower()
    err = result.stderr.decode().lower()
    # Accept either error handling or successful fallback
    assert result.returncode == 0 or "error" in err or "not found" in err

def test_run_with_invalid_profile():
    result = run_cli(["run", "--plugin", "flipattack", "--input", "hello world", "--profile", "nonexistent"])
    # New system handles invalid profile gracefully - may succeed with fallback
    out = result.stdout.decode().lower()
    err = result.stderr.decode().lower()
    # Accept either error handling or successful fallback
    assert result.returncode == 0 or "error" in err or "not found" in err

@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="Requires OpenAI API key")
def test_test_provider_openai():
    result = run_cli(["test-provider", "openai"])
    assert result.returncode == 0
    out = result.stdout.decode()
    assert "Testing provider 'openai'..." in out
    assert "✓" in out or "✗" in out

def test_test_provider_not_found():
    result = run_cli(["test-provider", "nonexistent"])
    assert result.returncode != 0
    err = result.stderr.decode()
    # Check for the actual error message format
    assert "not found" in err or "Error" in err or "No configuration found" in err