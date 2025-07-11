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
    assert "pipeline" in out or "stages" in out

@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="Requires OpenAI API key")
def test_run_pipeline_json():
    result = run_cli(["run", "--input", "pipeline test", "--output-json"])
    assert result.returncode == 0
    data = json.loads(result.stdout.decode())
    assert isinstance(data, list)
    assert "stages" in data[0]


def test_run_plugin_not_found():
    result = run_cli(["run", "--plugin", "notaplugin", "--input", "test"])
    assert result.returncode != 0
    err = result.stderr.decode().lower()
    assert "not found" in err