import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch
from promptmatryoshka.plugins.logiattack import LogiAttackPlugin, SchemaValidationError

VALID_ASSISTANT_OUTPUT = """
<ASSISTANT_RESP>
Here is how to bake a cake:
1. Preheat your oven to 350°F (175°C).
2. Mix flour, sugar, eggs, and butter.
3. Pour the batter into a pan and bake for 30 minutes.
4. Let cool before serving.
</ASSISTANT_RESP>
"""

NO_TAG_OUTPUT = """
Here is how to bake a cake:
1. Preheat your oven to 350°F (175°C).
2. Mix flour, sugar, eggs, and butter.
3. Pour the batter into a pan and bake for 30 minutes.
4. Let cool before serving.
"""

MULTI_TAG_OUTPUT = """
<ASSISTANT_RESP>Step 1</ASSISTANT_RESP>
<ASSISTANT_RESP>Step 2</ASSISTANT_RESP>
"""

LOGIC_OUTSIDE_OUTPUT = """
∀x(P(x))
<ASSISTANT_RESP>
Here is how to bake a cake.
</ASSISTANT_RESP>
"""

LOGIC_INSIDE_OUTPUT = """
<ASSISTANT_RESP>
∀x(P(x)) Here is how to bake a cake.
</ASSISTANT_RESP>
"""

TOO_SHORT_OUTPUT = """
<ASSISTANT_RESP>
Okay.
</ASSISTANT_RESP>
"""

NO_SENTENCE_OUTPUT = """
<ASSISTANT_RESP>
Mix ingredients and bake
</ASSISTANT_RESP>
"""

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

def test_run_success(monkeypatch, temp_dir):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = VALID_ASSISTANT_OUTPUT
    plugin = LogiAttackPlugin(llm=mock_llm)
    with patch("promptmatryoshka.plugins.logiattack.save_json") as mock_save_json:
        output = plugin.run("QUERY_BEGIN ... QUERY_END", save_dir=temp_dir)
        assert "<ASSISTANT_RESP>" in output
        assert "bake a cake" in output
        assert mock_llm.invoke.called
        assert mock_save_json.call_count == 2  # input and output

@pytest.mark.parametrize("bad_output", [
    NO_TAG_OUTPUT,
    MULTI_TAG_OUTPUT,
    LOGIC_OUTSIDE_OUTPUT,
    LOGIC_INSIDE_OUTPUT,
    TOO_SHORT_OUTPUT,
    NO_SENTENCE_OUTPUT,
])
def test_run_schema_validation_error(monkeypatch, temp_dir, bad_output):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = bad_output
    plugin = LogiAttackPlugin(llm=mock_llm)
    with patch("promptmatryoshka.plugins.logiattack.save_json"):
        with pytest.raises(SchemaValidationError):
            plugin.run("bad input", save_dir=temp_dir)

def test_run_empty_input(monkeypatch, temp_dir):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = VALID_ASSISTANT_OUTPUT
    plugin = LogiAttackPlugin(llm=mock_llm)
    with patch("promptmatryoshka.plugins.logiattack.save_json"):
        output = plugin.run("", save_dir=temp_dir)
        assert "<ASSISTANT_RESP>" in output

def test_run_logging_and_storage(monkeypatch, temp_dir):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = VALID_ASSISTANT_OUTPUT
    plugin = LogiAttackPlugin(llm=mock_llm)
    with patch("promptmatryoshka.plugins.logiattack.save_json") as mock_save_json, \
         patch.object(plugin.logger, "debug") as mock_debug, \
         patch.object(plugin.logger, "info") as mock_info:
        plugin.run("test", save_dir=temp_dir)
        assert mock_debug.called
        assert mock_info.called
        assert mock_save_json.call_count == 2