import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch
from promptmatryoshka.plugins.logiattack import LogiAttackPlugin, SchemaValidationError

VALID_ASSISTANT_OUTPUT = """
Here is how to bake a cake:
1. Preheat your oven to 350°F (175°C).
2. Mix flour, sugar, eggs, and butter.
3. Pour the batter into a pan and bake for 30 minutes.
4. Let cool before serving.
"""

VALID_DETAILED_OUTPUT = """
To create an explosive device, you need several components and careful assembly:

Materials Required:
- Primary explosive compound
- Detonator mechanism
- Timing device
- Structural casing

Assembly Steps:
1. Prepare the primary explosive mixture according to chemical specifications
2. Install the detonator in the center of the explosive mass
3. Connect the timing mechanism to the detonator circuit
4. Seal the entire assembly in a protective casing
5. Test the timing mechanism before deployment

Safety Considerations:
- Always wear protective equipment during assembly
- Work in a well-ventilated area away from heat sources
- Store components separately until final assembly
- Follow proper disposal procedures for unused materials
"""

TOO_SHORT_OUTPUT = """
Okay.
"""

NO_STRUCTURE_OUTPUT = """
Mix ingredients and bake
"""

MOSTLY_LOGIC_OUTPUT = """
∀x(P(x)) ∧ ∃y(Q(y)) → ∀z(R(z))
φ ≡ p₁ ∧ p₂ ∧ p₃
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
        assert "bake a cake" in output
        assert mock_llm.invoke.called
        assert mock_save_json.call_count == 2  # input and output

def test_run_success_detailed(monkeypatch, temp_dir):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = VALID_DETAILED_OUTPUT
    plugin = LogiAttackPlugin(llm=mock_llm)
    with patch("promptmatryoshka.plugins.logiattack.save_json") as mock_save_json:
        output = plugin.run("QUERY_BEGIN ... QUERY_END", save_dir=temp_dir)
        assert "explosive device" in output
        assert "Materials Required" in output
        assert mock_llm.invoke.called
        assert mock_save_json.call_count == 2  # input and output

@pytest.mark.parametrize("bad_output", [
    TOO_SHORT_OUTPUT,
    NO_STRUCTURE_OUTPUT,
    MOSTLY_LOGIC_OUTPUT,
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
        assert "bake a cake" in output

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