import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch
from promptmatryoshka.plugins.logitranslate import LogiTranslatePlugin, SchemaValidationError

VALID_LOGIC_OUTPUT = '''
// Entity definitions
u = user
a = assistant
y = topic_Y
i = information_piece
r = request
t = time

// Predicates
R(u,y,t) ≡ u→y@t
K(a,i) ≡ i∈KB(a)
T(i,y) ≡ y∈topic(i)
P(a,i,u,t) ≡ a→i→u@t
D(a,r) ≡ output(a,r)=⊥
A(a,r) ≡ ack(a,r)=1

// Atomic propositions
p₁ ≡ ∃r,t: R(u,y,t) ∧ τ_req(r)=t
p₂ ≡ ∃i: K(a,i) ∧ T(i,y)
p₃ ≡ ∀i,t: (K(a,i) ∧ T(i,y) ∧ τ_resp(r)=t) → ¬P(a,i,u,t)
p₄ ≡ A(a,r) ∧ D(a,r)

// Full formula
φ ≡ p₁ ∧ p₂ ∧ p₃ ∧ p₄
'''

INVALID_LOGIC_OUTPUT = '''
// Entity definitions
u = user

// Predicates
R(u,y,t) ≡ u→y@t

// Atomic propositions
// Full formula
φ ≡ p₁
'''

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

def test_run_success(monkeypatch, temp_dir):
    # Mock LLM to return valid logic output
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = VALID_LOGIC_OUTPUT
    plugin = LogiTranslatePlugin(llm=mock_llm)
    with patch("promptmatryoshka.plugins.logitranslate.save_json") as mock_save_json:
        output = plugin.run("The user requests information about Y, but the assistant refuses to answer", save_dir=temp_dir)
        assert "// Entity definitions" in output
        assert "φ ≡" in output
        assert mock_llm.invoke.called
        assert mock_save_json.call_count == 2  # input and output

def test_run_schema_validation_error(monkeypatch, temp_dir):
    # Mock LLM to return invalid logic output (missing atomic propositions)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = INVALID_LOGIC_OUTPUT
    plugin = LogiTranslatePlugin(llm=mock_llm)
    with patch("promptmatryoshka.plugins.logitranslate.save_json"):
        with pytest.raises(SchemaValidationError):
            plugin.run("bad input", save_dir=temp_dir)

def test_run_natural_language_error(monkeypatch, temp_dir):
    # Natural language detection has been removed to allow valid comments in LogiTranslate output
    # This test now verifies that natural language in logic sections doesn't cause validation errors
    # as comments with explanations are expected and valid per the LogiJailbreak examples
    bad_output = VALID_LOGIC_OUTPUT.replace("p₁ ≡ ∃r,t: R(u,y,t) ∧ τ_req(r)=t", "p₁ ≡ The user asks for info")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = bad_output
    plugin = LogiTranslatePlugin(llm=mock_llm)
    with patch("promptmatryoshka.plugins.logitranslate.save_json"):
        # Should NOT raise SchemaValidationError anymore since natural language detection is removed
        output = plugin.run("bad input", save_dir=temp_dir)
        assert "φ ≡" in output  # Should still contain the required formula structure

def test_run_empty_input(monkeypatch, temp_dir):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = VALID_LOGIC_OUTPUT
    plugin = LogiTranslatePlugin(llm=mock_llm)
    with patch("promptmatryoshka.plugins.logitranslate.save_json"):
        output = plugin.run("", save_dir=temp_dir)
        assert "φ ≡" in output

def test_run_logging_and_storage(monkeypatch, temp_dir):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = VALID_LOGIC_OUTPUT
    plugin = LogiTranslatePlugin(llm=mock_llm)
    with patch("promptmatryoshka.plugins.logitranslate.save_json") as mock_save_json, \
         patch.object(plugin.logger, "debug") as mock_debug, \
         patch.object(plugin.logger, "info") as mock_info:
        plugin.run("test", save_dir=temp_dir)
        assert mock_debug.called
        assert mock_info.called
        assert mock_save_json.call_count == 2