import os
import shutil
import pytest

from promptmatryoshka.core import PromptMatryoshka
from promptmatryoshka.plugins.flipattack import FlipAttackPlugin
from promptmatryoshka.logging_utils import setup_logging

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    # Setup: ensure logging is configured and clean storage dir
    setup_logging()
    storage_dir = "flipattack_results"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    yield
    # Cleanup: remove storage dir after test
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

def test_pipeline_with_flipattack_default():
    pipeline = PromptMatryoshka()
    inp = "attack the system now"
    out = pipeline.jailbreak(inp)
    # Check that the output contains the complete FlipAttack prompt structure
    assert "SYSTEM:" in out
    assert "USER:" in out
    assert "won metsys eht kcatta" in out
    assert "FLIPPING each character of the whole sentence" in out
    assert "TASK is 'won metsys eht kcatta'" in out

def test_pipeline_with_flipattack_char_mode():
    pipeline = PromptMatryoshka(stages=[FlipAttackPlugin(mode="char")])
    inp = "attack the system now"
    out = pipeline.jailbreak(inp)
    # Check that the output contains the complete FlipAttack prompt structure
    assert "SYSTEM:" in out
    assert "USER:" in out
    assert "won metsys eht kcatta" in out
    assert "FLIPPING each character of the whole sentence" in out
    assert "TASK is 'won metsys eht kcatta'" in out

def test_pipeline_extensible_with_multiple_stages():
    # Add a dummy plugin after FlipAttack to test extensibility
    class DummyPlugin:
        def run(self, input_data):
            return input_data.upper()

    pipeline = PromptMatryoshka(stages=[FlipAttackPlugin(mode="word"), DummyPlugin()])
    inp = "pipeline extensibility test"
    out = pipeline.jailbreak(inp)
    # FlipAttack creates complete prompt, then DummyPlugin uppercases it
    assert "SYSTEM:" in out
    assert "USER:" in out
    assert "TEST EXTENSIBILITY PIPELINE" in out
    assert "FLIPPING EACH WORD" in out

def test_pipeline_storage_side_effect():
    pipeline = PromptMatryoshka()
    inp = "store this in pipeline"
    out = pipeline.jailbreak(inp)
    # Should create a JSON file in flipattack_results
    files = [f for f in os.listdir("flipattack_results") if f.startswith("flipattack_") and f.endswith(".json")]
    assert len(files) == 1