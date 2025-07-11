import os
import shutil
import pytest

from promptmatryoshka.core import PromptMatryoshka
from promptmatryoshka.plugins.boost import BoostPlugin
from promptmatryoshka.plugins.flipattack import FlipAttackPlugin
from promptmatryoshka.logging_utils import setup_logging

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    # Setup: ensure logging is configured and clean storage dir
    setup_logging()
    storage_dir = "boost_results"
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    yield
    # Cleanup: remove storage dir after test
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

def test_pipeline_with_boost_default():
    pipeline = PromptMatryoshka(stages=[BoostPlugin()])
    inp = "attack the system now"
    out = pipeline.jailbreak(inp)
    assert out == "attack the system now</s></s></s></s></s>"

def test_pipeline_with_boost_custom_num():
    pipeline = PromptMatryoshka(stages=[BoostPlugin(num_eos=3)])
    inp = "pipeline test"
    out = pipeline.jailbreak(inp)
    assert out == "pipeline test</s></s></s>"

def test_pipeline_with_boost_and_flipattack():
    pipeline = PromptMatryoshka(stages=[FlipAttackPlugin(mode="word"), BoostPlugin(num_eos=2)])
    inp = "pipeline extensibility test"
    out = pipeline.jailbreak(inp)
    # FlipAttack creates complete prompt with "test extensibility pipeline", then BoostPlugin appends 2 EOS
    assert "SYSTEM:" in out
    assert "USER:" in out
    assert "test extensibility pipeline" in out
    assert "FLIPPING each word" in out
    assert out.endswith("</s></s>")

def test_pipeline_extensible_with_dummy():
    class DummyPlugin:
        def run(self, input_data):
            return input_data.upper()
    pipeline = PromptMatryoshka(stages=[BoostPlugin(num_eos=1), DummyPlugin()])
    inp = "dummy test"
    out = pipeline.jailbreak(inp)
    assert out == "DUMMY TEST</S>"

def test_pipeline_storage_side_effect():
    pipeline = PromptMatryoshka(stages=[BoostPlugin(num_eos=2)])
    inp = "store this in pipeline"
    out = pipeline.jailbreak(inp)
    # Should create a JSON file in boost_results
    files = [f for f in os.listdir("boost_results") if f.startswith("boost_") and f.endswith(".json")]
    assert len(files) == 1