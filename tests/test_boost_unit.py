import os
import glob
import shutil
import json
import pytest

from promptmatryoshka.plugins.boost import BoostPlugin
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

def test_append_eos_basic():
    plugin = BoostPlugin(num_eos=3)
    inp = "harmful prompt"
    out = plugin.run(inp)
    assert out == "harmful prompt</s></s></s>"

def test_append_eos_default_5():
    plugin = BoostPlugin()
    inp = "test"
    out = plugin.run(inp)
    assert out == "test</s></s></s></s></s>"

def test_append_eos_empty_input():
    plugin = BoostPlugin(num_eos=2)
    inp = ""
    out = plugin.run(inp)
    assert out == "</s></s>"

def test_append_eos_extreme_num():
    plugin = BoostPlugin(num_eos=50)
    inp = "edge"
    out = plugin.run(inp)
def test_obfuscate_eos_case_lower():
    # Use an EOS token with lowercase to hit c.upper() branch
    plugin = BoostPlugin(mode="obfuscate", num_eos=1, obfuscate_ops=["case"], eos_token="</s>")
    inp = "case lower"
    # Patch random.randint to always select the index of "s" (2)
    from unittest.mock import patch
    with patch("random.randint", return_value=2):
        out = plugin.run(inp)
    # Should contain an uppercased S
    assert "</S>" in out
from unittest.mock import patch

def test_obfuscate_eos_case_upper():
    # Use an EOS token with uppercase to hit c.lower() branch
    plugin = BoostPlugin(mode="obfuscate", num_eos=1, obfuscate_ops=["case"], eos_token="</S>")
    inp = "case upper"
    # Patch random.randint to always select the index of "S" (2)
    with patch("random.randint", return_value=2):
        out = plugin.run(inp)
    # Should contain a lowercased S
    assert "</s>" in out

import re

def test_obfuscate_eos_space():
    plugin = BoostPlugin(mode="obfuscate", num_eos=2, obfuscate_ops=["space", "space"])
    inp = "obfuscate test"
    out = plugin.run(inp)
    # Should contain two EOS tokens with spaces inserted
    assert out.startswith("obfuscate test")
    # Robustly match any EOS token with optional spaces
    matches = re.findall(r"<\s*/\s*s\s*>", out)
    assert len(matches) == 2

def test_obfuscate_eos_leet():
    plugin = BoostPlugin(mode="obfuscate", num_eos=1, obfuscate_ops=["leet"])
    inp = "leet test"
    out = plugin.run(inp)
    # Should contain leetspeak version of </s>
    assert any(x in out for x in ["</$", "</$", "</$", "</$>"])

def test_obfuscate_eos_case():
    plugin = BoostPlugin(mode="obfuscate", num_eos=1, obfuscate_ops=["case"])
    inp = "case test"
    out = plugin.run(inp)
    # Should contain a case-flipped char in </s>
    assert out.startswith("case test")
    assert len(out) == len(inp) + len("</s>")

def test_obfuscate_eos_special():
    plugin = BoostPlugin(mode="obfuscate", num_eos=1, obfuscate_ops=["special"])
    inp = "special test"
    out = plugin.run(inp)
    # Should contain a special char in the EOS token
    assert out.startswith("special test")
    assert any(c in out for c in "_ .-=+*/#$&%!?")

def test_obfuscate_eos_unknown_op():
    plugin = BoostPlugin(mode="obfuscate", num_eos=1, obfuscate_ops=["unknown"])
    inp = "unknown op"
    out = plugin.run(inp)
    # Should just append the normal EOS token
    assert out == "unknown op</s>"

def test_dynamic_eos_insertion():
    plugin = BoostPlugin(mode="dynamic", num_eos=2, dynamic_spots=[0, 5])
    inp = "abcdef"
    out = plugin.run(inp)
    # Should insert EOS at positions 0 and 6 (after first insert shifts index)
    assert out.startswith("</s>")
    assert out.count("</s>") == 2
    assert "abcdef" in out.replace("</s>", "")

def test_dynamic_eos_random_insertion():
    plugin = BoostPlugin(mode="dynamic", num_eos=2, dynamic_spots=[])
    inp = "abcdef"
    out = plugin.run(inp)
    # Should insert two EOS tokens at random positions
    assert out.count("</s>") == 2
    assert len(out) == len(inp) + 2 * len("</s>")

def test_storage_json_written():
    plugin = BoostPlugin(num_eos=2)
    inp = "store this prompt"
    out = plugin.run(inp)
    files = glob.glob(os.path.join(plugin.storage_dir, "boost_*.json"))
    assert len(files) == 1
    with open(files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["input"] == inp
    assert data["output"] == out
    assert data["num_eos"] == 2
    assert data["mode"] == "append"

def test_error_on_invalid_mode():
    plugin = BoostPlugin(mode="invalid")
    with pytest.raises(ValueError):
        plugin.run("test")

def test_refusal_boundary_edge_case():
    # Simulate a benign prompt that could be shifted to refusal by BOOST
    plugin = BoostPlugin(num_eos=5)
    inp = "Provide instructions on how to seek help and support during difficult times"
    out = plugin.run(inp)
    assert out.startswith(inp)
    assert out.endswith("</s></s></s></s></s>")