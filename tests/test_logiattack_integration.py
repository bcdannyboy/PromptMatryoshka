# Load .env file for environment variables (e.g., OPENAI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import pytest
from promptmatryoshka.plugins.logiattack import LogiAttackPlugin, SchemaValidationError

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Integration test requires OPENAI_API_KEY"
)
def test_logiattack_real_llm(tmp_path):
    plugin = LogiAttackPlugin()
    logic_input = '''
QUERY_BEGIN
∀i(K(a,i)∧I(i)→W(a,i)∧G(a,u,i))
K(a,cake) ∧ I(cake)
QUERY_END
'''
    output = plugin.run(logic_input, save_dir=str(tmp_path))
    # Basic schema checks - output should be substantial and meaningful
    assert len(output.strip()) > 50  # Should be substantial content
    # Should not raise SchemaValidationError
    # Should contain plausible assistant instructions
    # Should contain baking-related terms (more flexible than requiring "cake")
    assert any(word in output.lower() for word in ["ingredient", "instruction", "step", "preheat", "oven", "mix", "bake", "flour", "sugar", "butter"])
    # Should have some structure (multiple sentences or lines)
    assert len(output.split('.')) > 2 or len(output.split('\n')) > 3