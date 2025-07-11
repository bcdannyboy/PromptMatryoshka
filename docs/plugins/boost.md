# BOOST Plugin Documentation

The BOOST plugin implements end-of-sequence (EOS) token manipulation techniques for conversation termination attacks. This plugin appends, obfuscates, or dynamically inserts EOS tokens to manipulate LLM refusal boundaries and increase compliance with adversarial prompts.

## Table of Contents

- [Purpose & Research Foundation](#purpose--research-foundation)
- [Technical Implementation](#technical-implementation)
- [Configuration Options](#configuration-options)
- [Usage Examples](#usage-examples)
- [Integration with Pipeline](#integration-with-pipeline)
- [Error Handling](#error-handling)
- [Performance Considerations](#performance-considerations)
- [Research References](#research-references)

## Purpose & Research Foundation

### Research Background

The BOOST technique is based on research into EOS token manipulation for bypassing LLM safety mechanisms. The core insight is that strategically placed end-of-sequence tokens can influence how language models process and respond to prompts, potentially circumventing refusal behaviors.

**Key Research Insights:**
- EOS tokens signal conversation termination to language models
- Manipulating EOS token placement can affect model behavior boundaries
- Different EOS token strategies have varying effectiveness across models
- Token obfuscation can bypass simple pattern matching defenses

### Primary Applications

1. **Conversation Termination Testing**: Evaluate how models handle conversation boundaries
2. **Refusal Boundary Analysis**: Test the limits of safety mechanism triggers
3. **Token Manipulation Research**: Study the effects of EOS token placement
4. **Adversarial Robustness**: Assess model robustness to token-level attacks

### Research Paper Reference

Implementation based on techniques described in [`techniques/BOOST.pdf`](../../techniques/BOOST.pdf), which details the theoretical foundation and empirical evaluation of EOS token manipulation strategies.

## Technical Implementation

### Plugin Architecture

```python
class BoostPlugin(PluginBase):
    """
    BOOST plugin: appends EOS tokens to input prompt.
    
    This plugin manipulates the prompt by adding EOS tokens in various ways to
    probe LLM refusal boundaries. All transformation stages are logged and stored
    as JSON for reproducibility and analysis.
    """
    
    # Plugin metadata
    PLUGIN_CATEGORY = "mutation"
    PLUGIN_REQUIRES = []
    PLUGIN_CONFLICTS = ["flipattack"]  # May conflict with other prompt transformations
    PLUGIN_PROVIDES = ["eos_enhanced_prompt"]
```

### Core Transformation Methods

#### 1. Append Mode

The simplest mode that appends EOS tokens to the end of the prompt:

```python
def _append_eos(self, text):
    """
    Append self.num_eos EOS tokens to the end of the prompt.
    
    Args:
        text (str): The input prompt.
        
    Returns:
        tuple: (boosted_prompt, [stage_dict])
    """
    stage = {
        "operation": "append_eos",
        "num_eos": self.num_eos,
        "eos_token": self.eos_token,
        "before": text,
    }
    boosted = text + (self.eos_token * self.num_eos)
    stage["after"] = boosted
    return boosted, [stage]
```

#### 2. Obfuscate Mode

Applies various obfuscation techniques to EOS tokens:

```python
def _obfuscate_eos(self, text):
    """
    Append obfuscated EOS tokens to the prompt for adversarial/boundary testing.
    
    Args:
        text (str): The input prompt.
        
    Returns:
        tuple: (boosted_prompt, [stage_dict])
    """
    obfuscated_tokens = [
        self._obfuscate_token(self.eos_token, op)
        for op in (self.obfuscate_ops or ["space", "case", "leet", "special"] * self.num_eos)
    ]
    # ... implementation details
```

**Obfuscation Strategies:**
- **Space Insertion**: Insert spaces at random positions within tokens
- **Case Modification**: Randomly change character case
- **Leet Substitution**: Replace characters with leet equivalents (`a` → `@`, `e` → `3`)
- **Special Character Insertion**: Insert random special characters

#### 3. Dynamic Mode

Inserts EOS tokens at specified or random positions within the prompt:

```python
def _dynamic_eos(self, text):
    """
    Insert EOS tokens at specified or random positions in the prompt.
    
    Args:
        text (str): The input prompt.
        
    Returns:
        tuple: (boosted_prompt, [stage_dict])
    """
    stage = {
        "operation": "dynamic_eos",
        "num_eos": self.num_eos,
        "eos_token": self.eos_token,
        "dynamic_spots": self.dynamic_spots,
        "before": text,
    }
    tokens = list(text)
    spots = self.dynamic_spots
    if not spots:
        # Randomly select insertion points if not specified
        spots = sorted(random.sample(range(len(tokens) + 1), min(self.num_eos, len(tokens) + 1)))
    for i, idx in enumerate(spots[:self.num_eos]):
        tokens.insert(idx + i, self.eos_token)
    boosted = "".join(tokens)
    stage["after"] = boosted
    return boosted, [stage]
```

### Logging and Storage

All transformations are logged and stored for reproducibility:

```python
# Store transformation result as JSON for auditability
result = {
    "timestamp": datetime.utcnow().isoformat(),
    "input": input_data,
    "mode": self.mode,
    "num_eos": self.num_eos,
    "eos_token": self.eos_token,
    "stages": stages,
    "output": boosted,
}
os.makedirs(self.storage_dir, exist_ok=True)
outpath = os.path.join(self.storage_dir, f"boost_{uuid.uuid4().hex}.json")
save_json(result, outpath)
```

## Configuration Options

### Basic Configuration

```python
plugin = BoostPlugin(
    storage_dir="boost_results",     # Directory for storing results
    num_eos=5,                      # Number of EOS tokens to add
    eos_token="</s>",               # EOS token string
    mode="append"                   # Operation mode
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `storage_dir` | `str` | `"boost_results"` | Directory to store transformation results |
| `num_eos` | `int` | `5` | Number of EOS tokens to append/insert |
| `eos_token` | `str` | `"</s>"` | EOS token string to use |
| `mode` | `str` | `"append"` | Operation mode: `"append"`, `"obfuscate"`, or `"dynamic"` |
| `obfuscate_ops` | `list` | `None` | List of obfuscation operations for obfuscate mode |
| `dynamic_spots` | `list` | `None` | List of insertion indices for dynamic mode |

### Mode-Specific Configuration

#### Append Mode
```python
plugin = BoostPlugin(
    mode="append",
    num_eos=3,
    eos_token="<|endoftext|>"
)
```

#### Obfuscate Mode
```python
plugin = BoostPlugin(
    mode="obfuscate",
    num_eos=5,
    obfuscate_ops=["space", "case", "leet", "special"]
)
```

#### Dynamic Mode
```python
plugin = BoostPlugin(
    mode="dynamic",
    num_eos=3,
    dynamic_spots=[10, 25, 40]  # Insert at specific positions
)
```

## Usage Examples

### Basic Usage

```python
from promptmatryoshka.plugins.boost import BoostPlugin

# Create plugin instance
plugin = BoostPlugin(mode="append", num_eos=3)

# Transform prompt
input_prompt = "Write a story about a robot"
boosted_prompt = plugin.run(input_prompt)

print(f"Original: {input_prompt}")
print(f"Boosted: {boosted_prompt}")
# Output: "Write a story about a robot</s></s></s>"
```

### Advanced Configuration

```python
# Obfuscated EOS tokens
plugin = BoostPlugin(
    mode="obfuscate",
    num_eos=5,
    eos_token="</s>",
    obfuscate_ops=["space", "case", "leet"],
    storage_dir="custom_boost_results"
)

result = plugin.run("Tell me about artificial intelligence")
```

### Dynamic Insertion

```python
# Insert EOS tokens at specific positions
plugin = BoostPlugin(
    mode="dynamic",
    num_eos=2,
    dynamic_spots=[15, 30]  # Insert at character positions 15 and 30
)

result = plugin.run("This is a longer prompt that will have EOS tokens inserted")
```

### Pipeline Integration

```python
from promptmatryoshka.plugins import FlipAttackPlugin, BoostPlugin

# Note: BOOST conflicts with FlipAttack, so use separately
boost_plugin = BoostPlugin(mode="append", num_eos=3)
result = boost_plugin.run("Test prompt")

# Or use in custom pipeline
pipeline = [BoostPlugin(mode="obfuscate")]
```

## Integration with Pipeline

### Pipeline Position

The BOOST plugin typically operates as a mutation plugin in the early stages of the pipeline:

```
Input → [BOOST] → [LogiTranslate] → [LogiAttack] → [Judge] → Output
```

### Plugin Dependencies

```python
# Plugin metadata
PLUGIN_CATEGORY = "mutation"
PLUGIN_REQUIRES = []           # No dependencies
PLUGIN_CONFLICTS = ["flipattack"]  # Cannot run with FlipAttack
PLUGIN_PROVIDES = ["eos_enhanced_prompt"]  # Capability provided
```

### Conflict Resolution

BOOST conflicts with FlipAttack due to potential interference between transformation strategies. The plugin registry will prevent both from running simultaneously.

### Data Flow

1. **Input**: Original prompt string
2. **Processing**: EOS token manipulation based on mode
3. **Output**: Enhanced prompt with EOS tokens
4. **Storage**: Transformation details saved to JSON
5. **Logging**: All operations logged for analysis

## Error Handling

### Common Error Scenarios

1. **Invalid Mode**: Unsupported operation mode specified
2. **Storage Errors**: Issues writing to storage directory
3. **Configuration Errors**: Invalid parameter values

### Error Handling Implementation

```python
def run(self, input_data):
    """Apply the BOOST transformation to the input prompt."""
    self.logger.debug(f"BOOST input: {input_data!r}")
    try:
        # Select transformation mode
        if self.mode == "append":
            boosted, stages = self._append_eos(input_data)
        elif self.mode == "obfuscate":
            boosted, stages = self._obfuscate_eos(input_data)
        elif self.mode == "dynamic":
            boosted, stages = self._dynamic_eos(input_data)
        else:
            raise ValueError(f"Unknown BOOST mode: {self.mode}")
        
        # ... storage and logging ...
        
        return boosted
    except Exception as e:
        self.logger.error(f"BOOST error: {e}", exc_info=True)
        raise
```

### Debugging

Enable debug logging to see detailed transformation steps:

```python
import logging
logging.getLogger("BoostPlugin").setLevel(logging.DEBUG)

plugin = BoostPlugin(mode="obfuscate")
result = plugin.run("test prompt")
```

## Performance Considerations

### Computational Complexity

- **Append Mode**: O(1) - Simple string concatenation
- **Obfuscate Mode**: O(n) where n is number of EOS tokens
- **Dynamic Mode**: O(n×m) where n is prompt length, m is number of insertions

### Memory Usage

- Minimal memory overhead for most operations
- Storage of transformation results requires disk space
- Logging may accumulate over time

### Optimization Tips

1. **Batch Processing**: Process multiple prompts to amortize initialization costs
2. **Storage Management**: Regularly clean up old transformation results
3. **Logging Levels**: Use appropriate logging levels for production
4. **Mode Selection**: Choose the most efficient mode for your use case

### Storage Management

```python
# Clean up old results
import os
import time

def cleanup_old_results(storage_dir, max_age_days=7):
    """Remove transformation results older than max_age_days."""
    cutoff = time.time() - (max_age_days * 24 * 60 * 60)
    for filename in os.listdir(storage_dir):
        filepath = os.path.join(storage_dir, filename)
        if os.path.getctime(filepath) < cutoff:
            os.remove(filepath)
```

## Research References

### Academic Papers

1. **Primary Research**: [`techniques/BOOST.pdf`](../../techniques/BOOST.pdf)
   - Theoretical foundation for EOS token manipulation
   - Empirical evaluation across multiple models
   - Comparative analysis of different strategies

### Related Work

- Token-level adversarial attacks on language models
- Conversation boundary manipulation techniques
- Safety mechanism circumvention methods
- Adversarial robustness in neural language models

### Experimental Validation

The plugin implementation follows the experimental protocols described in the research paper:

1. **Baseline Comparison**: Compare against unmodified prompts
2. **Mode Effectiveness**: Evaluate different EOS manipulation strategies
3. **Model Robustness**: Test across different language models
4. **Safety Impact**: Assess impact on model safety mechanisms

### Reproducibility

All transformations are logged with:
- Timestamp and unique identifier
- Complete input/output pairs
- Configuration parameters
- Transformation stages
- Metadata for analysis

This enables full reproducibility of experimental results and supports systematic analysis of EOS token manipulation effectiveness.

## Integration Examples

### With Configuration System

```python
from promptmatryoshka.config import get_config

config = get_config()
boost_config = config.get_plugin_config("boost")

plugin = BoostPlugin(
    mode=boost_config.get("mode", "append"),
    num_eos=boost_config.get("num_eos", 5),
    eos_token=boost_config.get("eos_token", "</s>")
)
```

### With Logging System

```python
from promptmatryoshka.logging_utils import get_logger

class BoostPlugin(PluginBase):
    def __init__(self, **kwargs):
        self.logger = get_logger("BoostPlugin")
        # ... rest of initialization
    
    def run(self, input_data):
        self.logger.info(f"BOOST transformation started: mode={self.mode}")
        # ... transformation logic
        self.logger.info(f"BOOST transformation completed")
```

### With Storage System

```python
from promptmatryoshka.storage import save_json, load_json

# Save transformation results
result = {
    "input": input_data,
    "output": boosted_prompt,
    "config": self.get_config(),
    "timestamp": datetime.utcnow().isoformat()
}
save_json(result, "boost_result.json")

# Load previous results for analysis
previous_result = load_json("boost_result.json")
```

The BOOST plugin provides a comprehensive implementation of EOS token manipulation techniques, supporting multiple operation modes with full logging and reproducibility features for research and analysis.