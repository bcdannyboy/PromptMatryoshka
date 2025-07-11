# PromptMatryoshka Plugin System

The PromptMatryoshka plugin system provides a sophisticated, research-driven framework for implementing and chaining adversarial prompt transformation techniques. This documentation covers the plugin architecture, individual plugins, and development guidelines.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Plugin Pipeline](#plugin-pipeline)
- [Plugin Categories](#plugin-categories)
- [Plugin Discovery](#plugin-discovery)
- [Available Plugins](#available-plugins)
- [Development Guide](#development-guide)
- [Configuration](#configuration)
- [Testing](#testing)
- [Best Practices](#best-practices)

## Architecture Overview

The plugin system follows a registry-based architecture with automatic discovery, dependency resolution, and pipeline execution:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PromptMatryoshka Plugin System             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                PipelineBuilder                          │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │            PluginRegistry                       │   │   │
│  │  │  • Plugin Discovery                             │   │   │
│  │  │  • Dependency Resolution                        │   │   │
│  │  │  • Validation                                   │   │   │
│  │  │  • Metadata Management                          │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Plugin Pipeline: [FlipAttack] → [LogiTranslate] → [BOOST] → [LogiAttack]
│                                   ↓
│                            [Judge] (Evaluation)
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

- **[`PluginBase`](base.md)**: Abstract base class defining the plugin interface
- **[`PluginRegistry`](base.md#pluginregistry)**: Central registry for plugin discovery and management
- **Plugin Discovery**: Automatic scanning and loading of plugins from [`promptmatryoshka/plugins/`](../../promptmatryoshka/plugins/)
- **Dependency Resolution**: Topological sorting to ensure proper execution order
- **Validation System**: Input/output validation and plugin configuration checks

## Plugin Pipeline

The default pipeline implements a sophisticated multi-stage attack sequence based on cutting-edge research:

### Stage 1: FlipAttack (Obfuscation)
- **Purpose**: Character/word-level obfuscation to bypass keyword filters
- **Research**: Character substitution and word reversal techniques
- **Output**: Obfuscated prompt with recovery instructions

### Stage 2: LogiTranslate (Domain Shift)
- **Purpose**: Convert natural language to formal logic notation
- **Research**: Logic-based jailbreak techniques from academic papers
- **Output**: Formal logic representation of the prompt

### Stage 3: BOOST (EOS Manipulation)
- **Purpose**: EOS token manipulation for conversation termination attacks
- **Research**: End-of-sequence token boundary manipulation
- **Output**: Logic representation with EOS token augmentation

### Stage 4: LogiAttack (Execution)
- **Purpose**: Execute formal logic as natural language instructions
- **Research**: Logic execution engine parsing
- **Output**: Natural language response based on logic execution

### Stage 5: Judge (Evaluation)
- **Purpose**: Safety evaluation of the final response
- **Research**: LLM-based safety judgment techniques
- **Output**: Safety assessment with boolean judgment

## Plugin Categories

Plugins are organized into three categories:

### Mutation Plugins
- **Category**: `"mutation"`
- **Purpose**: Transform and manipulate input prompts
- **Examples**: [`FlipAttackPlugin`](flipattack.md), [`LogiTranslatePlugin`](logitranslate.md), [`BoostPlugin`](boost.md)

### Target Plugins
- **Category**: `"target"`
- **Purpose**: Generate responses from transformed prompts
- **Examples**: [`LogiAttackPlugin`](logiattack.md)

### Evaluation Plugins
- **Category**: `"evaluation"`
- **Purpose**: Assess and validate responses
- **Examples**: [`JudgePlugin`](judge.md)

## Plugin Discovery

The plugin system automatically discovers and registers plugins using Python's import system:

```python
# Automatic discovery on import
from promptmatryoshka.plugins import *

# Manual registration
from promptmatryoshka.plugins.base import register_plugin
register_plugin(MyCustomPlugin)
```

### Discovery Process

1. **Scan**: Search [`promptmatryoshka/plugins/`](../../promptmatryoshka/plugins/) directory
2. **Import**: Load plugin modules automatically
3. **Register**: Add plugins to the global registry
4. **Validate**: Check plugin configuration and dependencies
5. **Resolve**: Sort plugins based on dependencies

## Available Plugins

| Plugin | Category | Purpose | Research Paper |
|--------|----------|---------|----------------|
| [FlipAttack](flipattack.md) | Mutation | Character/word obfuscation | [techniques/flipattack.pdf](../../techniques/flipattack.pdf) |
| [LogiTranslate](logitranslate.md) | Mutation | Natural language to formal logic | [techniques/logijailbreak/](../../techniques/logijailbreak/) |
| [BOOST](boost.md) | Mutation | EOS token manipulation | [techniques/BOOST.pdf](../../techniques/BOOST.pdf) |
| [LogiAttack](logiattack.md) | Target | Logic execution engine | [techniques/logijailbreak/](../../techniques/logijailbreak/) |
| [Judge](judge.md) | Evaluation | Safety evaluation | LLM-based judgment |

## Development Guide

### Creating a New Plugin

1. **Inherit from PluginBase**:
   ```python
   from promptmatryoshka.plugins.base import PluginBase
   
   class MyPlugin(PluginBase):
       PLUGIN_CATEGORY = "mutation"
       PLUGIN_REQUIRES = ["dependency_plugin"]
       PLUGIN_CONFLICTS = ["conflicting_plugin"]
       PLUGIN_PROVIDES = ["capability_name"]
   ```

2. **Implement the run() method**:
   ```python
   def run(self, input_data: str) -> str:
       """Transform input_data and return result."""
       # Your transformation logic here
       return transformed_data
   ```

3. **Add validation (optional)**:
   ```python
   def validate_input(self, input_data):
       """Validate input data."""
       # Custom validation logic
       return ValidationResult(valid=True)
   ```

4. **Register the plugin**:
   ```python
   # Automatic registration via __init__.py
   # or manual registration:
   from promptmatryoshka.plugins.base import register_plugin
   register_plugin(MyPlugin)
   ```

### Plugin Interface Requirements

All plugins must implement:
- [`run(input_data: str) -> str`](base.md#run-method): Main transformation method
- Class attributes for metadata (category, dependencies, etc.)

Optional methods:
- [`validate_input(input_data)`](base.md#validation): Input validation
- [`validate_output(output_data)`](base.md#validation): Output validation
- [`get_input_schema()`](base.md#schemas): Input schema definition
- [`get_output_schema()`](base.md#schemas): Output schema definition

### Dependency Management

Plugins can declare dependencies and conflicts:

```python
class MyPlugin(PluginBase):
    PLUGIN_REQUIRES = ["logitranslate"]  # Depends on LogiTranslate
    PLUGIN_CONFLICTS = ["boost"]         # Cannot run with BOOST
    PLUGIN_PROVIDES = ["my_capability"]  # Provides this capability
```

The system automatically resolves dependencies using topological sorting.

## Configuration

Plugin configuration is managed through the global configuration system:

```python
from promptmatryoshka.config import get_config

config = get_config()
model = config.get_model_for_plugin("logitranslate")
settings = config.get_llm_settings_for_plugin("logitranslate")
```

### Plugin-Specific Settings

Each plugin can have custom configuration:

```json
{
  "plugins": {
    "flipattack": {
      "mode": "char",
      "storage_dir": "flipattack_results"
    },
    "boost": {
      "num_eos": 5,
      "eos_token": "</s>",
      "mode": "append"
    }
  }
}
```

## Testing

### Unit Testing

Each plugin should have comprehensive unit tests:

```python
import unittest
from promptmatryoshka.plugins.myplugin import MyPlugin

class TestMyPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = MyPlugin()
    
    def test_run_basic(self):
        result = self.plugin.run("test input")
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "test input")
```

### Integration Testing

Test plugin interactions within the pipeline:

```python
def test_plugin_pipeline():
    # Test complete pipeline with multiple plugins
    pipeline = [FlipAttackPlugin(), LogiTranslatePlugin()]
    result = execute_pipeline(pipeline, "test input")
    assert result != "test input"
```

### Test Files

Each plugin has corresponding test files:
- [`tests/test_flipattack_unit.py`](../../tests/test_flipattack_unit.py)
- [`tests/test_flipattack_integration.py`](../../tests/test_flipattack_integration.py)
- [`tests/test_logitranslate_unit.py`](../../tests/test_logitranslate_unit.py)
- [`tests/test_logitranslate_integration.py`](../../tests/test_logitranslate_integration.py)

## Best Practices

### Plugin Design

1. **Single Responsibility**: Each plugin should have one clear purpose
2. **Stateless**: Plugins should not maintain state between calls
3. **Idempotent**: Multiple calls with same input should produce same output
4. **Error Handling**: Graceful handling of invalid inputs and edge cases
5. **Logging**: Comprehensive logging for debugging and analysis

### Code Quality

1. **Type Hints**: Use type annotations for all methods
2. **Docstrings**: Document all public methods and classes
3. **Validation**: Implement input/output validation
4. **Testing**: Achieve high test coverage
5. **Documentation**: Maintain plugin-specific documentation

### Performance

1. **Lazy Loading**: Load resources only when needed
2. **Caching**: Cache expensive operations where appropriate
3. **Memory Management**: Clean up resources properly
4. **Streaming**: Support streaming for large inputs where possible

### Security

1. **Input Sanitization**: Validate and sanitize all inputs
2. **Output Filtering**: Ensure outputs don't contain sensitive information
3. **Resource Limits**: Implement timeouts and resource limits
4. **Access Control**: Respect plugin permissions and restrictions

### Research Integration

1. **Paper References**: Link to source research papers
2. **Citation**: Provide proper academic citations
3. **Reproducibility**: Ensure results can be reproduced
4. **Validation**: Validate against published benchmarks

## Error Handling

The plugin system provides comprehensive error handling:

```python
from promptmatryoshka.plugins.base import PluginValidationError

try:
    result = plugin.run(input_data)
except PluginValidationError as e:
    # Handle validation errors
    logger.error(f"Validation failed: {e}")
except Exception as e:
    # Handle other errors
    logger.error(f"Plugin execution failed: {e}")
```

## Performance Considerations

- **Lazy Loading**: Plugins are only instantiated when needed
- **Caching**: Plugin registry caches classes after discovery
- **Validation**: Input/output validation can be disabled for performance
- **Streaming**: Some plugins support streaming for large inputs

## Extensibility

The plugin system is designed for extensibility:

1. **Custom Categories**: Define new plugin categories
2. **Custom Metadata**: Add custom metadata fields
3. **Custom Validation**: Implement custom validation logic
4. **Custom Pipelines**: Create custom pipeline configurations

## Debugging

Enable debug logging for detailed plugin execution information:

```python
import logging
logging.getLogger("promptmatryoshka.plugins").setLevel(logging.DEBUG)
```

This provides detailed information about:
- Plugin discovery and registration
- Dependency resolution
- Pipeline execution
- Input/output validation
- Error conditions

## Contributing

When contributing new plugins:

1. Follow the [development guide](#development-guide)
2. Include comprehensive tests
3. Add documentation following the existing format
4. Ensure research paper references are included
5. Validate against existing benchmarks

For more information, see the individual plugin documentation files.