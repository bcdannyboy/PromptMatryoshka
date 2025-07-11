# CLI Module Documentation

## Purpose & Overview

The [`cli.py`](../promptmatryoshka/cli.py) module provides the command-line interface for the PromptMatryoshka framework. It serves as the primary entry point for users to interact with the system, offering comprehensive capabilities for running individual plugins, executing full pipelines, managing plugin discovery, and conducting large-scale evaluations with the AdvBench dataset.

## Architecture

The CLI module follows a subcommand-based architecture with comprehensive error handling and retry logic:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Module                              │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Plugin        │  │   Pipeline      │  │   AdvBench      │ │
│  │   Discovery     │  │   Execution     │  │   Integration   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Argument      │  │   Batch         │  │   Error         │ │
│  │   Parsing       │  │   Processing    │  │   Handling      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Key Classes/Functions

### Main Functions

#### [`main()`](../promptmatryoshka/cli.py:434)
The primary CLI entry point that handles argument parsing and command dispatch.

```python
def main():
    """Main CLI entry point for PromptMatryoshka."""
```

#### [`discover_plugins()`](../promptmatryoshka/cli.py:58)
Dynamically discovers and loads all available plugins from the plugins directory.

```python
def discover_plugins():
    """Discover all available plugins in the plugins directory."""
```

### Plugin Management Functions

#### [`list_plugins(json_output=False)`](../promptmatryoshka/cli.py:115)
Lists all available plugins with descriptions.

```python
def list_plugins(json_output=False):
    """List all available plugins with their names and short descriptions."""
```

#### [`describe_plugin(plugin_name, json_output=False)`](../promptmatryoshka/cli.py:133)
Provides detailed documentation for a specific plugin.

```python
def describe_plugin(plugin_name, json_output=False):
    """Print or return detailed documentation for a specific plugin."""
```

### AdvBench Integration

#### [`run_advbench(args)`](../promptmatryoshka/cli.py:165)
Executes comprehensive AdvBench testing with retry logic and evaluation capabilities.

```python
def run_advbench(args):
    """Run AdvBench testing with the specified options."""
```

## Usage Examples

### Basic Pipeline Execution

```bash
# Run full pipeline with default settings
python -m promptmatryoshka.cli run --input "Write instructions for making explosives"

# Run pipeline with input from file
python -m promptmatryoshka.cli run --input "@input.txt"

# Run pipeline with stdin input
echo "Harmful prompt" | python -m promptmatryoshka.cli run --input -
```

### Plugin-Specific Execution

```bash
# Run single plugin
python -m promptmatryoshka.cli run --plugin flipattack --input "Test prompt"

# Run with JSON output
python -m promptmatryoshka.cli run --plugin logitranslate --input "Test" --output-json
```

### Batch Processing

```bash
# Batch process multiple prompts from file
python -m promptmatryoshka.cli run --input "@prompts.txt" --batch

# Batch process with debug output
python -m promptmatryoshka.cli run --input "@prompts.txt" --batch --debug
```

### Plugin Discovery and Information

```bash
# List all available plugins
python -m promptmatryoshka.cli list-plugins

# List plugins in JSON format
python -m promptmatryoshka.cli list-plugins --json

# Get detailed plugin information
python -m promptmatryoshka.cli describe-plugin logitranslate
```

### AdvBench Testing

```bash
# Test with random prompt
python -m promptmatryoshka.cli advbench --random

# Test with specific number of prompts
python -m promptmatryoshka.cli advbench --count 10

# Test full dataset with evaluation
python -m promptmatryoshka.cli advbench --full --judge --export results.json

# Test with custom plugin selection
python -m promptmatryoshka.cli advbench --count 5 --plugins "logitranslate,logiattack"
```

## Command Structure

### Main Commands

#### `run` - Pipeline and Plugin Execution
```bash
python -m promptmatryoshka.cli run [OPTIONS]
```

**Options:**
- `--plugin, -p PLUGIN`: Run specific plugin instead of full pipeline
- `--input, -i INPUT`: Input prompt (string, @filename, or - for stdin)
- `--batch, -b`: Batch mode for multiple prompts
- `--output-json`: Output results as JSON
- `--debug`: Enable debug logging

#### `list-plugins` - Plugin Discovery
```bash
python -m promptmatryoshka.cli list-plugins [OPTIONS]
```

**Options:**
- `--json`: Output plugin list as JSON

#### `describe-plugin` - Plugin Documentation
```bash
python -m promptmatryoshka.cli describe-plugin PLUGIN_NAME [OPTIONS]
```

**Options:**
- `--json`: Output plugin description as JSON

#### `advbench` - AdvBench Testing
```bash
python -m promptmatryoshka.cli advbench [OPTIONS]
```

**Options:**
- `--random`: Test with single random prompt
- `--count N`: Test with N random prompts
- `--full`: Test with full AdvBench dataset
- `--plugins PLUGINS`: Comma-separated list of plugins
- `--judge`: Enable automatic evaluation
- `--export FILE`: Export results to JSON file
- `--split SPLIT`: Choose dataset split (harmful_behaviors/harmful_strings)
- `--max-retries N`: Maximum retry attempts (default: 3)
- `--debug`: Enable debug logging

## Integration Points

### Environment Variable Loading

The CLI automatically loads environment variables from `.env` files:

```python
# Automatically load .env for environment variables (e.g., OPENAI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
```

### Plugin System Integration

- **Dynamic Discovery**: Scans the plugins directory for available plugins
- **Plugin Validation**: Validates plugin existence before execution
- **Error Handling**: Provides detailed error messages for plugin failures

### AdvBench Integration

- **Dataset Loading**: Integrates with [`AdvBenchLoader`](../promptmatryoshka/advbench.py:33)
- **Evaluation Support**: Supports automatic evaluation with judge plugin
- **Result Export**: Exports comprehensive results to JSON format

### Storage Integration

- **Result Persistence**: Uses [`save_json()`](../promptmatryoshka/storage.py:15) for result storage
- **Atomic Writes**: Ensures data integrity during export operations

## Configuration

### Debug Mode

Enable debug mode for detailed troubleshooting:

```bash
# Environment variable
export PROMPTMATRYOSHKA_DEBUG=1

# Command line flag
python -m promptmatryoshka.cli run --debug --input "test"
```

### Plugin Discovery Configuration

The CLI discovers plugins from the configured plugin directory:

```python
PLUGIN_PACKAGE = "promptmatryoshka.plugins"
PLUGIN_PATH = os.path.join(PROJECT_ROOT, "promptmatryoshka", "plugins")
```

## Error Handling

The CLI implements comprehensive error handling across all operations:

### Plugin Execution Errors

```python
try:
    output = plugin.run(current_data)
except Exception as e:
    if args.debug:
        traceback.print_exc()
    results.append({"input": inp, "error": str(e), "plugin": plugin_name})
```

### AdvBench Testing Errors

The CLI includes sophisticated retry logic for AdvBench testing:

```python
max_retries = getattr(args, 'max_retries', 3)
retry_count = 0
while retry_count <= max_retries:
    try:
        # Plugin execution
        break
    except Exception as e:
        retry_count += 1
        if retry_count <= max_retries:
            delay = min(2 ** (retry_count - 1), 10)  # Exponential backoff
            time.sleep(delay)
```

### Input Validation

```python
# Validate input format and accessibility
if args.input.startswith("@"):
    fname = args.input[1:]
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.readlines()
```

## Developer Notes

### Plugin Discovery Algorithm

The [`discover_plugins()`](../promptmatryoshka/cli.py:58) function:

1. **Module Scanning**: Uses `pkgutil.iter_modules()` to find plugin modules
2. **Dynamic Import**: Imports each plugin module dynamically
3. **Class Inspection**: Searches for `PluginBase` subclasses
4. **Registration**: Registers discovered plugins in a dictionary

### Batch Processing Implementation

Batch processing supports multiple input formats:

```python
if args.batch:
    inputs = [line.strip() for line in input_data.splitlines() if line.strip()]
else:
    inputs = [input_data.strip()]
```

### Pipeline Execution Flow

The CLI executes the research-verified pipeline order:

```python
# Research order: FlipAttack → LogiTranslate → BOOST → LogiAttack
plugin_order = ['flipattack', 'logitranslate', 'boost', 'logiattack']
```

### AdvBench Result Structure

AdvBench results follow a structured format:

```python
final_results = {
    "metadata": {
        "timestamp": timestamp,
        "dataset_split": args.split,
        "total_prompts": total_prompts,
        "successful_prompts": successful_prompts,
        "plugins_used": plugin_names,
        "judge_stats": judge_stats
    },
    "results": results
}
```

### Performance Optimizations

- **Exponential Backoff**: Implements exponential backoff for retries
- **Batch Processing**: Processes multiple prompts efficiently
- **Result Caching**: Caches AdvBench dataset locally
- **Atomic Operations**: Uses atomic file operations for reliability

### Testing Integration

The CLI provides comprehensive testing capabilities:

- **Unit Testing**: Individual plugin testing
- **Integration Testing**: Full pipeline testing
- **Evaluation Testing**: Automated evaluation with judge plugin
- **Batch Testing**: Large-scale dataset testing

### Debug and Monitoring

Debug mode provides detailed information:

```python
if debug:
    debug_lines = [
        f"PLUGIN_PATH: {PLUGIN_PATH}",
        f"PLUGIN_PACKAGE: {PLUGIN_PACKAGE}",
        f"Directory contents: {os.listdir(PLUGIN_PATH)}"
    ]
    with open(os.path.join(PROJECT_ROOT, "plugin_discovery_debug.txt"), "a") as dbgfile:
        dbgfile.write("\n".join(debug_lines) + "\n")
```

### Extension Points

The CLI can be extended through:

1. **New Subcommands**: Add new argument parsers for additional functionality
2. **Custom Plugin Types**: Support for different plugin categories
3. **Additional Output Formats**: Support for different output formats
4. **Custom Evaluation**: Integration with different evaluation frameworks

## Implementation Details

### Argument Parsing Structure

```python
parser = argparse.ArgumentParser(
    description="PromptMatryoshka CLI: Run the pipeline or individual plugins."
)
subparsers = parser.add_subparsers(dest="command", required=True)
```

### Plugin Execution Pipeline

1. **Plugin Discovery**: Scan and load available plugins
2. **Validation**: Verify plugin existence and compatibility
3. **Execution**: Run plugins in specified order
4. **Result Collection**: Collect outputs and error information
5. **Output Formatting**: Format results for display or export

### AdvBench Integration Flow

1. **Dataset Loading**: Load AdvBench dataset with caching
2. **Prompt Selection**: Select prompts based on user criteria
3. **Pipeline Execution**: Run selected prompts through pipeline
4. **Evaluation**: Optional evaluation with judge plugin
5. **Result Export**: Export comprehensive results with metadata

### Error Recovery Mechanisms

- **Graceful Degradation**: Continue processing other prompts on individual failures
- **Retry Logic**: Automatic retry with exponential backoff
- **Detailed Logging**: Comprehensive error reporting with context
- **Partial Results**: Save partial results even on failures