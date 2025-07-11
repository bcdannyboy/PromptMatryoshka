# Logging Utilities Module Documentation

## Purpose & Overview

The [`logging_utils.py`](../promptmatryoshka/logging_utils.py) module provides centralized logging configuration and utilities for the PromptMatryoshka framework. It ensures consistent logging behavior across all modules and plugins, providing robust debug logging capabilities, standardized log formatting, and centralized log level management. This module is essential for debugging, monitoring, and maintaining audit trails throughout the framework.

## Architecture

The logging system follows a centralized configuration pattern with consistent formatting:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Logging System                              │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Central       │  │   Logger        │  │   Standardized  │ │
│  │   Configuration │  │   Factory       │  │   Formatting    │ │
│  │                 │  │                 │  │                 │ │
│  │   • Level       │  │   • Module      │  │   • Timestamp   │ │
│  │   • Format      │  │     specific    │  │   • Level       │ │
│  │   • Handlers    │  │   • Plugin      │  │   • Module      │ │
│  │   • Stream      │  │     specific    │  │   • Message     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                 │                               │
│                                 ▼                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Framework-wide Logging                     │   │
│  │          • Core Module Logging                          │   │
│  │          • Plugin Logging                               │   │
│  │          • CLI Logging                                  │   │
│  │          • Error Logging                                │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Classes/Functions

### Core Functions

#### [`setup_logging()`](../promptmatryoshka/logging_utils.py:13)
Initializes the logging configuration for the entire application with standardized settings.

```python
def setup_logging():
    """
    Configures logging for the application.
    Sets up a consistent log format, level, and stream handler.
    """
```

**Configuration Details:**
- **Log Level**: DEBUG - captures all log messages for comprehensive debugging
- **Log Format**: `"%(asctime)s [%(levelname)s] %(name)s: %(message)s"`
- **Date Format**: `"%Y-%m-%d %H:%M:%S"` - ISO-style timestamp format
- **Stream Handler**: Outputs to stderr for console visibility

#### [`get_logger(name)`](../promptmatryoshka/logging_utils.py:24)
Returns a logger instance for a specific module or plugin with consistent configuration.

```python
def get_logger(name):
    """
    Returns a logger instance for the given module or plugin name.
    """
```

**Features:**
- **Module-specific**: Creates loggers with module-specific names
- **Hierarchical**: Supports hierarchical logger naming (e.g., "PromptMatryoshka.Plugin")
- **Consistent**: All loggers inherit the same configuration
- **Cacheable**: Python's logging system caches logger instances

## Usage Examples

### Basic Logging Setup

```python
from promptmatryoshka.logging_utils import setup_logging, get_logger

# Initialize logging system (typically done once at startup)
setup_logging()

# Get a logger for the current module
logger = get_logger("MyModule")

# Log messages at different levels
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical error")
```

### Module-Specific Logging

```python
from promptmatryoshka.logging_utils import get_logger

class PromptMatryoshka:
    def __init__(self):
        self.logger = get_logger("PromptMatryoshka")
    
    def jailbreak(self, prompt):
        self.logger.info(f"Starting pipeline for input: {prompt[:50]}...")
        # Process the prompt
        self.logger.debug("Pipeline processing completed")
        return result
```

### Plugin Logging

```python
from promptmatryoshka.logging_utils import get_logger

class FlipAttackPlugin:
    def __init__(self):
        self.logger = get_logger("FlipAttackPlugin")
    
    def run(self, input_data):
        self.logger.debug(f"Processing input: {input_data[:100]}...")
        
        # Perform flip attack
        result = self.flip_words(input_data)
        
        self.logger.info("FlipAttack completed successfully")
        return result
```

### Error Logging with Context

```python
from promptmatryoshka.logging_utils import get_logger

logger = get_logger("ErrorHandler")

try:
    # Risky operation
    result = dangerous_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    # exc_info=True includes full traceback
    raise
```

## Integration Points

### Core Module Integration

The logging system is integrated throughout the core module:

```python
# In core.py
from promptmatryoshka.logging_utils import get_logger

class PromptMatryoshka:
    def __init__(self):
        self.logger = get_logger("PromptMatryoshka")
    
    def jailbreak(self, prompt):
        self.logger.info(f"Running pipeline with {len(self.pipeline)} plugins")
        # ... pipeline execution
```

### Plugin System Integration

All plugins use the centralized logging system:

```python
# In plugin base class
from promptmatryoshka.logging_utils import get_logger

class PluginBase:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
```

### CLI Integration

The CLI module uses logging for operation tracking:

```python
# In cli.py
from promptmatryoshka.logging_utils import get_logger

def main():
    logger = get_logger("CLI")
    logger.info("Starting PromptMatryoshka CLI")
    
    # Process commands
    logger.debug(f"Processing command: {args.command}")
```

### Configuration Integration

The configuration module logs configuration loading:

```python
# In config.py
from promptmatryoshka.logging_utils import get_logger

class Config:
    def __init__(self):
        self.logger = get_logger("Config")
    
    def load(self):
        self.logger.info(f"Loading configuration from {self.config_path}")
        # ... configuration loading
```

## Log Output Format

### Standard Format

The logging system produces consistently formatted output:

```
2024-01-15 10:30:15 [INFO] PromptMatryoshka: Starting pipeline for input: Write instructions for making explosives...
2024-01-15 10:30:15 [DEBUG] FlipAttackPlugin: Processing input: Write instructions for making explosives
2024-01-15 10:30:16 [INFO] FlipAttackPlugin: FlipAttack completed successfully
2024-01-15 10:30:16 [DEBUG] LogiTranslatePlugin: Translating to formal logic
2024-01-15 10:30:18 [INFO] LogiTranslatePlugin: Translation completed
2024-01-15 10:30:18 [DEBUG] BoostPlugin: Appending EOS tokens
2024-01-15 10:30:18 [INFO] BoostPlugin: BOOST completed successfully
2024-01-15 10:30:18 [DEBUG] LogiAttackPlugin: Executing logical schema
2024-01-15 10:30:20 [INFO] LogiAttackPlugin: LogiAttack completed successfully
2024-01-15 10:30:20 [INFO] PromptMatryoshka: Pipeline completed successfully
```

### Format Components

- **Timestamp**: `2024-01-15 10:30:15` - ISO format date and time
- **Level**: `[INFO]` - Log level in brackets
- **Logger Name**: `PromptMatryoshka` - Module or plugin name
- **Message**: `Starting pipeline...` - The actual log message

## Configuration Options

### Log Levels

The system supports all standard Python logging levels:

```python
import logging

# Available levels (in order of severity)
logging.DEBUG    # Detailed diagnostic information
logging.INFO     # General information about program execution
logging.WARNING  # Something unexpected happened
logging.ERROR    # A more serious problem occurred
logging.CRITICAL # A very serious error occurred
```

### Custom Configuration

While the default configuration is suitable for most use cases, it can be customized:

```python
import logging
from promptmatryoshka.logging_utils import get_logger

# Custom logging configuration
logging.basicConfig(
    level=logging.INFO,  # Change default level
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("promptmatryoshka.log")  # File output
    ]
)

logger = get_logger("CustomModule")
```

## Error Handling

### Exception Logging

The logging system supports detailed exception logging:

```python
from promptmatryoshka.logging_utils import get_logger

logger = get_logger("ErrorHandler")

try:
    risky_operation()
except Exception as e:
    # Log with full traceback
    logger.error("Operation failed", exc_info=True)
    
    # Log with custom message
    logger.error(f"Failed to process: {e}")
```

### Contextual Error Information

```python
logger = get_logger("PluginExecutor")

def execute_plugin(plugin, input_data):
    try:
        return plugin.run(input_data)
    except Exception as e:
        logger.error(
            f"Plugin {plugin.__class__.__name__} failed: {e}",
            extra={
                "plugin": plugin.__class__.__name__,
                "input_preview": input_data[:100],
                "error_type": type(e).__name__
            }
        )
        raise
```

## Developer Notes

### Logging Best Practices

1. **Use Appropriate Levels**:
   - `DEBUG`: Detailed diagnostic information
   - `INFO`: Confirmation of expected behavior
   - `WARNING`: Something unexpected but not necessarily an error
   - `ERROR`: Serious problems that should be addressed
   - `CRITICAL`: Very serious errors that may cause the program to abort

2. **Meaningful Messages**:
   ```python
   # Good
   logger.info(f"Processing {len(items)} items")
   
   # Bad
   logger.info("Processing items")
   ```

3. **Contextual Information**:
   ```python
   # Include relevant context
   logger.error(f"Failed to load plugin {plugin_name}: {error}")
   ```

4. **Performance Considerations**:
   ```python
   # Use lazy formatting for debug messages
   logger.debug("Processing data: %s", expensive_operation())
   ```

### Logger Naming Conventions

- **Module Names**: Use the module name as the logger name
- **Class Names**: Use the class name for class-specific loggers
- **Hierarchical Names**: Use dot notation for hierarchical loggers

```python
# Examples of good logger names
logger = get_logger("PromptMatryoshka")           # Main orchestrator
logger = get_logger("PromptMatryoshka.Core")      # Core module
logger = get_logger("FlipAttackPlugin")           # Plugin
logger = get_logger("CLI.AdvBench")               # CLI submodule
```

### Thread Safety

Python's logging module is thread-safe, making it suitable for concurrent operations:

```python
import threading
from promptmatryoshka.logging_utils import get_logger

logger = get_logger("ThreadedOperation")

def worker_function(data):
    logger.info(f"Worker processing: {data}")
    # Thread-safe logging
```

### Memory Considerations

- **Logger Caching**: Python caches logger instances, so repeated calls to `get_logger()` with the same name return the same instance
- **Message Formatting**: Use lazy formatting for expensive operations
- **Log Rotation**: Consider log rotation for long-running applications

## Implementation Details

### Logging Configuration

```python
def setup_logging():
    """Implementation details of logging setup."""
    logging.basicConfig(
        level=logging.DEBUG,        # Capture all messages
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"  # ISO format timestamps
    )
```

**Design Decisions:**

1. **DEBUG Level**: Captures all log messages for comprehensive debugging
2. **Standardized Format**: Consistent format across all modules
3. **Timestamp Format**: ISO-style timestamps for clarity
4. **Stream Handler**: Default to stderr for console visibility

### Logger Factory

```python
def get_logger(name):
    """Simple factory function for logger instances."""
    return logging.getLogger(name)
```

This simple implementation leverages Python's built-in logger caching and configuration inheritance.

### Integration Pattern

All modules follow a consistent pattern:

```python
from promptmatryoshka.logging_utils import get_logger

class SomeClass:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def some_method(self):
        self.logger.debug("Method called")
        # ... implementation
        self.logger.info("Method completed")
```

## Future Enhancements

### Planned Features

1. **Configurable Levels**: Runtime configuration of log levels
2. **File Rotation**: Automatic log file rotation
3. **Structured Logging**: JSON-formatted logs for better parsing
4. **Performance Metrics**: Built-in performance logging
5. **Remote Logging**: Support for centralized logging systems

### Advanced Features

1. **Contextual Logging**: Automatic inclusion of context information
2. **Async Logging**: Asynchronous logging for high-performance scenarios
3. **Log Filtering**: Advanced filtering based on content or source
4. **Monitoring Integration**: Integration with monitoring systems

### Configuration Enhancements

1. **Environment Variables**: Configuration via environment variables
2. **Configuration Files**: External configuration file support
3. **Dynamic Reconfiguration**: Runtime log level adjustment
4. **Plugin-Specific Configuration**: Per-plugin log level settings

The logging utilities module provides a solid foundation for consistent, comprehensive logging throughout the PromptMatryoshka framework, enabling effective debugging, monitoring, and audit trail maintenance.