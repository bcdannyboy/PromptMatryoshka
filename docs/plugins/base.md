# Plugin Base Class Documentation

This document provides comprehensive documentation for the [`PluginBase`](../../promptmatryoshka/plugins/base.py) abstract class and the plugin registry system that forms the foundation of the PromptMatryoshka plugin architecture.

## Table of Contents

- [PluginBase Class](#pluginbase-class)
- [Plugin Metadata](#plugin-metadata)
- [PluginRegistry](#pluginregistry)
- [Validation System](#validation-system)
- [Error Handling](#error-handling)
- [Plugin Discovery](#plugin-discovery)
- [Development Examples](#development-examples)

## PluginBase Class

The [`PluginBase`](../../promptmatryoshka/plugins/base.py#L60) abstract class defines the standard interface that all PromptMatryoshka plugins must implement.

### Class Definition

```python
class PluginBase:
    """
    Abstract base class for all PromptMatryoshka plugins.
    
    All plugins must implement the run() method, which takes a string input
    and returns a string output. This interface is used by the pipeline
    orchestrator to invoke each stage.
    """
    
    # Class-level metadata - to be overridden by subclasses
    PLUGIN_CATEGORY: str = None
    PLUGIN_REQUIRES: List[str] = []
    PLUGIN_CONFLICTS: List[str] = []
    PLUGIN_PROVIDES: List[str] = []
```

### Required Implementation

Every plugin must implement the [`run()`](../../promptmatryoshka/plugins/base.py#L115) method:

```python
def run(self, input_data: str) -> str:
    """
    Transform the input data and return the result.
    
    Args:
        input_data (str): The input prompt or intermediate data.
        
    Returns:
        str: The transformed output.
        
    Raises:
        NotImplementedError: If not implemented by subclass.
    """
    raise NotImplementedError("Plugins must implement the run() method.")
```

### Class Attributes

#### PLUGIN_CATEGORY

Defines the plugin's category for organization and pipeline management.

**Valid Categories:**
- `"mutation"`: Transforms and manipulates input prompts
- `"target"`: Generates responses from transformed prompts  
- `"evaluation"`: Assesses and validates responses

```python
class MyPlugin(PluginBase):
    PLUGIN_CATEGORY = "mutation"
```

#### PLUGIN_REQUIRES

Lists dependencies that must be satisfied before this plugin can run.

```python
class MyPlugin(PluginBase):
    PLUGIN_REQUIRES = ["logitranslate"]  # Depends on LogiTranslate plugin
```

#### PLUGIN_CONFLICTS

Lists plugins that conflict with this plugin and cannot run together.

```python
class MyPlugin(PluginBase):
    PLUGIN_CONFLICTS = ["boost"]  # Cannot run with BOOST plugin
```

#### PLUGIN_PROVIDES

Lists capabilities or features this plugin provides to other plugins.

```python
class MyPlugin(PluginBase):
    PLUGIN_PROVIDES = ["logical_schema"]  # Provides logical schema capability
```

### Core Methods

#### get_plugin_name()

Returns the plugin's name derived from the class name.

```python
@classmethod
def get_plugin_name(cls) -> str:
    """Get the plugin name (class name in lowercase)."""
    return cls.__name__.lower().replace('plugin', '')
```

#### get_metadata()

Returns comprehensive metadata about the plugin.

```python
@classmethod
def get_metadata(cls) -> PluginMetadata:
    """Get plugin metadata."""
    return PluginMetadata(
        name=cls.get_plugin_name(),
        category=cls.PLUGIN_CATEGORY,
        requires=cls.PLUGIN_REQUIRES.copy(),
        conflicts=cls.PLUGIN_CONFLICTS.copy(),
        provides=cls.PLUGIN_PROVIDES.copy(),
        description=cls.__doc__ or "",
        input_schema=cls.get_input_schema(),
        output_schema=cls.get_output_schema()
    )
```

### Validation Methods

#### validate_input()

Validates input data against the plugin's requirements.

```python
def validate_input(self, input_data: Any) -> ValidationResult:
    """
    Validate input data against the plugin's input schema.
    
    Args:
        input_data: The input data to validate.
        
    Returns:
        ValidationResult: Validation result with any errors or warnings.
    """
    # Basic validation - can be overridden by subclasses
    if not isinstance(input_data, str):
        return ValidationResult(
            valid=False,
            errors=[f"Input must be a string, got {type(input_data)}"]
        )
    
    return ValidationResult(valid=True)
```

#### validate_output()

Validates output data against the plugin's output schema.

```python
def validate_output(self, output_data: Any) -> ValidationResult:
    """
    Validate output data against the plugin's output schema.
    
    Args:
        output_data: The output data to validate.
        
    Returns:
        ValidationResult: Validation result with any errors or warnings.
    """
    # Basic validation - can be overridden by subclasses
    if not isinstance(output_data, str):
        return ValidationResult(
            valid=False,
            errors=[f"Output must be a string, got {type(output_data)}"]
        )
    
    return ValidationResult(valid=True)
```

#### validate_configuration()

Validates the plugin's configuration and metadata.

```python
@classmethod
def validate_configuration(cls) -> ValidationResult:
    """
    Validate the plugin's configuration.
    
    Returns:
        ValidationResult: Validation result with any errors or warnings.
    """
    errors = []
    warnings = []
    
    # Check if category is set
    if cls.PLUGIN_CATEGORY is None:
        warnings.append(f"Plugin {cls.get_plugin_name()} has no category set")
    
    # Check for self-dependencies
    plugin_name = cls.get_plugin_name()
    if plugin_name in cls.PLUGIN_REQUIRES:
        errors.append(f"Plugin {plugin_name} cannot require itself")
    
    # Check for self-conflicts
    if plugin_name in cls.PLUGIN_CONFLICTS:
        errors.append(f"Plugin {plugin_name} cannot conflict with itself")
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
```

### Schema Methods

#### get_input_schema()

Returns the JSON schema for input validation.

```python
@classmethod
def get_input_schema(cls) -> Dict[str, Any]:
    """
    Get the input schema for this plugin.
    
    Returns:
        Dict[str, Any]: JSON schema for input validation.
    """
    return {"type": "string"}
```

#### get_output_schema()

Returns the JSON schema for output validation.

```python
@classmethod
def get_output_schema(cls) -> Dict[str, Any]:
    """
    Get the output schema for this plugin.
    
    Returns:
        Dict[str, Any]: JSON schema for output validation.
    """
    return {"type": "string"}
```

## Plugin Metadata

The [`PluginMetadata`](../../promptmatryoshka/plugins/base.py#L33) dataclass encapsulates all plugin information:

```python
@dataclass
class PluginMetadata:
    """Plugin metadata structure."""
    name: str
    category: str
    requires: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
```

### Metadata Fields

- **name**: Plugin identifier (derived from class name)
- **category**: Plugin category (`mutation`, `target`, `evaluation`)
- **requires**: List of required plugin names or categories
- **conflicts**: List of conflicting plugin names or categories
- **provides**: List of capabilities this plugin provides
- **version**: Plugin version string
- **description**: Plugin description (from docstring)
- **input_schema**: JSON schema for input validation
- **output_schema**: JSON schema for output validation

## PluginRegistry

The [`PluginRegistry`](../../promptmatryoshka/plugins/base.py#L243) manages plugin discovery, registration, and validation.

### Registry Methods

#### register()

Registers a plugin class with validation.

```python
def register(self, plugin_class: type) -> None:
    """
    Register a plugin class.
    
    Args:
        plugin_class: The plugin class to register.
    """
    if not issubclass(plugin_class, PluginBase):
        raise PluginValidationError(
            f"Plugin {plugin_class.__name__} must inherit from PluginBase"
        )
    
    # Validate plugin configuration
    validation_result = plugin_class.validate_configuration()
    if not validation_result.valid:
        raise PluginValidationError(
            f"Plugin {plugin_class.__name__} validation failed: "
            f"{'; '.join(validation_result.errors)}"
        )
    
    plugin_name = plugin_class.get_plugin_name()
    self._plugins[plugin_name] = plugin_class
    self._metadata[plugin_name] = plugin_class.get_metadata()
```

#### get_plugin_class()

Retrieves a plugin class by name.

```python
def get_plugin_class(self, name: str) -> Optional[type]:
    """Get a plugin class by name."""
    return self._plugins.get(name)
```

#### get_plugins_by_category()

Retrieves all plugins in a specific category.

```python
def get_plugins_by_category(self, category: str) -> List[str]:
    """Get all plugin names in a specific category."""
    return [
        name for name, metadata in self._metadata.items()
        if metadata.category == category
    ]
```

#### validate_dependencies()

Validates dependencies for a list of plugins.

```python
def validate_dependencies(self, plugin_names: List[str]) -> ValidationResult:
    """
    Validate dependencies for a list of plugins.
    
    Args:
        plugin_names: List of plugin names to validate.
        
    Returns:
        ValidationResult: Validation result with any errors or warnings.
    """
    errors = []
    warnings = []
    
    # Check if all plugins exist
    for name in plugin_names:
        if name not in self._plugins:
            errors.append(f"Plugin '{name}' not found")
    
    if errors:
        return ValidationResult(valid=False, errors=errors)
    
    # Check dependencies
    for name in plugin_names:
        metadata = self._metadata[name]
        
        # Check if required plugins are included
        for required in metadata.requires:
            if required not in plugin_names:
                # Check if required plugin is a category
                category_plugins = self.get_plugins_by_category(required)
                if category_plugins:
                    # Check if any plugin from the required category is included
                    if not any(p in plugin_names for p in category_plugins):
                        errors.append(
                            f"Plugin '{name}' requires category '{required}' "
                            f"but no plugins from that category are included"
                        )
                else:
                    errors.append(
                        f"Plugin '{name}' requires '{required}' but it's not included"
                    )
        
        # Check for conflicts
        for conflict in metadata.conflicts:
            if conflict in plugin_names:
                errors.append(
                    f"Plugin '{name}' conflicts with '{conflict}' "
                    f"but both are included"
                )
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
```

### Global Registry

A global registry instance is available throughout the system:

```python
# Global plugin registry instance
_plugin_registry = PluginRegistry()

def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance."""
    return _plugin_registry

def register_plugin(plugin_class: type) -> None:
    """Register a plugin class with the global registry."""
    _plugin_registry.register(plugin_class)
```

## Validation System

The validation system ensures plugin integrity and compatibility.

### ValidationResult

The [`ValidationResult`](../../promptmatryoshka/plugins/base.py#L47) dataclass encapsulates validation outcomes:

```python
@dataclass
class ValidationResult:
    """Result of plugin validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
```

### Validation Types

1. **Configuration Validation**: Ensures plugin metadata is valid
2. **Input Validation**: Validates input data against schema
3. **Output Validation**: Validates output data against schema
4. **Dependency Validation**: Ensures all dependencies are satisfied

### Custom Validation

Plugins can implement custom validation logic:

```python
class MyPlugin(PluginBase):
    def validate_input(self, input_data):
        # Custom validation logic
        if not self._is_valid_format(input_data):
            return ValidationResult(
                valid=False,
                errors=["Input format is invalid"]
            )
        return ValidationResult(valid=True)
    
    def _is_valid_format(self, data):
        # Custom format checking
        return True
```

## Error Handling

### PluginValidationError

Raised when plugin validation fails:

```python
class PluginValidationError(Exception):
    """Raised when plugin validation fails."""
    pass
```

### Error Scenarios

1. **Invalid Inheritance**: Plugin doesn't inherit from `PluginBase`
2. **Configuration Errors**: Invalid metadata or configuration
3. **Missing Dependencies**: Required plugins not available
4. **Conflicting Plugins**: Conflicting plugins loaded together
5. **Validation Failures**: Input/output validation failures

### Error Handling Example

```python
from promptmatryoshka.plugins.base import PluginValidationError

try:
    plugin = MyPlugin()
    result = plugin.run(input_data)
except PluginValidationError as e:
    logger.error(f"Plugin validation failed: {e}")
    # Handle validation error
except Exception as e:
    logger.error(f"Plugin execution failed: {e}")
    # Handle other errors
```

## Plugin Discovery

The plugin system automatically discovers and loads plugins from the [`promptmatryoshka/plugins/`](../../promptmatryoshka/plugins/) directory.

### Discovery Process

1. **Module Scanning**: Scan plugin directory for Python files
2. **Class Discovery**: Find classes that inherit from `PluginBase`
3. **Registration**: Automatically register discovered plugins
4. **Validation**: Validate plugin configuration and dependencies

### Manual Registration

Plugins can also be registered manually:

```python
from promptmatryoshka.plugins.base import register_plugin

class MyCustomPlugin(PluginBase):
    PLUGIN_CATEGORY = "mutation"
    
    def run(self, input_data):
        return f"Processed: {input_data}"

# Register the plugin
register_plugin(MyCustomPlugin)
```

## Development Examples

### Basic Plugin

```python
from promptmatryoshka.plugins.base import PluginBase

class SimplePlugin(PluginBase):
    """A simple example plugin."""
    
    PLUGIN_CATEGORY = "mutation"
    PLUGIN_REQUIRES = []
    PLUGIN_CONFLICTS = []
    PLUGIN_PROVIDES = ["simple_transform"]
    
    def run(self, input_data: str) -> str:
        """Apply simple transformation."""
        return f"TRANSFORMED: {input_data}"
```

### Plugin with Dependencies

```python
class AdvancedPlugin(PluginBase):
    """An advanced plugin with dependencies."""
    
    PLUGIN_CATEGORY = "target"
    PLUGIN_REQUIRES = ["logitranslate"]  # Requires LogiTranslate
    PLUGIN_CONFLICTS = ["boost"]         # Cannot run with BOOST
    PLUGIN_PROVIDES = ["advanced_processing"]
    
    def run(self, input_data: str) -> str:
        """Process input with advanced logic."""
        # Assume input_data is already transformed by LogiTranslate
        return self._process_logic(input_data)
    
    def _process_logic(self, data: str) -> str:
        # Advanced processing logic
        return f"PROCESSED: {data}"
```

### Plugin with Custom Validation

```python
class ValidatedPlugin(PluginBase):
    """A plugin with custom validation."""
    
    PLUGIN_CATEGORY = "evaluation"
    
    def run(self, input_data: str) -> str:
        """Evaluate input data."""
        return "EVALUATION_RESULT"
    
    def validate_input(self, input_data):
        """Custom input validation."""
        if not isinstance(input_data, str):
            return ValidationResult(
                valid=False,
                errors=["Input must be a string"]
            )
        
        if len(input_data) < 10:
            return ValidationResult(
                valid=False,
                warnings=["Input is very short"]
            )
        
        return ValidationResult(valid=True)
    
    @classmethod
    def get_input_schema(cls):
        """Custom input schema."""
        return {
            "type": "string",
            "minLength": 10,
            "maxLength": 10000
        }
```

### Plugin with Configuration

```python
class ConfigurablePlugin(PluginBase):
    """A plugin that uses configuration."""
    
    PLUGIN_CATEGORY = "mutation"
    
    def __init__(self, config_param="default"):
        self.config_param = config_param
    
    def run(self, input_data: str) -> str:
        """Transform input using configuration."""
        return f"{self.config_param}: {input_data}"
    
    @classmethod
    def from_config(cls, config_dict):
        """Create plugin from configuration."""
        return cls(config_param=config_dict.get("param", "default"))
```

## Integration with Core System

The plugin system integrates seamlessly with the core PromptMatryoshka system:

### Configuration Integration

```python
from promptmatryoshka.config import get_config
from promptmatryoshka.plugins.base import get_plugin_registry

# Get plugin configuration
config = get_config()
plugin_config = config.get_plugin_config("myplugin")

# Get plugin from registry
registry = get_plugin_registry()
plugin_class = registry.get_plugin_class("myplugin")
```

### Pipeline Integration

```python
from promptmatryoshka.core import PipelineBuilder
from promptmatryoshka.plugins.base import get_plugin_registry

# Build pipeline using registry
registry = get_plugin_registry()
builder = PipelineBuilder(registry)
pipeline = builder.build_pipeline(["flipattack", "logitranslate", "logiattack"])
```

### Logging Integration

```python
from promptmatryoshka.logging_utils import get_logger

class MyPlugin(PluginBase):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def run(self, input_data: str) -> str:
        self.logger.debug(f"Processing input: {input_data}")
        result = self._process(input_data)
        self.logger.info(f"Processing completed: {result}")
        return result
```

This comprehensive base class system provides a robust foundation for plugin development while maintaining consistency and reliability across the entire plugin ecosystem.