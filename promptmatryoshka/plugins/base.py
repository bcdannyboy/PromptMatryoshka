"""Base plugin interface for PromptMatryoshka plugins.

This module defines the standard interface for all attack stage plugins.
All plugins must inherit from PluginBase and implement the run() method.

Plugin contract:
    - Each plugin processes a string input and returns a string output.
    - Plugins are designed to be composable in a multi-stage pipeline.
    - Plugins declare their category and dependencies via class attributes.

Usage:
    class MyPlugin(PluginBase):
        PLUGIN_CATEGORY = "mutation"
        PLUGIN_REQUIRES = ["some_dependency"]
        
        def run(self, input_data):
            # Transform input_data
            return output_data
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class PluginCategory(Enum):
    """Predefined plugin categories."""
    MUTATION = "mutation"
    TARGET = "target"
    EVALUATION = "evaluation"


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


@dataclass
class ValidationResult:
    """Result of plugin validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PluginValidationError(Exception):
    """Raised when plugin validation fails."""
    pass


class PluginBase:
    """
    Abstract base class for all PromptMatryoshka plugins.

    All plugins must implement the run() method, which takes a string input
    and returns a string output. This interface is used by the pipeline
    orchestrator to invoke each stage.
    
    Class Attributes:
        PLUGIN_CATEGORY (str): Plugin category (mutation, target, evaluation)
        PLUGIN_REQUIRES (List[str]): List of required plugin names or categories
        PLUGIN_CONFLICTS (List[str]): List of conflicting plugin names or categories
        PLUGIN_PROVIDES (List[str]): List of capabilities this plugin provides

    Methods:
        run(input_data: str) -> str
            Processes input and returns output for the pipeline.

    Example:
        class MyPlugin(PluginBase):
            PLUGIN_CATEGORY = "mutation"
            PLUGIN_REQUIRES = ["logitranslate"]
            
            def run(self, input_data):
                # ... transform input_data ...
                return output_data
    """
    
    # Class-level metadata - to be overridden by subclasses
    PLUGIN_CATEGORY: str = None
    PLUGIN_REQUIRES: List[str] = []
    PLUGIN_CONFLICTS: List[str] = []
    PLUGIN_PROVIDES: List[str] = []
    
    def __init_subclass__(cls, **kwargs):
        """Validate plugin class configuration on subclass creation."""
        super().__init_subclass__(**kwargs)
        
        # Validate category
        if cls.PLUGIN_CATEGORY is not None:
            valid_categories = [cat.value for cat in PluginCategory]
            if cls.PLUGIN_CATEGORY not in valid_categories:
                raise PluginValidationError(
                    f"Invalid plugin category '{cls.PLUGIN_CATEGORY}'. "
                    f"Must be one of: {valid_categories}"
                )
        
        # Ensure list attributes are actually lists
        if not isinstance(cls.PLUGIN_REQUIRES, list):
            cls.PLUGIN_REQUIRES = list(cls.PLUGIN_REQUIRES) if cls.PLUGIN_REQUIRES else []
        if not isinstance(cls.PLUGIN_CONFLICTS, list):
            cls.PLUGIN_CONFLICTS = list(cls.PLUGIN_CONFLICTS) if cls.PLUGIN_CONFLICTS else []
        if not isinstance(cls.PLUGIN_PROVIDES, list):
            cls.PLUGIN_PROVIDES = list(cls.PLUGIN_PROVIDES) if cls.PLUGIN_PROVIDES else []

    def run(self, input_data):
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
    
    @classmethod
    def get_plugin_name(cls) -> str:
        """Get the plugin name (class name in lowercase)."""
        return cls.__name__.lower().replace('plugin', '')
    
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        """
        Get plugin metadata.
        
        Returns:
            PluginMetadata: Plugin metadata structure.
        """
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
    
    @classmethod
    def get_input_schema(cls) -> Dict[str, Any]:
        """
        Get the input schema for this plugin.
        
        Returns:
            Dict[str, Any]: JSON schema for input validation.
        """
        return {"type": "string"}
    
    @classmethod
    def get_output_schema(cls) -> Dict[str, Any]:
        """
        Get the output schema for this plugin.
        
        Returns:
            Dict[str, Any]: JSON schema for output validation.
        """
        return {"type": "string"}
    
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


class PluginRegistry:
    """Registry for managing plugins and their metadata."""
    
    def __init__(self):
        self._plugins: Dict[str, type] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
    
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
    
    def get_plugin_class(self, name: str) -> Optional[type]:
        """Get a plugin class by name."""
        return self._plugins.get(name)
    
    def get_plugin_metadata(self, name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by name."""
        return self._metadata.get(name)
    
    def get_plugins_by_category(self, category: str) -> List[str]:
        """Get all plugin names in a specific category."""
        return [
            name for name, metadata in self._metadata.items()
            if metadata.category == category
        ]
    
    def get_all_plugins(self) -> Dict[str, PluginMetadata]:
        """Get all registered plugins and their metadata."""
        return self._metadata.copy()
    
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


# Global plugin registry instance
_plugin_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance."""
    return _plugin_registry


def register_plugin(plugin_class: type) -> None:
    """Register a plugin class with the global registry."""
    _plugin_registry.register(plugin_class)