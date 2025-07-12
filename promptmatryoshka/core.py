"""Core engine for PromptMatryoshka.

Coordinates the multi-stage jailbreak pipeline, manages plugin loading,
and exposes the main API for running attacks with dependency resolution.

Classes:
    PromptMatryoshka: Main orchestrator class with dependency resolution.
    PipelineBuilder: Builds and validates plugin pipelines.
"""

from typing import List, Dict, Any, Optional, Union, Type
from collections import defaultdict, deque
import inspect
import importlib
import os
from pathlib import Path

from promptmatryoshka.plugins.base import (
    PluginBase,
    PluginRegistry,
    PluginValidationError,
    ValidationResult,
    get_plugin_registry,
    register_plugin
)
from promptmatryoshka.logging_utils import get_logger
from promptmatryoshka.config import get_config, Config
from promptmatryoshka.llm_factory import get_factory, LLMFactory
from promptmatryoshka.exceptions import (
    LLMError,
    LLMConfigurationError,
    LLMUnsupportedProviderError,
    LLMValidationError,
    PipelineValidationError,
    CircularDependencyError
)

logger = get_logger("PromptMatryoshka")


class PipelineBuilder:
    """Builds and validates plugin pipelines with dependency resolution."""
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        """
        Initialize the pipeline builder.
        
        Args:
            registry: Plugin registry to use. If None, uses global registry.
        """
        self.registry = registry or get_plugin_registry()
        self.logger = get_logger("PipelineBuilder")
    
    def build_pipeline(self, plugin_names: List[str]) -> List[PluginBase]:
        """
        Build a pipeline from plugin names with dependency resolution.
        
        Args:
            plugin_names: List of plugin names to include in the pipeline.
            
        Returns:
            List of plugin instances in dependency-resolved order.
            
        Raises:
            PipelineValidationError: If pipeline validation fails.
            CircularDependencyError: If circular dependencies are detected.
        """
        self.logger.info(f"Building pipeline with plugins: {plugin_names}")
        
        # Validate that all plugins exist
        missing_plugins = []
        for name in plugin_names:
            if not self.registry.get_plugin_class(name):
                missing_plugins.append(name)
        
        if missing_plugins:
            raise PipelineValidationError(
                f"Missing plugins: {missing_plugins}"
            )
        
        # Resolve dependencies
        resolved_names = self._resolve_dependencies(plugin_names)
        self.logger.info(f"Resolved plugin order: {resolved_names}")
        
        # Validate the complete pipeline
        validation_result = self.registry.validate_dependencies(resolved_names)
        if not validation_result.valid:
            raise PipelineValidationError(
                f"Pipeline validation failed: {'; '.join(validation_result.errors)}"
            )
        
        # Create plugin instances
        pipeline = []
        for name in resolved_names:
            plugin_class = self.registry.get_plugin_class(name)
            if plugin_class:
                try:
                    plugin_instance = plugin_class()
                    pipeline.append(plugin_instance)
                    self.logger.debug(f"Created plugin instance: {name}")
                except Exception as e:
                    raise PipelineValidationError(
                        f"Failed to create plugin instance '{name}': {e}"
                    )
        
        return pipeline
    
    def _resolve_dependencies(self, plugin_names: List[str]) -> List[str]:
        """
        Resolve plugin dependencies using topological sort.
        
        Args:
            plugin_names: List of plugin names to resolve.
            
        Returns:
            List of plugin names in dependency order.
            
        Raises:
            CircularDependencyError: If circular dependencies are detected.
        """
        # Build dependency graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        all_plugins = set(plugin_names)
        
        # Add dependencies to the graph
        for name in plugin_names:
            metadata = self.registry.get_plugin_metadata(name)
            if metadata:
                in_degree[name] = 0  # Initialize
                for dep in metadata.requires:
                    # Check if dependency is a plugin name
                    if self.registry.get_plugin_class(dep):
                        if dep not in all_plugins:
                            all_plugins.add(dep)
                            self.logger.info(f"Adding required dependency: {dep}")
                        graph[dep].append(name)
                        in_degree[name] += 1
                    else:
                        # Check if dependency is a category
                        category_plugins = self.registry.get_plugins_by_category(dep)
                        if category_plugins:
                            # Find a plugin from the required category in our list
                            found = False
                            for cat_plugin in category_plugins:
                                if cat_plugin in all_plugins:
                                    graph[cat_plugin].append(name)
                                    in_degree[name] += 1
                                    found = True
                                    break
                            if not found:
                                raise PipelineValidationError(
                                    f"Plugin '{name}' requires category '{dep}' "
                                    f"but no plugins from that category are available"
                                )
        
        # Initialize in-degree for all plugins
        for plugin in all_plugins:
            if plugin not in in_degree:
                in_degree[plugin] = 0
        
        # Topological sort using Kahn's algorithm
        queue = deque([plugin for plugin in all_plugins if in_degree[plugin] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for circular dependencies
        if len(result) != len(all_plugins):
            remaining = [p for p in all_plugins if p not in result]
            raise CircularDependencyError(
                f"Circular dependency detected among plugins: {remaining}"
            )
        
        return result
    
    def validate_pipeline(self, plugin_names: List[str]) -> ValidationResult:
        """
        Validate a pipeline configuration without building it.
        
        Args:
            plugin_names: List of plugin names to validate.
            
        Returns:
            ValidationResult with any errors or warnings.
        """
        errors = []
        warnings = []
        
        try:
            # Test dependency resolution
            resolved_names = self._resolve_dependencies(plugin_names)
            
            # Validate with registry
            validation_result = self.registry.validate_dependencies(resolved_names)
            errors.extend(validation_result.errors)
            warnings.extend(validation_result.warnings)
            
        except (PipelineValidationError, CircularDependencyError) as e:
            errors.append(str(e))
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class PromptMatryoshka:
    """
    Orchestrates the pipeline: loads plugins, manages data flow,
    and exposes the main jailbreak() API with dependency resolution.

    Methods:
        jailbreak(prompt: str, plugins=None) -> str
            Runs the prompt through all pipeline stages and returns the result.
    """

    def __init__(self, plugins: Optional[List[Union[str, PluginBase]]] = None,
                 stages: Optional[List[PluginBase]] = None,  # For backward compatibility
                 auto_discover: bool = True,
                 config_path: Optional[str] = None,
                 provider: Optional[str] = None,
                 profile: Optional[str] = None):
        """
        Initialize the PromptMatryoshka orchestrator.
        
        Args:
            plugins: List of plugin names or instances to use. If None, uses auto-discovery.
            stages: List of plugin instances (deprecated, use plugins instead).
            auto_discover: Whether to automatically discover and register plugins.
            config_path: Path to configuration file.
            provider: Default provider to use for plugins.
            profile: Default profile to use for plugins.
        """
        self.logger = get_logger("PromptMatryoshka")
        self.registry = get_plugin_registry()
        self.builder = PipelineBuilder(self.registry)
        
        # Initialize configuration and LLM factory
        self.config = get_config(config_path or "config.json")
        self.llm_factory = get_factory()
        
        # Store provider and profile preferences
        self.default_provider = provider
        self.default_profile = profile
        
        # Auto-discover plugins if requested
        if auto_discover:
            self._auto_discover_plugins()
        
        # Handle backward compatibility for stages parameter
        if stages is not None:
            if plugins is not None:
                raise ValueError("Cannot specify both 'plugins' and 'stages' parameters")
            self.logger.warning("The 'stages' parameter is deprecated, use 'plugins' instead")
            self.pipeline = stages
        elif plugins is None:
            # Use a default pipeline if available
            self.pipeline = self._build_default_pipeline()
        else:
            self.pipeline = self._build_pipeline_from_spec(plugins)
    
    def _auto_discover_plugins(self):
        """Automatically discover and register plugins."""
        self.logger.info("Auto-discovering plugins...")
        
        # Import and register all plugins in the plugins directory
        plugins_dir = Path(__file__).parent / "plugins"
        
        for plugin_file in plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("_") or plugin_file.name == "base.py":
                continue
                
            module_name = f"promptmatryoshka.plugins.{plugin_file.stem}"
            try:
                module = importlib.import_module(module_name)
                
                # Find plugin classes in the module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, PluginBase) and 
                        obj is not PluginBase):
                        try:
                            register_plugin(obj)
                            self.logger.debug(f"Registered plugin: {obj.get_plugin_name()}")
                        except PluginValidationError as e:
                            self.logger.warning(f"Failed to register plugin {name}: {e}")
                        
            except Exception as e:
                self.logger.warning(f"Failed to import plugin module {module_name}: {e}")
    
    def _build_default_pipeline(self) -> List[PluginBase]:
        """Build a default pipeline from available plugins."""
        all_plugins = self.registry.get_all_plugins()
        
        if not all_plugins:
            self.logger.warning("No plugins available for default pipeline")
            return []
        
        # Try to build the full research-based pipeline: FlipAttack → LogiTranslate → BOOST → LogiAttack
        # This matches the pipeline used in the advbench command for consistency
        full_pipeline_names = ['flipattack', 'logitranslate', 'boost', 'logiattack']
        
        # First, try to build the complete pipeline
        try:
            full_pipeline = self.builder.build_pipeline(full_pipeline_names)
            self.logger.info(f"Built full default pipeline with {len(full_pipeline)} plugins: {full_pipeline_names}")
            return full_pipeline
        except Exception as e:
            self.logger.debug(f"Failed to build full pipeline: {e}")
            # Fall back to partial pipeline or individual plugins
        
        # Try to build a partial pipeline with available plugins from the preferred order
        pipeline_instances = []
        for plugin_name in full_pipeline_names:
            try:
                plugin_class = self.registry.get_plugin_class(plugin_name)
                if plugin_class:
                    # Test if we can create an instance
                    test_instance = plugin_class()
                    pipeline_instances.append(test_instance)
                    self.logger.debug(f"Added {plugin_name} to partial pipeline")
            except Exception as e:
                self.logger.debug(f"Plugin {plugin_name} not available: {e}")
                continue
        
        if pipeline_instances:
            self.logger.info(f"Built partial pipeline with {len(pipeline_instances)} plugins")
            return pipeline_instances
        
        # If no plugins from the preferred pipeline work, try mutation plugins as fallback
        mutation_plugins = self.registry.get_plugins_by_category("mutation")
        for plugin_name in mutation_plugins:
            try:
                plugin_class = self.registry.get_plugin_class(plugin_name)
                if plugin_class:
                    # Test if we can create an instance
                    test_instance = plugin_class()
                    self.logger.info(f"Using fallback mutation plugin: {plugin_name}")
                    return [test_instance]
            except Exception as e:
                self.logger.debug(f"Plugin {plugin_name} not available: {e}")
                continue
        
        # Final fallback: try any working plugin
        for plugin_name in all_plugins.keys():
            try:
                plugin_class = self.registry.get_plugin_class(plugin_name)
                if plugin_class:
                    # Test if we can create an instance
                    test_instance = plugin_class()
                    self.logger.info(f"Using fallback plugin: {plugin_name}")
                    return [test_instance]
            except Exception as e:
                self.logger.debug(f"Plugin {plugin_name} not available: {e}")
                continue
        
        self.logger.warning("No plugins could be initialized for default pipeline")
        return []
    
    def _build_pipeline_from_spec(self, plugins: List[Union[str, PluginBase]]) -> List[PluginBase]:
        """Build pipeline from a specification of plugin names or instances."""
        # Separate names from instances
        plugin_names = []
        plugin_instances = []
        
        for plugin in plugins:
            if isinstance(plugin, str):
                plugin_names.append(plugin)
            elif isinstance(plugin, PluginBase):
                plugin_instances.append(plugin)
            else:
                raise ValueError(f"Invalid plugin specification: {plugin}")
        
        # Build pipeline from names
        if plugin_names:
            try:
                resolved_instances = self.builder.build_pipeline(plugin_names)
                plugin_instances.extend(resolved_instances)
            except (PipelineValidationError, CircularDependencyError) as e:
                self.logger.error(f"Failed to build pipeline: {e}")
                raise
        
        return plugin_instances

    def jailbreak(self, prompt: str, plugins: Optional[List[Union[str, PluginBase]]] = None,
                  provider: Optional[str] = None, profile: Optional[str] = None) -> str:
        """
        Runs the prompt through all pipeline stages and returns the result.

        Args:
            prompt (str): The input prompt to process.
            plugins: Optional override for the pipeline plugins.
            provider: Optional provider override for this run.
            profile: Optional profile override for this run.

        Returns:
            str: The final output after all pipeline stages.
        """
        # Use custom pipeline if provided, otherwise use default
        if plugins is not None:
            pipeline = self._build_pipeline_from_spec(plugins)
        else:
            pipeline = self.pipeline
        
        if not pipeline:
            self.logger.warning("No plugins in pipeline, returning input unchanged")
            return prompt
        
        # Use provided provider/profile or fall back to instance defaults
        run_provider = provider or self.default_provider
        run_profile = profile or self.default_profile
        
        self.logger.info(f"Running pipeline with {len(pipeline)} plugins"
                        f"{f' (provider: {run_provider})' if run_provider else ''}"
                        f"{f' (profile: {run_profile})' if run_profile else ''}")
        
        # Run the pipeline
        data = prompt
        for i, stage in enumerate(pipeline):
            # Handle both PluginBase and non-PluginBase classes for backward compatibility
            if hasattr(stage, 'get_plugin_name'):
                plugin_name = stage.get_plugin_name()
            else:
                plugin_name = stage.__class__.__name__
            
            self.logger.debug(f"Running stage {i+1}/{len(pipeline)}: {plugin_name}")
            
            try:
                # Validate input (only for PluginBase instances)
                if hasattr(stage, 'validate_input'):
                    input_validation = stage.validate_input(data)
                    if not input_validation.valid:
                        self.logger.warning(
                            f"Plugin {plugin_name} input validation failed: "
                            f"{'; '.join(input_validation.errors)}"
                        )
                
                # Try to configure plugin with new LLM system if it supports it
                if hasattr(stage, 'configure_llm') and (run_provider or run_profile):
                    try:
                        llm_interface = self._create_llm_for_plugin(plugin_name, run_provider, run_profile)
                        if llm_interface:
                            stage.configure_llm(llm_interface)
                    except Exception as e:
                        self.logger.warning(f"Failed to configure LLM for plugin {plugin_name}: {e}")
                
                # Run the plugin
                data = stage.run(data)
                
                # Validate output (only for PluginBase instances)
                if hasattr(stage, 'validate_output'):
                    output_validation = stage.validate_output(data)
                    if not output_validation.valid:
                        self.logger.warning(
                            f"Plugin {plugin_name} output validation failed: "
                            f"{'; '.join(output_validation.errors)}"
                        )
                
                self.logger.debug(f"Stage {plugin_name} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Plugin {plugin_name} failed: {e}")
                raise
        
        self.logger.info("Pipeline completed successfully")
        return data
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline."""
        plugin_info = []
        for plugin in self.pipeline:
            if hasattr(plugin, 'get_plugin_name'):
                # PluginBase instance
                plugin_info.append({
                    "name": plugin.get_plugin_name(),
                    "category": getattr(plugin, 'PLUGIN_CATEGORY', 'unknown'),
                    "requires": getattr(plugin, 'PLUGIN_REQUIRES', []),
                    "conflicts": getattr(plugin, 'PLUGIN_CONFLICTS', []),
                    "provides": getattr(plugin, 'PLUGIN_PROVIDES', [])
                })
            else:
                # Non-PluginBase class (backward compatibility)
                plugin_info.append({
                    "name": plugin.__class__.__name__,
                    "category": "unknown",
                    "requires": [],
                    "conflicts": [],
                    "provides": []
                })
        
        return {
            "plugins": plugin_info,
            "total_plugins": len(self.pipeline),
            "available_plugins": list(self.registry.get_all_plugins().keys())
        }
    
    def validate_configuration(self) -> ValidationResult:
        """Validate the current pipeline configuration."""
        if not self.pipeline:
            return ValidationResult(
                valid=False,
                errors=["No plugins in pipeline"]
            )
        
        plugin_names = [plugin.get_plugin_name() for plugin in self.pipeline]
        return self.builder.validate_pipeline(plugin_names)
    
    def _create_llm_for_plugin(self, plugin_name: str, provider: Optional[str] = None,
                               profile: Optional[str] = None) -> Optional[Any]:
        """Create an LLM interface for a specific plugin using the new configuration system.
        
        Args:
            plugin_name: Name of the plugin
            provider: Optional provider override
            profile: Optional profile override
            
        Returns:
            LLM interface instance or None if creation fails
        """
        try:
            # If profile is specified, use it
            if profile:
                self.logger.debug(f"Creating LLM for plugin {plugin_name} using profile {profile}")
                return self.llm_factory.create_from_profile(profile)
            
            # If provider is specified, use it with plugin config
            if provider:
                self.logger.debug(f"Creating LLM for plugin {plugin_name} using provider {provider}")
                plugin_config = self.config.get_plugin_config(plugin_name)
                
                if plugin_config:
                    llm_config = {}
                    for field in ["model", "temperature", "max_tokens", "top_p",
                                 "frequency_penalty", "presence_penalty", "request_timeout"]:
                        value = getattr(plugin_config, field, None)
                        if value is not None:
                            llm_config[field] = value
                    
                    # Use provider's default model if not specified
                    if "model" not in llm_config:
                        provider_config = self.config.get_provider_config(provider)
                        if provider_config:
                            llm_config["model"] = provider_config.default_model
                    
                    return self.llm_factory.create_interface(provider, llm_config)
            
            # Use the configuration system to create LLM for plugin
            return self.config.create_llm_for_plugin(plugin_name)
            
        except Exception as e:
            self.logger.warning(f"Failed to create LLM for plugin {plugin_name}: {e}")
            return None
    
    def set_provider(self, provider: str) -> None:
        """Set the default provider for the pipeline.
        
        Args:
            provider: Provider name to use
        """
        self.default_provider = provider
        self.logger.info(f"Default provider set to: {provider}")
    
    def set_profile(self, profile: str) -> None:
        """Set the default profile for the pipeline.
        
        Args:
            profile: Profile name to use
        """
        self.default_profile = profile
        self.logger.info(f"Default profile set to: {profile}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers.
        
        Returns:
            List of provider names
        """
        return self.config.get_available_providers()
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available profiles.
        
        Returns:
            List of profile names
        """
        return self.config.get_available_profiles()