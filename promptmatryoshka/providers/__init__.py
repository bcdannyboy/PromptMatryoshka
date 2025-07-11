"""Provider registry for PromptMatryoshka LLM interfaces.

This module manages the registration and discovery of LLM providers.
It provides a centralized registry for all available providers and
handles provider instantiation and configuration.

The registry allows for dynamic provider discovery and supports
both built-in providers and custom provider implementations.

Functions:
    register_provider: Register a new provider class
    get_provider: Get a provider class by name
    list_providers: List all registered providers
    is_provider_available: Check if a provider is available
    get_provider_info: Get information about a provider
"""

from typing import Dict, Type, List, Optional, Any
import importlib
import logging
from abc import ABC

from ..llm_interface import LLMInterface
from ..exceptions import LLMUnsupportedProviderError, LLMConfigurationError

logger = logging.getLogger(__name__)

# Provider registry - maps provider names to their classes
_PROVIDER_REGISTRY: Dict[str, Type[LLMInterface]] = {}

# Provider availability cache - tracks which providers are available
_PROVIDER_AVAILABILITY: Dict[str, bool] = {}

# Built-in provider configurations
BUILTIN_PROVIDERS = {
    "openai": {
        "module": "promptmatryoshka.providers.openai_provider",
        "class": "OpenAIProvider",
        "dependencies": ["langchain-openai"],
        "description": "OpenAI GPT models (GPT-3.5, GPT-4, etc.)"
    },
    "anthropic": {
        "module": "promptmatryoshka.providers.anthropic_provider",
        "class": "AnthropicProvider",
        "dependencies": ["langchain-anthropic"],
        "description": "Anthropic Claude models"
    },
    "ollama": {
        "module": "promptmatryoshka.providers.ollama_provider",
        "class": "OllamaProvider",
        "dependencies": ["langchain-ollama"],
        "description": "Ollama local models"
    },
    "huggingface": {
        "module": "promptmatryoshka.providers.huggingface_provider",
        "class": "HuggingFaceProvider",
        "dependencies": ["langchain-huggingface"],
        "description": "HuggingFace transformers models"
    }
}


def register_provider(name: str, provider_class: Type[LLMInterface]) -> None:
    """Register a new LLM provider.
    
    Args:
        name: Provider name (should be lowercase)
        provider_class: Provider class that inherits from LLMInterface
        
    Raises:
        TypeError: If provider_class is not a subclass of LLMInterface
        ValueError: If provider name is invalid
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Provider name must be a non-empty string")
    
    name = name.lower().strip()
    
    if not issubclass(provider_class, LLMInterface):
        raise TypeError(f"Provider class {provider_class.__name__} must inherit from LLMInterface")
    
    if name in _PROVIDER_REGISTRY:
        logger.warning(f"Overriding existing provider registration for '{name}'")
    
    _PROVIDER_REGISTRY[name] = provider_class
    logger.info(f"Registered provider '{name}' -> {provider_class.__name__}")


def get_provider(name: str) -> Type[LLMInterface]:
    """Get a provider class by name.
    
    Args:
        name: Provider name
        
    Returns:
        Provider class
        
    Raises:
        LLMUnsupportedProviderError: If provider is not registered
    """
    name = name.lower().strip()
    
    if name not in _PROVIDER_REGISTRY:
        # Try to auto-load built-in provider
        if name in BUILTIN_PROVIDERS:
            _load_builtin_provider(name)
        
        # Check again after potential auto-loading
        if name not in _PROVIDER_REGISTRY:
            available_providers = list(_PROVIDER_REGISTRY.keys())
            raise LLMUnsupportedProviderError(
                f"Provider '{name}' is not registered",
                provider_name=name,
                supported_providers=available_providers
            )
    
    return _PROVIDER_REGISTRY[name]


def list_providers() -> List[str]:
    """List all registered provider names.
    
    Returns:
        List of provider names
    """
    return list(_PROVIDER_REGISTRY.keys())


def list_builtin_providers() -> List[str]:
    """List all built-in provider names.
    
    Returns:
        List of built-in provider names
    """
    return list(BUILTIN_PROVIDERS.keys())


def is_provider_available(name: str) -> bool:
    """Check if a provider is available.
    
    A provider is considered available if it's registered and all its
    dependencies are satisfied.
    
    Args:
        name: Provider name
        
    Returns:
        True if provider is available
    """
    name = name.lower().strip()
    
    # Check cache first
    if name in _PROVIDER_AVAILABILITY:
        return _PROVIDER_AVAILABILITY[name]
    
    # Check if provider is registered
    if name not in _PROVIDER_REGISTRY:
        # Try to auto-load built-in provider
        if name in BUILTIN_PROVIDERS:
            try:
                _load_builtin_provider(name)
            except Exception as e:
                logger.debug(f"Failed to load built-in provider '{name}': {e}")
                _PROVIDER_AVAILABILITY[name] = False
                return False
        else:
            _PROVIDER_AVAILABILITY[name] = False
            return False
    
    # Check dependencies for built-in providers
    if name in BUILTIN_PROVIDERS:
        available = _check_dependencies(BUILTIN_PROVIDERS[name]["dependencies"])
        _PROVIDER_AVAILABILITY[name] = available
        return available
    
    # For custom providers, assume available if registered
    _PROVIDER_AVAILABILITY[name] = True
    return True


def get_provider_info(name: str) -> Dict[str, Any]:
    """Get information about a provider.
    
    Args:
        name: Provider name
        
    Returns:
        Dictionary with provider information
        
    Raises:
        LLMUnsupportedProviderError: If provider is not found
    """
    name = name.lower().strip()
    
    # Check built-in providers first
    if name in BUILTIN_PROVIDERS:
        info = BUILTIN_PROVIDERS[name].copy()
        info["name"] = name
        info["available"] = is_provider_available(name)
        info["registered"] = name in _PROVIDER_REGISTRY
        return info
    
    # Check registered providers
    if name in _PROVIDER_REGISTRY:
        provider_class = _PROVIDER_REGISTRY[name]
        return {
            "name": name,
            "class": provider_class.__name__,
            "module": provider_class.__module__,
            "description": provider_class.__doc__ or "Custom provider",
            "available": True,
            "registered": True,
            "dependencies": []
        }
    
    # Provider not found
    available_providers = list(set(_PROVIDER_REGISTRY.keys()) | set(BUILTIN_PROVIDERS.keys()))
    raise LLMUnsupportedProviderError(
        f"Provider '{name}' not found",
        provider_name=name,
        supported_providers=available_providers
    )


def discover_providers() -> Dict[str, Dict[str, Any]]:
    """Discover all available providers.
    
    This function attempts to load all built-in providers and returns
    information about all providers (both available and unavailable).
    
    Returns:
        Dictionary mapping provider names to their information
    """
    providers = {}
    
    # Load information for all built-in providers
    for name in BUILTIN_PROVIDERS:
        try:
            providers[name] = get_provider_info(name)
        except Exception as e:
            logger.debug(f"Failed to get info for provider '{name}': {e}")
            providers[name] = {
                "name": name,
                "available": False,
                "error": str(e)
            }
    
    # Add information for custom registered providers
    for name in _PROVIDER_REGISTRY:
        if name not in providers:
            try:
                providers[name] = get_provider_info(name)
            except Exception as e:
                logger.debug(f"Failed to get info for custom provider '{name}': {e}")
                providers[name] = {
                    "name": name,
                    "available": False,
                    "error": str(e)
                }
    
    return providers


def _load_builtin_provider(name: str) -> None:
    """Load a built-in provider.
    
    Args:
        name: Provider name
        
    Raises:
        LLMUnsupportedProviderError: If provider cannot be loaded
    """
    if name not in BUILTIN_PROVIDERS:
        raise LLMUnsupportedProviderError(f"'{name}' is not a built-in provider")
    
    provider_config = BUILTIN_PROVIDERS[name]
    
    try:
        # Import the provider module
        module = importlib.import_module(provider_config["module"])
        
        # Get the provider class
        provider_class = getattr(module, provider_config["class"])
        
        # Register the provider
        register_provider(name, provider_class)
        
        logger.info(f"Successfully loaded built-in provider '{name}'")
        
    except ImportError as e:
        error_msg = f"Failed to import provider '{name}': {e}"
        logger.debug(error_msg)
        raise LLMUnsupportedProviderError(
            f"Provider '{name}' is not available - missing dependencies",
            provider_name=name,
            original_exception=e
        )
    except AttributeError as e:
        error_msg = f"Provider class '{provider_config['class']}' not found in module '{provider_config['module']}': {e}"
        logger.error(error_msg)
        raise LLMUnsupportedProviderError(
            f"Provider '{name}' implementation error",
            provider_name=name,
            original_exception=e
        )
    except Exception as e:
        error_msg = f"Unexpected error loading provider '{name}': {e}"
        logger.error(error_msg)
        raise LLMUnsupportedProviderError(
            f"Failed to load provider '{name}'",
            provider_name=name,
            original_exception=e
        )


def _check_dependencies(dependencies: List[str]) -> bool:
    """Check if all dependencies are available.
    
    Args:
        dependencies: List of dependency names
        
    Returns:
        True if all dependencies are available
    """
    for dep in dependencies:
        try:
            importlib.import_module(dep.replace("-", "_"))
        except ImportError:
            logger.debug(f"Missing dependency: {dep}")
            return False
    return True


def clear_registry() -> None:
    """Clear the provider registry.
    
    This function is primarily used for testing purposes.
    """
    global _PROVIDER_REGISTRY, _PROVIDER_AVAILABILITY
    _PROVIDER_REGISTRY.clear()
    _PROVIDER_AVAILABILITY.clear()
    logger.info("Provider registry cleared")


# Auto-discovery of providers on module import
def _auto_discover():
    """Auto-discover providers on module import."""
    logger.debug("Auto-discovering providers...")
    
    # We don't auto-load providers here to avoid import errors
    # Providers will be loaded on-demand when requested
    available_count = 0
    for name in BUILTIN_PROVIDERS:
        try:
            if is_provider_available(name):
                available_count += 1
        except Exception as e:
            logger.debug(f"Failed to check availability for provider '{name}': {e}")
    
    logger.info(f"Provider discovery complete: {available_count}/{len(BUILTIN_PROVIDERS)} built-in providers available")


# Run auto-discovery when module is imported
try:
    _auto_discover()
except Exception as e:
    logger.warning(f"Provider auto-discovery failed: {e}")