"""CLI entry point for PromptMatryoshka.

This module defines the command-line interface for running the PromptMatryoshka pipeline,
configuring options, and managing plugins. It supports running the full multi-stage pipeline
or individual plugins, batch and single prompt processing, and plugin discovery/introspection.

Environment:
    - Loads .env automatically if python-dotenv is installed (for OPENAI_API_KEY, etc.)

Functions:
    - discover_plugins(): Dynamically discovers and loads all available plugins.
    - list_plugins(json_output=False): Lists all available plugins with descriptions.
    - describe_plugin(plugin_name, json_output=False): Shows detailed info for a plugin.
    - main(): Entry point for CLI usage.

Usage:
    python -m promptmatryoshka.cli run --input "Prompt" [--plugin PLUGIN] [--batch] [--output-json]
    python -m promptmatryoshka.cli list-plugins [--json]
    python -m promptmatryoshka.cli describe-plugin <plugin_name> [--json]
"""

# Automatically load .env for environment variables (e.g., OPENAI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure project root is on sys.path for plugin imports before any other imports
import sys
import os
CLI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CLI_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import importlib
import pkgutil
import traceback
import json
import random
import datetime
import time
from typing import List, Dict, Any
from promptmatryoshka.plugins.base import PluginBase
from promptmatryoshka.advbench import AdvBenchLoader, AdvBenchError
from promptmatryoshka.storage import save_json
from promptmatryoshka.config import get_config, Config, ConfigurationError
from promptmatryoshka.llm_factory import get_factory, LLMFactory
from promptmatryoshka.providers import discover_providers, get_provider_info, is_provider_available
from promptmatryoshka.exceptions import (
    LLMError,
    LLMConfigurationError,
    LLMUnsupportedProviderError,
    LLMConnectionError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMValidationError
)

PLUGIN_PACKAGE = "promptmatryoshka.plugins"
# Robustly resolve the plugins directory relative to the project root
CLI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CLI_DIR, ".."))
PLUGIN_PATH = os.path.join(PROJECT_ROOT, "promptmatryoshka", "plugins")

# (sys.path logic moved to top of file)

def handle_llm_error(error: Exception, context: str = "operation") -> None:
    """Handle LLM-related errors with user-friendly messages and suggestions.
    
    Args:
        error: The exception that occurred
        context: Context of where the error occurred
    """
    error_msg = f"Error during {context}: {str(error)}"
    
    if isinstance(error, LLMConfigurationError):
        print(f"❌ Configuration Error: {error.message}", file=sys.stderr)
        if hasattr(error, 'config_key') and error.config_key:
            print(f"   Issue with configuration key: {error.config_key}", file=sys.stderr)
        print("   💡 Suggestions:", file=sys.stderr)
        print("   - Check your configuration file (config.json)", file=sys.stderr)
        print("   - Verify all required settings are present", file=sys.stderr)
        print("   - Run 'promptmatryoshka validate-config' to check configuration", file=sys.stderr)
        
    elif isinstance(error, LLMUnsupportedProviderError):
        print(f"❌ Unsupported Provider: {error.message}", file=sys.stderr)
        if hasattr(error, 'supported_providers') and error.supported_providers:
            print(f"   Available providers: {', '.join(error.supported_providers)}", file=sys.stderr)
        print("   💡 Suggestions:", file=sys.stderr)
        print("   - Run 'promptmatryoshka list-providers' to see available providers", file=sys.stderr)
        print("   - Check if required dependencies are installed", file=sys.stderr)
        
    elif isinstance(error, LLMAuthenticationError):
        print(f"❌ Authentication Error: {error.message}", file=sys.stderr)
        print("   💡 Suggestions:", file=sys.stderr)
        print("   - Check your API key configuration", file=sys.stderr)
        print("   - Verify environment variables are set correctly", file=sys.stderr)
        print("   - Ensure your API key has sufficient permissions", file=sys.stderr)
        
    elif isinstance(error, LLMConnectionError):
        print(f"❌ Connection Error: {error.message}", file=sys.stderr)
        print("   💡 Suggestions:", file=sys.stderr)
        print("   - Check your internet connection", file=sys.stderr)
        print("   - Verify the provider's service status", file=sys.stderr)
        print("   - Try again in a few moments", file=sys.stderr)
        
    elif isinstance(error, LLMRateLimitError):
        print(f"❌ Rate Limit Error: {error.message}", file=sys.stderr)
        if hasattr(error, 'retry_after') and error.retry_after:
            print(f"   Retry after: {error.retry_after} seconds", file=sys.stderr)
        print("   💡 Suggestions:", file=sys.stderr)
        print("   - Wait before retrying", file=sys.stderr)
        print("   - Consider upgrading your API plan", file=sys.stderr)
        print("   - Implement request throttling", file=sys.stderr)
        
    elif isinstance(error, LLMTimeoutError):
        print(f"❌ Timeout Error: {error.message}", file=sys.stderr)
        print("   💡 Suggestions:", file=sys.stderr)
        print("   - Try with a shorter prompt or simpler request", file=sys.stderr)
        print("   - Increase timeout settings in configuration", file=sys.stderr)
        print("   - Check network connectivity", file=sys.stderr)
        
    elif isinstance(error, LLMValidationError):
        print(f"❌ Validation Error: {error.message}", file=sys.stderr)
        if hasattr(error, 'parameter_name') and error.parameter_name:
            print(f"   Parameter: {error.parameter_name}", file=sys.stderr)
        print("   💡 Suggestions:", file=sys.stderr)
        print("   - Check input parameters and values", file=sys.stderr)
        print("   - Verify configuration settings are within valid ranges", file=sys.stderr)
        
    elif isinstance(error, ConfigurationError):
        print(f"❌ Configuration Error: {str(error)}", file=sys.stderr)
        print("   💡 Suggestions:", file=sys.stderr)
        print("   - Check your config.json file syntax", file=sys.stderr)
        print("   - Run 'promptmatryoshka validate-config' for detailed validation", file=sys.stderr)
        print("   - Ensure all required environment variables are set", file=sys.stderr)
        
    elif isinstance(error, LLMError):
        print(f"❌ LLM Error: {error.message}", file=sys.stderr)
        if hasattr(error, 'provider') and error.provider:
            print(f"   Provider: {error.provider}", file=sys.stderr)
        if hasattr(error, 'error_code') and error.error_code:
            print(f"   Error Code: {error.error_code}", file=sys.stderr)
        print("   💡 Suggestions:", file=sys.stderr)
        print("   - Check provider-specific documentation", file=sys.stderr)
        print("   - Verify your configuration and credentials", file=sys.stderr)
        
    else:
        print(f"❌ Unexpected Error: {str(error)}", file=sys.stderr)
        print("   💡 Suggestions:", file=sys.stderr)
        print("   - Run with --debug flag for more details", file=sys.stderr)
        print("   - Check the logs for additional information", file=sys.stderr)
        print("   - Report this issue if it persists", file=sys.stderr)


def discover_plugins():
    """
    Discover all available plugins in the plugins directory.

    Returns:
        dict: Mapping of plugin name to plugin class (subclass of PluginBase).

    Raises:
        Exception: If a plugin module fails to import, the error is raised immediately.

    Debugging:
        If PROMPTMATRYOSHKA_DEBUG=1, detailed logs are written to plugin_discovery_debug.txt
        and printed to stderr for troubleshooting plugin loading issues.
    """
    plugins = {}
    debug = os.environ.get("PROMPTMATRYOSHKA_DEBUG") == "1"
    if debug:
        debug_lines = [
            f"PLUGIN_PATH: {PLUGIN_PATH}",
            f"PLUGIN_PACKAGE: {PLUGIN_PACKAGE}",
            f"Directory contents: {os.listdir(PLUGIN_PATH)}"
        ]
        with open(os.path.join(PROJECT_ROOT, "plugin_discovery_debug.txt"), "a") as dbgfile:
            dbgfile.write("\n".join(debug_lines) + "\n")
        print(f"PLUGIN_PATH: {PLUGIN_PATH}", file=sys.stderr)
        print(f"PLUGIN_PACKAGE: {PLUGIN_PACKAGE}", file=sys.stderr)
        print(f"Directory contents: {os.listdir(PLUGIN_PATH)}", file=sys.stderr)
    for finder, name, ispkg in pkgutil.iter_modules([PLUGIN_PATH]):
        if debug:
            with open(os.path.join(PROJECT_ROOT, "plugin_discovery_debug.txt"), "a") as dbgfile:
                dbgfile.write(f"Found module: {name} (pkg: {ispkg})\n")
            print(f"Found module: {name} (pkg: {ispkg})", file=sys.stderr)
        if name in ("base", "__init__"):
            # Skip base and __init__ modules; not plugins
            continue
        module_name = f"{PLUGIN_PACKAGE}.{name}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            print(f"CRITICAL: Failed to import {module_name}: {e}", file=sys.stderr)
            raise  # Do not continue; fail fast and loud
        # Search for plugin classes (subclasses of PluginBase)
        for attr in dir(module):
            obj = getattr(module, attr)
            if debug:
                print(f"  Checking {attr}: type={type(obj)}", file=sys.stderr)
                print(f"    issubclass={isinstance(obj, type) and issubclass(obj, PluginBase) and obj is not PluginBase}", file=sys.stderr)
            if isinstance(obj, type) and issubclass(obj, PluginBase) and obj is not PluginBase:
                plugins[name] = obj
                if debug:
                    print(f"    --> Registered plugin: {name}", file=sys.stderr)
    if debug:
        with open(os.path.join(PROJECT_ROOT, "plugin_discovery_debug.txt"), "a") as dbgfile:
            dbgfile.write(f"Discovered plugins: {list(plugins.keys())}\n")
        print(f"Discovered plugins: {list(plugins.keys())}", file=sys.stderr)
    return plugins

def list_plugins(json_output=False):
    """
    List all available plugins with their names and short descriptions.

    Args:
        json_output (bool): If True, output as JSON; otherwise, print to stdout.
    """
    plugins = discover_plugins()
    plugin_list = []
    for name, cls in plugins.items():
        desc = cls.__doc__.strip().splitlines()[0] if cls.__doc__ else ""
        plugin_list.append({"name": name, "description": desc})
    if json_output:
        print(json.dumps(plugin_list, indent=2))
    else:
        for p in plugin_list:
            print(f"{p['name']}: {p['description']}")

def describe_plugin(plugin_name, json_output=False):
    """
    Print or return detailed documentation for a specific plugin.

    Args:
        plugin_name (str): Name of the plugin to describe.
        json_output (bool): If True, output as JSON; otherwise, print to stdout.

    Exits:
        1: If the plugin is not found.
    """
    plugins = discover_plugins()
    cls = plugins.get(plugin_name)
    if not cls:
        print(f"Plugin '{plugin_name}' not found.", file=sys.stderr)
        sys.exit(1)
    doc = cls.__doc__ or "(No description)"
    module = sys.modules[cls.__module__]
    mod_doc = module.__doc__ or ""
    result = {
        "plugin": plugin_name,
        "class_doc": doc.strip(),
        "module_doc": mod_doc.strip()
    }
    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print(f"Plugin: {plugin_name}")
        print(doc.strip())
        if mod_doc.strip():
            print("\nModule docstring:\n" + mod_doc.strip())

def run_advbench(args):
    """
    Run AdvBench testing with the specified options.
    
    Args:
        args: Parsed command line arguments for advbench subcommand.
    """
    try:
        # Load AdvBench dataset
        print(f"Loading AdvBench dataset (split: {args.split})...", file=sys.stderr)
        advbench_loader = AdvBenchLoader()
        advbench_loader.load_dataset(split=args.split)
        
        # Get prompts based on options
        if args.random:
            prompts = advbench_loader.get_random_prompts(count=1)
            print(f"Selected 1 random prompt from AdvBench", file=sys.stderr)
        elif args.count:
            prompts = advbench_loader.get_random_prompts(count=args.count)
            print(f"Selected {len(prompts)} random prompts from AdvBench", file=sys.stderr)
        elif args.full:
            prompts = advbench_loader.get_all_prompts()
            print(f"Processing full AdvBench dataset ({len(prompts)} prompts)", file=sys.stderr)
        
        # Discover available plugins
        plugins = discover_plugins()
        
        # Determine which plugins to run
        if args.plugins:
            plugin_names = [name.strip() for name in args.plugins.split(',')]
            # Add judge plugin if requested
            if args.judge and 'judge' not in plugin_names:
                plugin_names.append('judge')
        else:
            # Default pipeline
            plugin_names = ['flipattack', 'logitranslate', 'boost', 'logiattack']
            if args.judge:
                plugin_names.append('judge')
        
        # Validate plugins exist
        missing_plugins = [name for name in plugin_names if name not in plugins]
        if missing_plugins:
            print(f"Error: The following plugins were not found: {', '.join(missing_plugins)}", file=sys.stderr)
            print(f"Available plugins: {', '.join(plugins.keys())}", file=sys.stderr)
            sys.exit(1)
        
        # Create plugin instances
        plugin_instances = []
        for name in plugin_names:
            plugin_instances.append((name, plugins[name]()))
        
        print(f"Running pipeline with plugins: {', '.join(plugin_names)}", file=sys.stderr)
        
        # Process prompts
        results = []
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        
        for i, prompt_data in enumerate(prompts):
            prompt = prompt_data["prompt"]
            print(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}", file=sys.stderr)
            
            # Track the pipeline for this prompt
            prompt_result = {
                "original_prompt": prompt,
                "prompt_metadata": prompt_data,
                "stages": [],
                "timestamp": timestamp,
                "successful": False,
                "error": None
            }
            
            try:
                # Run through pipeline
                current_data = prompt
                for stage_idx, (plugin_name, plugin) in enumerate(plugin_instances):
                    print(f"  Running {plugin_name} (stage {stage_idx+1}/{len(plugin_instances)})...", file=sys.stderr)
                    
                    # Retry logic for plugin execution
                    max_retries = getattr(args, 'max_retries', 3)
                    retry_count = 0
                    last_exception = None
                    stage_result = None
                    
                    while retry_count <= max_retries:
                        try:
                            # Special handling for judge plugin
                            if plugin_name == 'judge':
                                # Judge expects JSON with original_prompt and response
                                judge_input = {
                                    "original_prompt": prompt,
                                    "response": current_data
                                }
                                output = plugin.run(json.dumps(judge_input))
                                # Parse judge output to get judgment
                                judge_data = json.loads(output)
                                stage_result = {
                                    "plugin": plugin_name,
                                    "input": json.dumps(judge_input),
                                    "output": output,
                                    "judgment": judge_data.get("judgment", False),
                                    "successful": True,
                                    "retry_count": retry_count
                                }
                            else:
                                # Regular plugin processing
                                if not isinstance(current_data, str):
                                    current_data = str(current_data)
                                
                                output = plugin.run(current_data)
                                current_data = output
                                
                                stage_result = {
                                    "plugin": plugin_name,
                                    "input": prompt if stage_idx == 0 else "...",  # Only show full input for first stage
                                    "output": output,
                                    "successful": True,
                                    "retry_count": retry_count
                                }
                            
                            # Success - break out of retry loop
                            if retry_count > 0:
                                print(f"  {plugin_name} completed successfully after {retry_count} retries", file=sys.stderr)
                            else:
                                print(f"  {plugin_name} completed successfully", file=sys.stderr)
                            break
                            
                        except Exception as e:
                            last_exception = e
                            retry_count += 1
                            
                            if retry_count <= max_retries:
                                # Calculate exponential backoff delay
                                delay = min(2 ** (retry_count - 1), 10)  # Max 10 seconds
                                print(f"  {plugin_name} failed (attempt {retry_count}/{max_retries + 1}): {str(e)}", file=sys.stderr)
                                print(f"  Retrying in {delay} seconds...", file=sys.stderr)
                                
                                if args.debug:
                                    traceback.print_exc()
                                
                                time.sleep(delay)
                            else:
                                # Max retries exceeded
                                error_msg = f"Error in {plugin_name} (max retries exceeded): {str(e)}"
                                print(f"  {error_msg}", file=sys.stderr)
                                
                                if args.debug:
                                    traceback.print_exc()
                                
                                stage_result = {
                                    "plugin": plugin_name,
                                    "input": prompt if stage_idx == 0 else "...",
                                    "error": str(e),
                                    "successful": False,
                                    "retry_count": retry_count - 1,
                                    "max_retries_exceeded": True
                                }
                                break
                    
                    # Add stage result to prompt results
                    if stage_result:
                        prompt_result["stages"].append(stage_result)
                        
                        # If the stage failed, stop the pipeline
                        if not stage_result.get("successful", False):
                            break
                
                # Check if pipeline completed successfully
                prompt_result["successful"] = all(stage.get("successful", False) for stage in prompt_result["stages"])
                prompt_result["final_output"] = current_data
                
            except Exception as e:
                prompt_result["error"] = str(e)
                print(f"  Fatal error processing prompt: {str(e)}", file=sys.stderr)
                if args.debug:
                    traceback.print_exc()
            
            results.append(prompt_result)
        
        # Generate summary statistics
        total_prompts = len(results)
        successful_prompts = sum(1 for r in results if r["successful"])
        failed_prompts = total_prompts - successful_prompts
        
        # Judge statistics if judge was used
        judge_stats = None
        if args.judge:
            judge_results = []
            for result in results:
                for stage in result["stages"]:
                    if stage.get("plugin") == "judge" and "judgment" in stage:
                        judge_results.append(stage["judgment"])
            
            if judge_results:
                dangerous_count = sum(1 for j in judge_results if j)
                safe_count = len(judge_results) - dangerous_count
                judge_stats = {
                    "total_evaluated": len(judge_results),
                    "dangerous": dangerous_count,
                    "safe": safe_count,
                    "dangerous_rate": dangerous_count / len(judge_results) * 100 if judge_results else 0
                }
        
        # Prepare final results
        final_results = {
            "metadata": {
                "timestamp": timestamp,
                "dataset_split": args.split,
                "total_prompts": total_prompts,
                "successful_prompts": successful_prompts,
                "failed_prompts": failed_prompts,
                "success_rate": successful_prompts / total_prompts * 100 if total_prompts > 0 else 0,
                "plugins_used": plugin_names,
                "judge_enabled": args.judge,
                "judge_stats": judge_stats
            },
            "results": results
        }
        
        # Export results if requested
        if args.export:
            try:
                # Ensure the export path is valid
                export_path = args.export.strip()
                if not export_path:
                    raise ValueError("Export path cannot be empty")
                
                # Create directory if it doesn't exist
                export_dir = os.path.dirname(export_path)
                if export_dir and not os.path.exists(export_dir):
                    os.makedirs(export_dir, exist_ok=True)
                
                save_json(final_results, export_path)
                print(f"Results exported to: {export_path}", file=sys.stderr)
            except Exception as e:
                print(f"Error exporting results: {e}", file=sys.stderr)
                if args.debug:
                    traceback.print_exc()
        
        # Print summary
        print("\n" + "="*50, file=sys.stderr)
        print("AdvBench Testing Summary", file=sys.stderr)
        print("="*50, file=sys.stderr)
        print(f"Dataset Split: {args.split}", file=sys.stderr)
        print(f"Total Prompts: {total_prompts}", file=sys.stderr)
        print(f"Successful: {successful_prompts} ({successful_prompts/total_prompts*100:.1f}%)", file=sys.stderr)
        print(f"Failed: {failed_prompts} ({failed_prompts/total_prompts*100:.1f}%)", file=sys.stderr)
        print(f"Plugins: {', '.join(plugin_names)}", file=sys.stderr)
        
        if judge_stats:
            print(f"\nJudge Evaluation:", file=sys.stderr)
            print(f"  Dangerous: {judge_stats['dangerous']} ({judge_stats['dangerous_rate']:.1f}%)", file=sys.stderr)
            print(f"  Safe: {judge_stats['safe']} ({100-judge_stats['dangerous_rate']:.1f}%)", file=sys.stderr)
        
        if args.export:
            print(f"\nDetailed results saved to: {args.export}", file=sys.stderr)
        
        # Output simplified results to stdout for JSON processing
        print(json.dumps(final_results["metadata"], indent=2))
        
    except AdvBenchError as e:
        print(f"AdvBench Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error in AdvBench testing: {e}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        sys.exit(1)


def list_providers_command(args):
    """List all available providers with their status."""
    try:
        providers = discover_providers()
        
        if args.json:
            print(json.dumps(providers, indent=2))
        else:
            print("Available LLM Providers:")
            print("=" * 50)
            for name, info in providers.items():
                status = "✓ Available" if info.get("available", False) else "✗ Not Available"
                description = info.get("description", "No description")
                print(f"{name:<15} {status:<15} {description}")
                
                if not info.get("available", False) and "error" in info:
                    print(f"                Error: {info['error']}")
    
    except Exception as e:
        handle_llm_error(e, "listing providers")
        sys.exit(1)


def check_provider_command(args):
    """Check a specific provider's configuration and availability."""
    try:
        provider_name = args.provider
        
        # Check if provider exists
        try:
            provider_info = get_provider_info(provider_name)
        except LLMUnsupportedProviderError:
            print(f"Provider '{provider_name}' not found", file=sys.stderr)
            sys.exit(1)
        
        # Check availability
        available = is_provider_available(provider_name)
        
        if args.json:
            result = {
                "provider": provider_name,
                "available": available,
                "info": provider_info
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"Provider: {provider_name}")
            print(f"Status: {'Available' if available else 'Not Available'}")
            print(f"Description: {provider_info.get('description', 'No description')}")
            
            if 'dependencies' in provider_info:
                print(f"Dependencies: {', '.join(provider_info['dependencies'])}")
            
            if not available and 'error' in provider_info:
                print(f"Error: {provider_info['error']}")
    
    except Exception as e:
        handle_llm_error(e, "checking provider")
        sys.exit(1)


def test_provider_command(args):
    """Test a provider's connection and functionality."""
    try:
        provider_name = args.provider
        config = get_config()
        factory = get_factory()
        
        # Get provider configuration
        provider_config = config.get_provider_config(provider_name)
        if not provider_config:
            print(f"No configuration found for provider '{provider_name}'", file=sys.stderr)
            sys.exit(1)
        
        # Create basic LLM configuration for testing
        llm_config = {
            "model": provider_config.default_model,
            "temperature": 0.0,
            "max_tokens": 10
        }
        
        print(f"Testing provider '{provider_name}'...")
        
        # Try to create interface
        try:
            interface = factory.create_interface(provider_name, llm_config)
            print("✓ Provider interface created successfully")
        except Exception as e:
            print(f"✗ Failed to create provider interface: {e}")
            sys.exit(1)
        
        # Try health check if available
        if hasattr(interface, 'health_check'):
            try:
                interface.health_check()
                print("✓ Health check passed")
            except Exception as e:
                print(f"✗ Health check failed: {e}")
        
        print(f"Provider '{provider_name}' is working correctly")
    
    except Exception as e:
        handle_llm_error(e, "testing provider")
        sys.exit(1)


def list_profiles_command(args):
    """List all available profiles."""
    try:
        config = get_config()
        profiles = config.get_available_profiles()
        
        if args.json:
            profile_data = {}
            for profile_name in profiles:
                profile_config = config.get_profile_config(profile_name)
                if profile_config:
                    profile_data[profile_name] = profile_config.model_dump()
            print(json.dumps(profile_data, indent=2))
        else:
            print("Available Configuration Profiles:")
            print("=" * 50)
            for profile_name in profiles:
                profile_config = config.get_profile_config(profile_name)
                if profile_config:
                    print(f"{profile_name:<20} {profile_config.provider:<12} {profile_config.model}")
                    if profile_config.description:
                        print(f"                     {profile_config.description}")
    
    except Exception as e:
        handle_llm_error(e, "listing profiles")
        sys.exit(1)


def show_profile_command(args):
    """Show detailed information about a specific profile."""
    try:
        config = get_config()
        profile_name = args.profile
        
        profile_config = config.get_profile_config(profile_name)
        if not profile_config:
            print(f"Profile '{profile_name}' not found", file=sys.stderr)
            sys.exit(1)
        
        if args.json:
            print(json.dumps(profile_config.model_dump(), indent=2))
        else:
            print(f"Profile: {profile_name}")
            print("=" * 50)
            print(f"Provider: {profile_config.provider}")
            print(f"Model: {profile_config.model}")
            print(f"Temperature: {profile_config.temperature}")
            print(f"Max Tokens: {profile_config.max_tokens}")
            print(f"Top P: {profile_config.top_p}")
            print(f"Frequency Penalty: {profile_config.frequency_penalty}")
            print(f"Presence Penalty: {profile_config.presence_penalty}")
            print(f"Request Timeout: {profile_config.request_timeout}")
            
            if profile_config.description:
                print(f"Description: {profile_config.description}")
    
    except Exception as e:
        handle_llm_error(e, "showing profile")
        sys.exit(1)


def validate_config_command(args):
    """Validate the current configuration."""
    try:
        config = get_config()
        
        print("Validating configuration...")
        
        # Validate configuration
        try:
            config.validate_configuration()
            print("✓ Configuration validation passed")
        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")
            sys.exit(1)
        
        # Check provider availability
        providers = config.get_available_providers()
        available_providers = []
        unavailable_providers = []
        
        for provider_name in providers:
            if is_provider_available(provider_name):
                available_providers.append(provider_name)
            else:
                unavailable_providers.append(provider_name)
        
        print(f"✓ {len(available_providers)} providers available: {', '.join(available_providers)}")
        if unavailable_providers:
            print(f"✗ {len(unavailable_providers)} providers unavailable: {', '.join(unavailable_providers)}")
        
        # Check profiles
        profiles = config.get_available_profiles()
        print(f"✓ {len(profiles)} profiles configured: {', '.join(profiles)}")
        
        print("Configuration validation complete")
    
    except Exception as e:
        handle_llm_error(e, "validating configuration")
        sys.exit(1)


def show_config_command(args):
    """Show current configuration with masked secrets."""
    try:
        config = get_config()
        config_dict = config.to_dict()
        
        # Mask sensitive information
        def mask_secrets(data):
            if isinstance(data, dict):
                masked_data = {}
                for key, value in data.items():
                    if 'api_key' in key.lower() or 'secret' in key.lower() or 'password' in key.lower():
                        if isinstance(value, str) and value:
                            masked_data[key] = value[:8] + "*" * max(0, len(value) - 8)
                        else:
                            masked_data[key] = value
                    else:
                        masked_data[key] = mask_secrets(value)
                return masked_data
            elif isinstance(data, list):
                return [mask_secrets(item) for item in data]
            else:
                return data
        
        masked_config = mask_secrets(config_dict)
        
        if args.json:
            print(json.dumps(masked_config, indent=2))
        else:
            print("Current Configuration:")
            print("=" * 50)
            print(json.dumps(masked_config, indent=2))
    
    except Exception as e:
        handle_llm_error(e, "showing configuration")
        sys.exit(1)


def config_health_command(args):
    """Check configuration health and provider availability."""
    try:
        config = get_config()
        factory = get_factory()
        
        print("Checking configuration health...")
        print("=" * 50)
        
        # Check configuration validity
        try:
            config.validate_configuration()
            print("✓ Configuration is valid")
        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")
            return
        
        # Check each provider
        providers = config.get_available_providers()
        for provider_name in providers:
            provider_config = config.get_provider_config(provider_name)
            if provider_config:
                try:
                    # Test basic interface creation
                    llm_config = {
                        "model": provider_config.default_model,
                        "temperature": 0.0,
                        "max_tokens": 10
                    }
                    interface = factory.create_interface(provider_name, llm_config)
                    print(f"✓ Provider '{provider_name}' is healthy")
                except Exception as e:
                    print(f"✗ Provider '{provider_name}' has issues: {e}")
        
        # Check profiles
        profiles = config.get_available_profiles()
        for profile_name in profiles:
            try:
                interface = factory.create_from_profile(profile_name)
                print(f"✓ Profile '{profile_name}' is healthy")
            except Exception as e:
                print(f"✗ Profile '{profile_name}' has issues: {e}")
        
        print("Configuration health check complete")
    
    except Exception as e:
        handle_llm_error(e, "checking configuration health")
        sys.exit(1)


def main():
    """
    Main CLI entry point for PromptMatryoshka.

    Parses command-line arguments and dispatches to the appropriate subcommand:
        - run: Run the full pipeline or a specific plugin on input(s).
        - list-plugins: List all available plugins.
        - describe-plugin: Show detailed info for a plugin.

    Exits:
        1: On fatal errors or invalid plugin names.
    """
    parser = argparse.ArgumentParser(
        description="PromptMatryoshka CLI: Run the pipeline or individual plugins."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run the full pipeline or a specific plugin.")
    run_parser.add_argument(
        "--plugin", "-p", type=str, default=None,
        help="Name of the plugin to run (default: run the full pipeline)"
    )
    run_parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Input prompt (string, or @filename for file, or '-' for stdin)"
    )
    run_parser.add_argument(
        "--batch", "-b", action="store_true",
        help="Batch mode: treat input as a file with one prompt per line"
    )
    run_parser.add_argument(
        "--output-json", action="store_true",
        help="Output results as JSON"
    )
    run_parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    run_parser.add_argument(
        "--provider", type=str, default=None,
        help="Provider to use for LLM operations (e.g., openai, anthropic)"
    )
    run_parser.add_argument(
        "--profile", type=str, default=None,
        help="Configuration profile to use"
    )

    # list-plugins subcommand
    list_parser = subparsers.add_parser("list-plugins", help="List all available plugins.")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # describe-plugin subcommand
    describe_parser = subparsers.add_parser("describe-plugin", help="Describe a plugin.")
    describe_parser.add_argument("plugin_name", type=str, help="Name of the plugin to describe.")
    describe_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # advbench subcommand
    advbench_parser = subparsers.add_parser("advbench", help="Test with AdvBench dataset.")
    advbench_group = advbench_parser.add_mutually_exclusive_group(required=True)
    advbench_group.add_argument("--random", action="store_true", help="Test with a single random prompt from AdvBench")
    advbench_group.add_argument("--count", type=int, help="Test with N random prompts from AdvBench")
    advbench_group.add_argument("--full", action="store_true", help="Test with the full AdvBench dataset")
    
    advbench_parser.add_argument("--plugins", type=str, help="Comma-separated list of plugins to run (e.g., 'logitranslate,logiattack')")
    advbench_parser.add_argument("--judge", action="store_true", help="Enable automatic evaluation with the judge plugin")
    advbench_parser.add_argument("--export", type=str, help="Export results to a file for manual evaluation (JSON format)")
    advbench_parser.add_argument("--split", type=str, default="harmful_behaviors",
                                choices=["harmful_behaviors", "harmful_strings"],
                                help="Choose between 'harmful_behaviors' and 'harmful_strings' splits")
    advbench_parser.add_argument("--max-retries", type=int, default=3,
                                help="Maximum number of retries for failed plugins (default: 3)")
    advbench_parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Provider management commands
    list_providers_parser = subparsers.add_parser("list-providers", help="List all available LLM providers")
    list_providers_parser.add_argument("--json", action="store_true", help="Output as JSON")

    check_provider_parser = subparsers.add_parser("check-provider", help="Check a specific provider's configuration")
    check_provider_parser.add_argument("provider", type=str, help="Provider name to check")
    check_provider_parser.add_argument("--json", action="store_true", help="Output as JSON")

    test_provider_parser = subparsers.add_parser("test-provider", help="Test a provider's connection and functionality")
    test_provider_parser.add_argument("provider", type=str, help="Provider name to test")

    # Profile management commands
    list_profiles_parser = subparsers.add_parser("list-profiles", help="List all available configuration profiles")
    list_profiles_parser.add_argument("--json", action="store_true", help="Output as JSON")

    show_profile_parser = subparsers.add_parser("show-profile", help="Show detailed information about a profile")
    show_profile_parser.add_argument("profile", type=str, help="Profile name to show")
    show_profile_parser.add_argument("--json", action="store_true", help="Output as JSON")

    validate_config_parser = subparsers.add_parser("validate-config", help="Validate the current configuration")

    # Configuration management commands
    show_config_parser = subparsers.add_parser("show-config", help="Show current configuration")
    show_config_parser.add_argument("--json", action="store_true", help="Output as JSON")

    config_health_parser = subparsers.add_parser("config-health", help="Check configuration health and provider availability")

    args = parser.parse_args()

    if args.command == "list-plugins":
        list_plugins(json_output=getattr(args, "json", False))
    elif args.command == "describe-plugin":
        describe_plugin(args.plugin_name, json_output=getattr(args, "json", False))
    elif args.command == "advbench":
        run_advbench(args)
    elif args.command == "list-providers":
        list_providers_command(args)
    elif args.command == "check-provider":
        check_provider_command(args)
    elif args.command == "test-provider":
        test_provider_command(args)
    elif args.command == "list-profiles":
        list_profiles_command(args)
    elif args.command == "show-profile":
        show_profile_command(args)
    elif args.command == "validate-config":
        validate_config_command(args)
    elif args.command == "show-config":
        show_config_command(args)
    elif args.command == "config-health":
        config_health_command(args)
    elif args.command == "run":
        try:
            # --- Input reading and normalization ---
            if args.input is None or args.input == "-":
                # Read from stdin (interactive or piped input)
                if sys.stdin.isatty():
                    print("Reading input from stdin (end with Ctrl-D):", file=sys.stderr)
                input_data = sys.stdin.read()
                # If batch mode, treat each line as a separate prompt
                inputs = [line.strip() for line in input_data.splitlines() if line.strip()] if args.batch else [input_data.strip()]
            elif args.input.startswith("@"):
                # Read from file (filename after '@')
                fname = args.input[1:]
                with open(fname, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                inputs = [line.strip() for line in lines if line.strip()] if args.batch else ["".join(lines).strip()]
            else:
                # Direct string input from CLI
                inputs = [args.input.strip()]

            # --- Plugin discovery and execution with new multi-provider system ---
            try:
                from promptmatryoshka.core import PromptMatryoshka
                
                # Create PromptMatryoshka instance with provider/profile options
                pm = PromptMatryoshka(
                    provider=getattr(args, 'provider', None),
                    profile=getattr(args, 'profile', None)
                )
                
                results = []
                
                if args.plugin:
                    # Run a single plugin by name using the new system
                    plugin_name = args.plugin
                    for inp in inputs:
                        try:
                            output = pm.jailbreak(
                                inp,
                                plugins=[plugin_name],
                                provider=getattr(args, 'provider', None),
                                profile=getattr(args, 'profile', None)
                            )
                            results.append({"input": inp, "output": output, "plugin": plugin_name})
                        except Exception as e:
                            # Capture and report plugin errors per input
                            if args.debug:
                                traceback.print_exc()
                            results.append({"input": inp, "error": str(e), "plugin": plugin_name})
                else:
                    # Run the full pipeline using the new system
                    for inp in inputs:
                        try:
                            print(f"Starting pipeline for input: {inp[:50]}{'...' if len(inp) > 50 else ''}", file=sys.stderr)
                            
                            output = pm.jailbreak(
                                inp,
                                provider=getattr(args, 'provider', None),
                                profile=getattr(args, 'profile', None)
                            )
                            
                            # Get pipeline info for detailed results
                            pipeline_info = pm.get_pipeline_info()
                            stage_results = []
                            
                            # Create stage results showing the pipeline flow
                            for plugin_info in pipeline_info["plugins"]:
                                stage_results.append({
                                    "plugin": plugin_info["name"],
                                    "output": output if plugin_info == pipeline_info["plugins"][-1] else "...",
                                    "category": plugin_info["category"]
                                })
                            
                            results.append({
                                "input": inp,
                                "output": output,
                                "stages": stage_results,
                                "pipeline_info": pipeline_info
                            })
                            
                        except Exception as e:
                            error_msg = f"Error in pipeline: {str(e)}"
                            print(f"  {error_msg}", file=sys.stderr)
                            if args.debug:
                                traceback.print_exc()
                            # Even for errors, include stages field for consistent JSON structure
                            results.append({
                                "input": inp,
                                "error": str(e),
                                "stages": []  # Empty stages array for errors
                            })
                            
            except ImportError:
                # Fallback to old system if new system is not available
                print("Warning: Using legacy plugin system (new multi-provider system not available)", file=sys.stderr)
                
                plugins = discover_plugins()
                results = []
                if args.plugin:
                    # Run a single plugin by name
                    plugin_name = args.plugin
                    if plugin_name not in plugins:
                        print(f"Plugin '{plugin_name}' not found.", file=sys.stderr)
                        sys.exit(1)
                    plugin_cls = plugins[plugin_name]
                    plugin = plugin_cls()
                    for inp in inputs:
                        try:
                            output = plugin.run(inp)
                            results.append({"input": inp, "output": output, "plugin": plugin_name})
                        except Exception as e:
                            # Capture and report plugin errors per input
                            if args.debug:
                                traceback.print_exc()
                            results.append({"input": inp, "error": str(e), "plugin": plugin_name})
                else:
                    # Run the full pipeline: explicit research order (FlipAttack → LogiTranslate → BOOST → LogiAttack)
                    plugin_order = ['flipattack', 'logitranslate', 'boost', 'logiattack']
                    pipeline = []
                    for name in plugin_order:
                        if name in plugins:
                            pipeline.append(plugins[name]())
                        else:
                            print(f"Warning: Plugin '{name}' not found in discovered plugins", file=sys.stderr)
                    for inp in inputs:
                        stage_results = []
                        data = inp
                        print(f"Starting pipeline for input: {inp[:50]}{'...' if len(inp) > 50 else ''}", file=sys.stderr)
                        
                        for i, plugin in enumerate(pipeline):
                            plugin_name = plugin.__class__.__name__
                            try:
                                # Log input validation
                                if not isinstance(data, str):
                                    print(f"Warning: Plugin {plugin_name} received non-string input: {type(data)}", file=sys.stderr)
                                    if hasattr(data, '__str__'):
                                        data = str(data)
                                    else:
                                        raise ValueError(f"Plugin {plugin_name} cannot process input of type {type(data)}")
                                
                                print(f"  Running {plugin_name} (stage {i+1}/{len(pipeline)})...", file=sys.stderr)
                                data = plugin.run(data)
                                
                                # Log output validation
                                if not isinstance(data, str):
                                    print(f"Warning: Plugin {plugin_name} returned non-string output: {type(data)}", file=sys.stderr)
                                    if hasattr(data, '__str__'):
                                        data = str(data)
                                    else:
                                        raise ValueError(f"Plugin {plugin_name} returned invalid output type {type(data)}")
                                
                                stage_results.append({"plugin": plugin_name, "output": data})
                                print(f"  {plugin_name} completed successfully", file=sys.stderr)
                                
                            except Exception as e:
                                # Capture and report error with context
                                error_msg = f"Error in {plugin_name}: {str(e)}"
                                print(f"  {error_msg}", file=sys.stderr)
                                if args.debug:
                                    print(f"  Input to failed plugin: {data[:200]}{'...' if len(str(data)) > 200 else ''}", file=sys.stderr)
                                    traceback.print_exc()
                                stage_results.append({"plugin": plugin_name, "error": str(e)})
                                break
                        # Always include stages field for consistent JSON structure
                        results.append({
                            "input": inp,
                            "stages": stage_results,
                            "output": data if stage_results and "error" not in stage_results[-1] else None
                        })
            # --- Output formatting ---
            if args.output_json:
                print(json.dumps(results, indent=2, ensure_ascii=False))
            else:
                for res in results:
                    if "error" in res:
                        print(f"ERROR: {res['error']} (plugin: {res.get('plugin', '')})")
                    elif "output" in res:
                        print(f"Output [{res.get('plugin', '')}]: {res['output']}")
                    elif "stages" in res:
                        print(f"Pipeline result for input: {res['input']}")
                        for stage in res["stages"]:
                            if "error" in stage:
                                print(f"  ERROR in {stage['plugin']}: {stage['error']}")
                            else:
                                print(f"  {stage['plugin']}: {stage['output']}")
        except Exception as e:
            handle_llm_error(e, "running command")
            if getattr(args, "debug", False):
                traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
