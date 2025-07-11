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
from promptmatryoshka.plugins.base import PluginBase

PLUGIN_PACKAGE = "promptmatryoshka.plugins"
# Robustly resolve the plugins directory relative to the project root
CLI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CLI_DIR, ".."))
PLUGIN_PATH = os.path.join(PROJECT_ROOT, "promptmatryoshka", "plugins")

# (sys.path logic moved to top of file)

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

    # list-plugins subcommand
    list_parser = subparsers.add_parser("list-plugins", help="List all available plugins.")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # describe-plugin subcommand
    describe_parser = subparsers.add_parser("describe-plugin", help="Describe a plugin.")
    describe_parser.add_argument("plugin_name", type=str, help="Name of the plugin to describe.")
    describe_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.command == "list-plugins":
        list_plugins(json_output=getattr(args, "json", False))
    elif args.command == "describe-plugin":
        describe_plugin(args.plugin_name, json_output=getattr(args, "json", False))
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

            # --- Plugin discovery and execution ---
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
                    results.append({"input": inp, "stages": stage_results})
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
            print(f"Fatal error: {e}", file=sys.stderr)
            if getattr(args, "debug", False):
                traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
