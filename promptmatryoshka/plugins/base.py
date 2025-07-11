"""Base plugin interface for PromptMatryoshka plugins.

This module defines the standard interface for all attack stage plugins.
All plugins must inherit from PluginBase and implement the run() method.

Plugin contract:
    - Each plugin processes a string input and returns a string output.
    - Plugins are designed to be composable in a multi-stage pipeline.

Usage:
    class MyPlugin(PluginBase):
        def run(self, input_data):
            # Transform input_data
            return output_data
"""

class PluginBase:
    """
    Abstract base class for all PromptMatryoshka plugins.

    All plugins must implement the run() method, which takes a string input
    and returns a string output. This interface is used by the pipeline
    orchestrator to invoke each stage.

    Methods:
        run(input_data: str) -> str
            Processes input and returns output for the pipeline.

    Example:
        class MyPlugin(PluginBase):
            def run(self, input_data):
                # ... transform input_data ...
                return output_data
    """
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