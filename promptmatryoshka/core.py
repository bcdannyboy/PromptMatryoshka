"""Core engine for PromptMatryoshka.

Coordinates the multi-stage jailbreak pipeline, manages plugin loading,
and exposes the main API for running attacks.

Classes:
    PromptMatryoshka: Main orchestrator class (stub).
"""

from promptmatryoshka.plugins.flipattack import FlipAttackPlugin
from promptmatryoshka.plugins.boost import BoostPlugin
# Other plugins can be imported similarly when implemented

class PromptMatryoshka:
    """
    Orchestrates the pipeline: loads plugins, manages data flow,
    and exposes the main jailbreak() API.

    Methods:
        jailbreak(prompt: str, stages=None) -> str
            Runs the prompt through all pipeline stages and returns the result.
    """

    def __init__(self, stages=None):
        """
        Args:
            stages (list): List of plugin instances to run in sequence.
                           If None, uses default pipeline [FlipAttackPlugin].
        """
        if stages is None:
            self.stages = [FlipAttackPlugin()]
        else:
            self.stages = stages

    def jailbreak(self, prompt, stages=None):
        """
        Runs the prompt through all pipeline stages and returns the result.

        Args:
            prompt (str): The input prompt to process.
            stages (list, optional): Override the pipeline with a custom list of plugin instances.

        Returns:
            str: The final output after all pipeline stages.
        """
        pipeline = stages if stages is not None else self.stages
        data = prompt
        for stage in pipeline:
            data = stage.run(data)
        return data