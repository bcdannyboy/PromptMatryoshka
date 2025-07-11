"""LLM interface abstraction for PromptMatryoshka.

Defines a standard interface for interacting with different LLM APIs or backends.
Handles model invocation, authentication, and response parsing.

Classes:
    LLMInterface: Abstract base class for LLM interactions.
"""

class LLMInterface:
    """
    Abstracts LLM API calls.

    Methods:
        generate(prompt: str) -> str
            Sends prompt to the LLM and returns the generated response.
    """
    pass