"""Logging utilities for PromptMatryoshka.

Configures robust debug logging for all modules and plugins.
Provides helper functions for consistent log formatting and levels.

Functions:
    setup_logging(): Initializes logging configuration.
    get_logger(name: str): Returns a logger instance for a given module.
"""

import logging

def setup_logging():
    """
    Configures logging for the application.
    Sets up a consistent log format, level, and stream handler.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def get_logger(name):
    """
    Returns a logger instance for the given module or plugin name.
    """
    return logging.getLogger(name)