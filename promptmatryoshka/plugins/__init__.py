"""Plugins subpackage for PromptMatryoshka.

Contains all attack stage plugins and plugin base class/interface.
"""

from .base import PluginBase
from .boost import BoostPlugin
from .flipattack import FlipAttackPlugin
from .logiattack import LogiAttackPlugin
from .logitranslate import LogiTranslatePlugin
from .judge import JudgePlugin

__all__ = [
    'PluginBase',
    'BoostPlugin',
    'FlipAttackPlugin',
    'LogiAttackPlugin',
    'LogiTranslatePlugin',
    'JudgePlugin'
]