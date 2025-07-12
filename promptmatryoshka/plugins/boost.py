"""BOOST plugin for PromptMatryoshka.

This module implements the BOOST technique: appending end-of-sequence (EOS) tokens
to prompts to manipulate LLM refusal boundaries and increase compliance. It supports
multiple adversarial modes (obfuscation, dynamic positioning) for edge-case testing.

Classes:
    BoostPlugin: Plugin implementation for the BOOST pipeline stage.

Usage:
    plugin = BoostPlugin(mode="append")
    output = plugin.run(prompt)
"""

from .base import PluginBase
from promptmatryoshka.logging_utils import get_logger
from promptmatryoshka.storage import save_json
import os
import uuid
from datetime import datetime
import random

class BoostPlugin(PluginBase):
    """
    BOOST plugin: appends EOS tokens to input prompt.

    This plugin manipulates the prompt by adding EOS tokens in various ways to
    probe LLM refusal boundaries. All transformation stages are logged and stored
    as JSON for reproducibility and analysis.

    Modes:
        - "append": Appends EOS tokens to the end of the prompt.
        - "obfuscate": Appends obfuscated EOS tokens (e.g., with spaces, case changes).
        - "dynamic": Inserts EOS tokens at specified or random positions.

    Args:
        storage_dir (str): Directory to store transformation results.
        num_eos (int): Number of EOS tokens to append/insert.
        eos_token (str): EOS token string.
        mode (str): One of 'append', 'obfuscate', 'dynamic'.
        obfuscate_ops (list): Obfuscation operations for 'obfuscate' mode.
        dynamic_spots (list): Insertion indices for 'dynamic' mode.
    """
    
    # Plugin metadata
    PLUGIN_CATEGORY = "mutation"
    PLUGIN_REQUIRES = []
    PLUGIN_CONFLICTS = []  # Can work together with other prompt transformations
    PLUGIN_PROVIDES = ["eos_enhanced_prompt"]

    def __init__(
        self,
        storage_dir="boost_results",
        num_eos=5,
        eos_token="</s>",
        mode="append",  # "append", "obfuscate", "dynamic"
        obfuscate_ops=None,
        dynamic_spots=None
    ):
        """
        Initialize the BoostPlugin.

        Args:
            storage_dir (str): Directory to store transformation results.
            num_eos (int): Number of EOS tokens to append/insert.
            eos_token (str): EOS token string.
            mode (str): 'append' (default), 'obfuscate', or 'dynamic'.
            obfuscate_ops (list): List of obfuscation operations (for 'obfuscate' mode).
            dynamic_spots (list): List of insertion indices (for 'dynamic' mode).
        """
        self.logger = get_logger("BoostPlugin")
        self.storage_dir = storage_dir
        self.num_eos = num_eos
        self.eos_token = eos_token
        self.mode = mode
        self.obfuscate_ops = obfuscate_ops or []
        self.dynamic_spots = dynamic_spots or []

    def run(self, input_data):
        """
        Apply the BOOST transformation to the input prompt.

        This method selects the transformation mode, logs all steps, and stores
        the result as a JSON file for reproducibility.

        Args:
            input_data (str): The prompt to transform.

        Returns:
            str: The BOOSTed prompt.

        Raises:
            Exception: On transformation or storage failure.
        """
        self.logger.debug(f"BOOST input: {input_data!r}")
        try:
            # Select transformation mode
            if self.mode == "append":
                boosted, stages = self._append_eos(input_data)
            elif self.mode == "obfuscate":
                boosted, stages = self._obfuscate_eos(input_data)
            elif self.mode == "dynamic":
                boosted, stages = self._dynamic_eos(input_data)
            else:
                raise ValueError(f"Unknown BOOST mode: {self.mode}")

            self.logger.info(f"BOOST ({self.mode}) output: {boosted!r}")

            # Store transformation result as JSON for auditability
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "input": input_data,
                "mode": self.mode,
                "num_eos": self.num_eos,
                "eos_token": self.eos_token,
                "stages": stages,
                "output": boosted,
            }
            os.makedirs(self.storage_dir, exist_ok=True)
            outpath = os.path.join(self.storage_dir, f"boost_{uuid.uuid4().hex}.json")
            save_json(result, outpath)
            self.logger.debug(f"BOOST result saved to {outpath}")

            return boosted
        except Exception as e:
            self.logger.error(f"BOOST error: {e}", exc_info=True)
            raise

    def _append_eos(self, text):
        """
        Append self.num_eos EOS tokens to the end of the prompt.

        Args:
            text (str): The input prompt.

        Returns:
            tuple: (boosted_prompt, [stage_dict])
        """
        stage = {
            "operation": "append_eos",
            "num_eos": self.num_eos,
            "eos_token": self.eos_token,
            "before": text,
        }
        boosted = text + (self.eos_token * self.num_eos)
        stage["after"] = boosted
        return boosted, [stage]

    def _obfuscate_eos(self, text):
        """
        Append obfuscated EOS tokens to the prompt for adversarial/boundary testing.

        Args:
            text (str): The input prompt.

        Returns:
            tuple: (boosted_prompt, [stage_dict])
        """
        obfuscated_tokens = [
            self._obfuscate_token(self.eos_token, op)
            for op in (self.obfuscate_ops or ["space", "case", "leet", "special"] * self.num_eos)
        ]
        stage = {
            "operation": "obfuscate_eos",
            "num_eos": self.num_eos,
            "eos_token": self.eos_token,
            "obfuscated_tokens": obfuscated_tokens,
            "before": text,
        }
        boosted = text + "".join(obfuscated_tokens[:self.num_eos])
        stage["after"] = boosted
        return boosted, [stage]

    def _obfuscate_token(self, token, op):
        """
        Apply a simple obfuscation operation to an EOS token.

        Args:
            token (str): The EOS token string.
            op (str): The obfuscation operation ("space", "case", "leet", "special").

        Returns:
            str: The obfuscated token.
        """
        if op == "space":
            # Insert a space at a random position
            idx = random.randint(0, len(token))
            return token[:idx] + " " + token[idx:]
        elif op == "case":
            # Randomly change the case of one character
            idx = random.randint(0, len(token)-1)
            c = token[idx]
            if c.islower():
                c = c.upper()
            elif c.isupper():
                c = c.lower()
            return token[:idx] + c + token[idx+1:]
        elif op == "leet":
            # Substitute characters with leet equivalents
            mapping = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$"}
            out = ""
            for c in token:
                out += mapping.get(c.lower(), c)
            return out
        elif op == "special":
            # Insert a random special character
            specials = ['_', '.', '-', '=', '+', '*', '/', '#', '$', '&', '%', '!', '?']
            idx = random.randint(0, len(token))
            return token[:idx] + random.choice(specials) + token[idx:]
        else:
            # No obfuscation
            return token

    def _dynamic_eos(self, text):
        """
        Insert EOS tokens at specified or random positions in the prompt.

        Args:
            text (str): The input prompt.

        Returns:
            tuple: (boosted_prompt, [stage_dict])
        """
        stage = {
            "operation": "dynamic_eos",
            "num_eos": self.num_eos,
            "eos_token": self.eos_token,
            "dynamic_spots": self.dynamic_spots,
            "before": text,
        }
        tokens = list(text)
        spots = self.dynamic_spots
        if not spots:
            # Randomly select insertion points if not specified
            spots = sorted(random.sample(range(len(tokens) + 1), min(self.num_eos, len(tokens) + 1)))
        for i, idx in enumerate(spots[:self.num_eos]):
            tokens.insert(idx + i, self.eos_token)
        boosted = "".join(tokens)
        stage["after"] = boosted
        return boosted, [stage]