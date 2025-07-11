"""Storage utilities for PromptMatryoshka.

Handles reading and writing pipeline data, logs, and experiment results
to JSON files for reproducibility and analysis.

Functions:
    save_json(data: dict, path: str): Saves data as JSON.
    load_json(path: str) -> dict: Loads data from JSON file.
"""

import json
import os
import tempfile

def save_json(data, path):
    """
    Saves a dictionary as a JSON file at the given path.
    Uses atomic write to avoid partial file corruption.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(path)) as tf:
        json.dump(data, tf, indent=2, ensure_ascii=False)
        tempname = tf.name
    os.replace(tempname, path)

def load_json(path):
    """
    Loads a dictionary from a JSON file at the given path.
    Returns the loaded dictionary.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)