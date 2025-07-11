"""AdvBench dataset loading and management for PromptMatryoshka.

This module provides functionality to load and manage the AdvBench dataset
for adversarial prompt testing, including support for different splits,
random sampling, and local caching.

Classes:
    AdvBenchLoader: Main class for loading and managing AdvBench data.
    AdvBenchError: Custom exception for AdvBench-related errors.

Functions:
    load_advbench_dataset(split="harmful_behaviors"): Load AdvBench dataset.
    get_random_prompts(dataset, count=1): Get random prompts from dataset.
"""

import os
import random
import json
import datetime
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from promptmatryoshka.logging_utils import get_logger
from promptmatryoshka.storage import save_json

logger = get_logger("AdvBench")


class AdvBenchError(Exception):
    """Raised when AdvBench operations fail."""
    pass


class AdvBenchLoader:
    """
    AdvBench dataset loader and manager.
    
    Provides functionality to load the AdvBench dataset, handle different splits,
    cache data locally, and provide random sampling capabilities.
    """
    
    def __init__(self, cache_dir: str = "advbench_cache"):
        """
        Initialize the AdvBench loader.
        
        Args:
            cache_dir (str): Directory to cache dataset locally.
        """
        self.cache_dir = cache_dir
        self.dataset = None
        self.current_split = None
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Ensure the cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Created AdvBench cache directory: {self.cache_dir}")
    
    def load_dataset(self, split: str = "harmful_behaviors", force_reload: bool = False) -> Dict[str, Any]:
        """
        Load the AdvBench dataset.
        
        Args:
            split (str): Dataset split to load ("harmful_behaviors" or "harmful_strings").
                        Note: The actual dataset only has "train" split, but we support
                        these names for compatibility.
            force_reload (bool): Force reload from remote even if cached.
            
        Returns:
            Dict[str, Any]: Dataset with metadata.
            
        Raises:
            AdvBenchError: If dataset loading fails.
        """
        if split not in ["harmful_behaviors", "harmful_strings"]:
            raise AdvBenchError(f"Invalid split '{split}'. Must be 'harmful_behaviors' or 'harmful_strings'.")
        
        cache_file = os.path.join(self.cache_dir, f"advbench_{split}.json")
        
        # Try to load from cache first
        if not force_reload and os.path.exists(cache_file):
            try:
                logger.info(f"Loading AdvBench dataset from cache: {cache_file}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                self.dataset = cached_data['dataset']
                self.current_split = split
                logger.info(f"Loaded {len(self.dataset)} entries from cache")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}. Will reload from remote.")
        
        # Load from remote
        try:
            logger.info(f"Loading AdvBench dataset from remote (using 'train' split)")
            dataset = load_dataset("walledai/AdvBench", split="train")
            
            # Convert to list of dictionaries for easier handling
            dataset_list = []
            for item in dataset:
                # AdvBench has 'prompt' and 'target' fields
                if "prompt" in item:
                    dataset_list.append({
                        "prompt": item["prompt"],
                        "target": item.get("target", ""),
                        "original_item": item
                    })
                else:
                    # Fallback for different dataset structures
                    dataset_list.append({
                        "prompt": str(item),
                        "target": "",
                        "original_item": item
                    })
            
            # For now, we treat both split names as the same dataset
            # In the future, we could potentially filter or split the data
            self.dataset = dataset_list
            self.current_split = split
            
            # Cache the dataset
            cache_data = {
                "dataset": dataset_list,
                "split": split,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "count": len(dataset_list)
            }
            
            save_json(cache_data, cache_file)
            logger.info(f"Cached AdvBench dataset ({len(dataset_list)} entries) to {cache_file}")
            
            return cache_data
            
        except Exception as e:
            raise AdvBenchError(f"Failed to load AdvBench dataset: {e}")
    
    def get_random_prompts(self, count: int = 1) -> List[Dict[str, Any]]:
        """
        Get random prompts from the loaded dataset.
        
        Args:
            count (int): Number of random prompts to return.
            
        Returns:
            List[Dict[str, Any]]: List of random prompts with metadata.
            
        Raises:
            AdvBenchError: If dataset is not loaded or count is invalid.
        """
        if self.dataset is None:
            raise AdvBenchError("Dataset not loaded. Call load_dataset() first.")
        
        if count <= 0:
            raise AdvBenchError("Count must be positive.")
        
        if count > len(self.dataset):
            logger.warning(f"Requested {count} prompts but dataset only has {len(self.dataset)}. Returning all.")
            count = len(self.dataset)
        
        selected = random.sample(self.dataset, count)
        logger.info(f"Selected {len(selected)} random prompts from AdvBench dataset")
        
        return selected
    
    def get_all_prompts(self) -> List[Dict[str, Any]]:
        """
        Get all prompts from the loaded dataset.
        
        Returns:
            List[Dict[str, Any]]: All prompts with metadata.
            
        Raises:
            AdvBenchError: If dataset is not loaded.
        """
        if self.dataset is None:
            raise AdvBenchError("Dataset not loaded. Call load_dataset() first.")
        
        logger.info(f"Returning all {len(self.dataset)} prompts from AdvBench dataset")
        return self.dataset.copy()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded dataset.
        
        Returns:
            Dict[str, Any]: Dataset information.
            
        Raises:
            AdvBenchError: If dataset is not loaded.
        """
        if self.dataset is None:
            raise AdvBenchError("Dataset not loaded. Call load_dataset() first.")
        
        return {
            "split": self.current_split,
            "count": len(self.dataset),
            "sample_prompts": [item["prompt"][:100] + "..." if len(item["prompt"]) > 100 else item["prompt"] 
                              for item in self.dataset[:3]]
        }


def load_advbench_dataset(split: str = "harmful_behaviors", cache_dir: str = "advbench_cache") -> AdvBenchLoader:
    """
    Convenience function to load AdvBench dataset.
    
    Args:
        split (str): Dataset split to load.
        cache_dir (str): Directory to cache dataset locally.
        
    Returns:
        AdvBenchLoader: Loaded dataset loader.
    """
    loader = AdvBenchLoader(cache_dir=cache_dir)
    loader.load_dataset(split=split)
    return loader


def get_random_prompts(dataset: List[Dict[str, Any]], count: int = 1) -> List[Dict[str, Any]]:
    """
    Get random prompts from a dataset list.
    
    Args:
        dataset (List[Dict[str, Any]]): Dataset list.
        count (int): Number of random prompts to return.
        
    Returns:
        List[Dict[str, Any]]: Random prompts.
    """
    if count <= 0:
        raise ValueError("Count must be positive.")
    
    if count > len(dataset):
        count = len(dataset)
    
    return random.sample(dataset, count)