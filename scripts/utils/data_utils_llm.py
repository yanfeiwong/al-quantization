from pathlib import Path
from typing import Dict, Any

from datasets import load_dataset, IterableDataset
from transformers import DataCollatorForLanguageModeling


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==========================================
# Dataset Registry
# ==========================================
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "wikitext-103-raw-v1": {
        "data_dir": str(PROJECT_ROOT / "data" / "wikitext-103-raw-v1"),
    },
}

# ==========================================
# Helper Functions
# ==========================================
def _is_not_empty(example):
    """Global function to replace lambda for Windows multiprocessing picklability."""
    return len(example["text"].strip()) > 0

# ==========================================
# Unified Entry Point
# ==========================================
def get_streaming_dataset(name: str, split: str = "train", seed: int = 921) -> IterableDataset:
    """
    Fetches the dataset in streaming mode to avoid massive disk caching.
    Relies on HuggingFace's auto-detection for local directory formats (e.g., Parquet).
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not found. Available: {list(DATASET_REGISTRY.keys())}")
    
    config = DATASET_REGISTRY[name]

    dataset = load_dataset(
        config["data_dir"], 
        split=split, 
        streaming=True
    )
    
    dataset = dataset.filter(_is_not_empty)
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    
    return dataset

# ==========================================
# Evaluation Dataset
# ==========================================
def get_eval_dataset(name: str, split: str = "validation"):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not found. Available: {list(DATASET_REGISTRY.keys())}")
    config = DATASET_REGISTRY[name]
    dataset = load_dataset(config["data_dir"], split=split)
    dataset = dataset.filter(_is_not_empty)
    return dataset

# ==========================================
# Data Collator Factory
# ==========================================
def get_data_collator(tokenizer):
    """
    Returns a standard DataCollator for Causal LM.
    Note: It handles dynamic padding but NOT truncation.
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, 
        pad_to_multiple_of=8, # Optimizes for Tensor Core efficiency
        return_tensors="pt"
    )
