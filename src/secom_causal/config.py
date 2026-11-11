"""Configuration loading and deterministic seeding utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML pipeline configuration from disk.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(config).__name__}")
    return config


def set_global_seed(seed: int) -> None:
    """Set random seeds across Python stdlib and NumPy for reproducibility.

    Args:
        seed: Non-negative integer seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
