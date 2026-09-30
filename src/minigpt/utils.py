"""Small shared helpers: seeding, device selection, config loading."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all GPUs) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return ``cuda`` if a GPU is usable, else ``cpu``."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)
