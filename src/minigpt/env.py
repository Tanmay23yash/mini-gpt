"""Project paths and environment setup.

Imported by ``minigpt/__init__.py`` so cache locations (HF_HOME, TORCH_HOME, ...)
are set before torch / datasets are imported anywhere in the package.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
RESULTS_DIR: Path = PROJECT_ROOT / "results"


def setup_env() -> None:
    """Load ``.env`` from the project root and create the cache directories.

    Existing environment variables win over ``.env`` values (``override=False``),
    so user-level settings are respected.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    for var in ("PIP_CACHE_DIR", "HF_HOME", "TORCH_HOME", "MPLCONFIGDIR", "WANDB_DIR", "WANDB_CACHE_DIR"):
        value = os.environ.get(var)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)
