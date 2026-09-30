"""Sanity checks for the project scaffold."""

import os

import torch

import minigpt
from minigpt.env import PROJECT_ROOT
from minigpt.utils import set_seed


def test_project_root_has_pyproject() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_cache_dirs_not_on_c_drive() -> None:
    for var in ("HF_HOME", "TORCH_HOME"):
        value = os.environ.get(var, "")
        assert value, f"{var} is not set"
        assert not value.upper().startswith("C:"), f"{var} points at C: ({value})"


def test_set_seed_is_reproducible() -> None:
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)
