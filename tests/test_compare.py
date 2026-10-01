"""Tests for the bootstrap statistics in src/minigpt/compare.py."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from minigpt.compare import _median_seed, ci, macro_f1_batch, paired_bootstrap


def test_macro_f1_batch_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=(5, 200))
    p = np.where(rng.random((5, 200)) < 0.6, y, rng.integers(0, 3, size=(5, 200)))
    ours = macro_f1_batch(y, p, 3)
    ref = [f1_score(y[i], p[i], average="macro", labels=[0, 1, 2], zero_division=0) for i in range(5)]
    np.testing.assert_allclose(ours, ref)


def test_macro_f1_batch_handles_a_class_never_predicted() -> None:
    y = np.array([[0, 1, 2, 2]])
    p = np.array([[0, 0, 2, 2]])  # class 1 never predicted -> its F1 is 0
    ref = f1_score(y[0], p[0], average="macro", labels=[0, 1, 2], zero_division=0)
    assert macro_f1_batch(y, p, 3)[0] == pytest.approx(ref)


def test_paired_bootstrap_is_paired_and_reproducible() -> None:
    y = [0, 1, 2] * 50
    preds = {"a": y, "b": [0] * 150}
    boot = paired_bootstrap(y, preds, 3, n_resamples=200, seed=1)
    assert boot["a"].shape == (200,)
    assert np.all(boot["a"] == 1.0)                       # perfect model: F1 1 on every resample
    assert np.all(boot["a"] > boot["b"])                  # same resample -> paired comparison
    again = paired_bootstrap(y, preds, 3, n_resamples=200, seed=1)
    np.testing.assert_array_equal(boot["b"], again["b"])


def test_ci_is_the_central_95_percent() -> None:
    lo, hi = ci(np.arange(1001, dtype=float))
    assert lo == pytest.approx(25.0)
    assert hi == pytest.approx(975.0)


def test_median_seed_picks_the_middle_run() -> None:
    runs = [{"seed": 1, "s": 0.70}, {"seed": 2, "s": 0.72}, {"seed": 3, "s": 0.71}]
    assert _median_seed(runs, lambda r: r["s"]) == 3
