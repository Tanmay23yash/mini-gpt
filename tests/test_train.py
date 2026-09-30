"""Tests for the pre-training loop (src/minigpt/train.py)."""

import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

from minigpt.model.gpt import GPT, GPTConfig
from minigpt.train import get_batch, get_lr, load_model, make_amp, train, train_step

LR = {"max_lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 100, "max_steps": 1000}


# --------------------------------------------------------------------------- get_lr


def test_lr_warmup_is_linear_and_never_zero() -> None:
    assert get_lr(0, **LR) == pytest.approx(1e-3 * 1 / 100)
    assert get_lr(49, **LR) == pytest.approx(1e-3 * 50 / 100)
    assert get_lr(99, **LR) == pytest.approx(1e-3)


def test_lr_peaks_after_warmup_then_decays_to_min() -> None:
    assert get_lr(100, **LR) == pytest.approx(1e-3)                  # start of cosine: max
    assert get_lr(550, **LR) == pytest.approx((1e-3 + 1e-4) / 2)     # halfway: midpoint
    assert get_lr(1000, **LR) == pytest.approx(1e-4)                 # end: min
    assert get_lr(5000, **LR) == pytest.approx(1e-4)                 # stays at min


def test_lr_decreases_monotonically_after_warmup() -> None:
    lrs = [get_lr(s, **LR) for s in range(100, 1001)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    assert all(1e-4 <= lr <= 1e-3 for lr in lrs)


# --------------------------------------------------------------------------- train_step


def _setup(device: str = "cpu", amp: bool = False):
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=50, block_size=16, d_model=32, n_layers=2, n_heads=2, dropout=0.0)).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    amp_ctx, scaler = make_amp(torch.device(device), amp)
    x = torch.randint(0, 50, (4, 16), device=device)
    y = torch.randint(0, 50, (4, 16), device=device)
    return model, optimizer, amp_ctx, scaler, x, y


def test_train_step_returns_floats_and_learns() -> None:
    model, opt, ctx, scaler, x, y = _setup()
    first, norm = train_step(model, x, y, opt, scaler, ctx, grad_clip=1.0)
    assert isinstance(first, float) and isinstance(norm, float)
    for _ in range(40):
        loss, _ = train_step(model, x, y, opt, scaler, ctx, grad_clip=1.0)
    assert loss < 0.5 * first


def test_train_step_changes_weights_and_clears_grads() -> None:
    model, opt, ctx, scaler, x, y = _setup()
    before = model.blocks[0].attn.qkv.weight.detach().clone()
    train_step(model, x, y, opt, scaler, ctx, grad_clip=1.0)
    assert not torch.equal(before, model.blocks[0].attn.qkv.weight)
    assert all(p.grad is None for p in model.parameters()), "call optimizer.zero_grad(set_to_none=True)"


def test_train_step_returns_norm_before_clipping() -> None:
    model, opt, ctx, scaler, x, y = _setup()
    _, norm = train_step(model, x, y, opt, scaler, ctx, grad_clip=1e-4)
    assert norm > 1e-4  # the returned value is the true norm, clipping happens afterwards


def test_train_step_uses_the_loss_of_the_given_batch() -> None:
    model, opt, ctx, scaler, x, y = _setup()
    with torch.no_grad():
        expected = model(x, targets=y).loss.item()
    loss, _ = train_step(model, x, y, opt, scaler, ctx, grad_clip=1.0)
    assert loss == pytest.approx(expected, rel=1e-5)  # loss BEFORE the update


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_train_step_with_fp16_amp_on_gpu() -> None:
    model, opt, ctx, scaler, x, y = _setup("cuda", amp=True)
    assert scaler.is_enabled()
    first, _ = train_step(model, x, y, opt, scaler, ctx, grad_clip=1.0)
    for _ in range(40):
        loss, _ = train_step(model, x, y, opt, scaler, ctx, grad_clip=1.0)
    assert math.isfinite(loss) and loss < 0.5 * first


# --------------------------------------------------------------------------- provided pieces


def test_get_batch_targets_are_inputs_shifted_by_one() -> None:
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, 8, 16, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (8, 16)
    assert x.dtype == torch.int64
    torch.testing.assert_close(y[:, :-1], x[:, 1:])
    torch.testing.assert_close(y[:, -1], x[:, -1] + 1)


def _tiny_run(tmp_path: Path, max_steps: int) -> dict:
    """A config that trains on a synthetic, perfectly predictable token stream."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    pattern = np.tile(np.arange(20, dtype=np.uint16), 500)  # 0, 1, ..., 19, 0, 1, ...
    pattern.tofile(data_dir / "train.bin")
    pattern[:2000].tofile(data_dir / "val.bin")
    return {
        "run_name": "pytest_tiny", "seed": 0, "data_dir": str(data_dir),
        "checkpoint_dir": str(tmp_path / "ckpt"), "results_dir": str(tmp_path / "results"),
        "model": {"vocab_size": 20, "block_size": 16, "d_model": 32, "n_layers": 1, "n_heads": 2,
                  "dropout": 0.0, "bias": True},
        "train": {"device": "cpu", "batch_size": 8, "max_steps": max_steps, "warmup_steps": 5, "max_lr": 3e-3,
                  "min_lr": 3e-4, "weight_decay": 0.1, "betas": [0.9, 0.95], "grad_clip": 1.0, "amp": False,
                  "log_interval": 10, "eval_interval": 20, "eval_iters": 2},
    }


def test_end_to_end_training_learns_and_writes_outputs(tmp_path: Path) -> None:
    summary = train(_tiny_run(tmp_path, max_steps=80))
    assert summary["best_val_loss"] < 0.5 * math.log(20)  # the pattern is learnable
    for f in ("ckpt/pytest_tiny/best.pt", "ckpt/pytest_tiny/last.pt", "results/pytest_tiny/history.csv",
              "results/pytest_tiny/loss_curve.png", "results/pytest_tiny/summary.json"):
        assert (tmp_path / f).exists(), f
    model = load_model(tmp_path / "ckpt/pytest_tiny/best.pt")
    assert isinstance(model, GPT)


def test_resume_continues_from_last_checkpoint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    train(_tiny_run(tmp_path, max_steps=40))
    summary = train(_tiny_run(tmp_path, max_steps=60), resume=True)
    assert "resumed from step 39" in capsys.readouterr().out
    assert summary["steps"] == 60
