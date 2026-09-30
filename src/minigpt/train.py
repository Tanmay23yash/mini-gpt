"""Pre-train the GPT on the HingCorpus token stream.   [LEARN FILE: get_lr and train_step]

Usage::

    python -m minigpt.train --config configs/gpt_debug.yaml    # ~1 min, CPU or GPU: does it learn?
    python -m minigpt.train --config configs/gpt_small.yaml    # real run on the GPU
    python -m minigpt.train --config configs/gpt_small.yaml --resume

One training step:

    1. sample a batch of random windows from train.bin:  x = tokens[i : i+T],  y = tokens[i+1 : i+T+1]
    2. forward: loss = cross-entropy of predicting y from x           (mixed precision on GPU)
    3. backward: gradients of the loss w.r.t. every weight
    4. clip the gradient norm to 1.0 (one bad batch can't blow up the weights)
    5. AdamW update with the learning rate for this step (warmup, then cosine decay)

Provided: batching, evaluation, optimizer setup, logging, checkpoints, plots.
What you implement: ``get_lr`` and ``train_step`` (the ``TODO``s). Tests: ``pytest tests/test_train.py -v``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from minigpt.env import CONFIG_DIR, PROJECT_ROOT
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.utils import get_device, load_config, set_seed

# --------------------------------------------------------------------------- LEARN: schedule + step


def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Learning rate for ``step``: linear warmup, then cosine decay to ``min_lr``.

        lr
        max_lr |     ____
               |    /    ‾‾‾--__
               |   /            ‾‾--__
        min_lr |  /                   ‾‾‾‾‾‾‾‾‾  (stays at min_lr after max_steps)
               +--|--------------------|------> step
                  warmup_steps         max_steps

    Rules:
        - step < warmup_steps:  lr = max_lr * (step + 1) / warmup_steps     (never exactly 0)
        - step >= max_steps:    lr = min_lr
        - otherwise:            progress = (step - warmup_steps) / (max_steps - warmup_steps)   in [0, 1]
                                coeff = 0.5 * (1 + cos(pi * progress))                          1 -> 0
                                lr = min_lr + coeff * (max_lr - min_lr)

    Why warmup: at the start Adam's statistics are noisy and a big lr can wreck the
    random init. Why decay: small steps at the end settle into a better minimum.
    """
    # TODO: implement
    raise NotImplementedError


def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer,
               scaler: torch.amp.GradScaler, amp_ctx: Any, grad_clip: float) -> tuple[float, float]:
    """One optimization step. Returns (loss, gradient norm before clipping) as Python floats.

    Args:
        model: the GPT (already in train mode).
        x, y: (B, T) input ids and next-token targets, on the model's device.
        optimizer: AdamW (its learning rate is already set for this step).
        scaler: GradScaler for fp16 mixed precision. When disabled (CPU), its methods
            are pass-throughs, so the same code works everywhere.
        amp_ctx: context manager for mixed precision (``torch.autocast(...)`` on GPU, a
            no-op on CPU). Only the FORWARD pass goes inside it.
        grad_clip: max global gradient norm.

    Steps:
        1. with amp_ctx:  loss = model(x, targets=y).loss
        2. scaler.scale(loss).backward()            # fp16 grads are scaled up so they don't underflow to 0
        3. scaler.unscale_(optimizer)               # back to true gradient values before clipping
        4. grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        5. scaler.step(optimizer)                   # = optimizer.step(), skipped if grads had inf/nan
        6. scaler.update()
        7. optimizer.zero_grad(set_to_none=True)    # gradients accumulate otherwise!
        8. return loss.item(), grad_norm.item()
    """
    # TODO: implement
    raise NotImplementedError


# --------------------------------------------------------------------------- data


def load_tokens(path: Path) -> np.memmap:
    """Memory-map a uint16 token file: reads from disk on demand instead of loading 100 MB into RAM."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: torch.device,
              generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows from the token stream: x = data[i:i+T], y = data[i+1:i+T+1] -> (B, T) each."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator).numpy()
    windows = data[starts[:, None] + np.arange(block_size + 1)]           # (B, T+1) in one indexing op
    chunk = torch.from_numpy(windows.astype(np.int64))
    x, y = chunk[:, :-1], chunk[:, 1:]
    if device.type == "cuda":
        x, y = x.contiguous().pin_memory(), y.contiguous().pin_memory()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x.contiguous().to(device), y.contiguous().to(device)


@torch.no_grad()
def estimate_loss(model: nn.Module, data: np.ndarray, cfg: dict[str, Any], device: torch.device,
                  amp_ctx: Any, seed: int) -> float:
    """Mean loss over ``eval_iters`` batches; a fixed seed so every evaluation uses the same windows."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(cfg["eval_iters"]):
        x, y = get_batch(data, cfg["batch_size"], cfg["block_size"], device, gen)
        with amp_ctx:
            losses.append(model(x, targets=y).loss.item())
    model.train()
    return float(np.mean(losses))


# --------------------------------------------------------------------------- setup helpers


def configure_optimizer(model: nn.Module, cfg: dict[str, Any], device: torch.device) -> torch.optim.AdamW:
    """AdamW with weight decay only on 2-D weights (matmuls, embeddings), not on biases / LayerNorms."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg["max_lr"], betas=tuple(cfg["betas"]), fused=device.type == "cuda")


def make_amp(device: torch.device, enabled: bool) -> tuple[Any, torch.amp.GradScaler]:
    """fp16 autocast + GradScaler on CUDA when enabled; plain fp32 otherwise.

    Measured on the GTX 1650 (no tensor cores): fp16 AMP was 4.6x SLOWER than fp32
    (862 vs 187 ms/step for gpt_small), so the local configs set ``amp: false``.
    GPUs with tensor cores (Kaggle T4/P100 notebooks) should use ``amp: true``.
    """
    use_amp = enabled and device.type == "cuda"
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
    return ctx, torch.amp.GradScaler("cuda", enabled=use_amp)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, step: int, best_val: float, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "best_val_loss": best_val, "config": cfg}, path)


def load_model(path: str | Path, device: torch.device | str = "cpu") -> GPT:
    """Rebuild a trained GPT from a checkpoint (for generation, plots, fine-tuning)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]["model"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def plot_loss(history: list[dict[str, float]], path: Path, title: str) -> None:
    """Train (every log step) and val (every eval) loss against tokens seen."""
    from minigpt.evaluate import SERIES, _plt, _style

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="#fcfcfb")
    train = [h for h in history if h.get("train_loss") is not None]
    val = [h for h in history if h.get("val_loss") is not None]
    for rows, key, name, color in ((train, "train_loss", "train", SERIES[0]), (val, "val_loss", "val", SERIES[1])):
        xs = [h["tokens"] / 1e6 for h in rows]
        ys = [h[key] for h in rows]
        ax.plot(xs, ys, color=color, linewidth=2, label=name, marker="o" if name == "val" else None, markersize=4)
        ax.annotate(f"{name} {ys[-1]:.3f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 0),
                    color="#52514e", fontsize=8, va="center")
    _style(ax)
    ax.set_xlabel("tokens seen (millions)", color="#52514e", fontsize=9)
    ax.set_ylabel("cross-entropy loss", color="#52514e", fontsize=9)
    ax.set_title(title, loc="left", color="#0b0b0b", fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor="#52514e")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


# --------------------------------------------------------------------------- main loop


def train(cfg: dict[str, Any], resume: bool = False) -> dict[str, Any]:
    """Run pre-training from a config dict. Returns a summary (also written to summary.json)."""
    run = cfg["run_name"]
    t_cfg = {**cfg["train"], "block_size": cfg["model"]["block_size"]}  # one source of truth for T
    set_seed(cfg["seed"])
    device = get_device() if t_cfg.get("device", "auto") == "auto" else torch.device(t_cfg["device"])
    data_dir = PROJECT_ROOT / cfg["data_dir"]
    train_data, val_data = load_tokens(data_dir / "train.bin"), load_tokens(data_dir / "val.bin")

    model = GPT(GPTConfig(**cfg["model"])).to(device)
    optimizer = configure_optimizer(model, t_cfg, device)
    amp_ctx, scaler = make_amp(device, t_cfg["amp"])
    ckpt_dir = PROJECT_ROOT / cfg["checkpoint_dir"] / run
    out_dir = PROJECT_ROOT / cfg["results_dir"] / run
    out_dir.mkdir(parents=True, exist_ok=True)

    start_step, best_val, history = 0, float("inf"), []
    if resume and (ckpt_dir / "last.pt").exists():
        ckpt = torch.load(ckpt_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step, best_val = ckpt["step"] + 1, ckpt["best_val_loss"]
        if (out_dir / "history.csv").exists():
            with (out_dir / "history.csv").open(encoding="utf-8") as f:
                history = [{k: (float(v) if v else None) for k, v in r.items()} for r in csv.DictReader(f)]
        print(f"resumed from step {ckpt['step']}")

    tokens_per_step = t_cfg["batch_size"] * t_cfg["block_size"]
    print(f"device {device} | params {model.num_params() / 1e6:.2f}M | train tokens {len(train_data) / 1e6:.1f}M"
          f" | {t_cfg['max_steps']:,} steps x {tokens_per_step:,} tokens = {t_cfg['max_steps'] * tokens_per_step / 1e6:.0f}M")
    writer = SummaryWriter(PROJECT_ROOT / "runs" / "stage2" / run)
    gen = torch.Generator().manual_seed(cfg["seed"] + start_step)
    model.train()
    t0, t_log = time.perf_counter(), time.perf_counter()

    for step in range(start_step, t_cfg["max_steps"]):
        lr = get_lr(step, t_cfg["max_lr"], t_cfg["min_lr"], t_cfg["warmup_steps"], t_cfg["max_steps"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, t_cfg["batch_size"], t_cfg["block_size"], device, gen)
        loss, grad_norm = train_step(model, x, y, optimizer, scaler, amp_ctx, t_cfg["grad_clip"])

        last = step == t_cfg["max_steps"] - 1
        row: dict[str, Any] = {"step": step, "tokens": (step + 1) * tokens_per_step, "lr": lr,
                               "train_loss": None, "val_loss": None, "grad_norm": grad_norm}
        if step % t_cfg["log_interval"] == 0 or last:
            elapsed = time.perf_counter() - t_log
            tok_s = tokens_per_step * (t_cfg["log_interval"] if step > start_step else 1) / max(elapsed, 1e-9)
            t_log = time.perf_counter()
            row["train_loss"] = loss
            writer.add_scalar("train/loss", loss, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/grad_norm", grad_norm, step)
            print(f"step {step:>6} | loss {loss:.4f} | lr {lr:.2e} | grad norm {grad_norm:.2f} | {tok_s:,.0f} tok/s")
        if step % t_cfg["eval_interval"] == 0 or last:
            val_loss = estimate_loss(model, val_data, t_cfg, device, amp_ctx, seed=cfg["seed"])
            row["val_loss"] = val_loss
            writer.add_scalar("val/loss", val_loss, step)
            print(f"  eval | val loss {val_loss:.4f} | val perplexity {math.exp(val_loss):.1f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scaler, step, best_val, cfg)
            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scaler, step, best_val, cfg)
        if row["train_loss"] is not None or row["val_loss"] is not None:
            history.append(row)
    writer.close()

    minutes = (time.perf_counter() - t0) / 60
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "tokens", "lr", "train_loss", "val_loss", "grad_norm"])
        w.writeheader()
        w.writerows(history)
    plot_loss(history, out_dir / "loss_curve.png", f"{run} ({model.num_params() / 1e6:.1f}M params)")
    summary = {"run_name": run, "params": model.num_params(), "best_val_loss": best_val,
               "best_val_perplexity": math.exp(best_val), "final_train_loss": history[-1]["train_loss"],
               "steps": t_cfg["max_steps"], "tokens_seen": t_cfg["max_steps"] * tokens_per_step,
               "minutes": round(minutes, 1), "device": str(device), "config": cfg}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"done in {minutes:.1f} min | best val loss {best_val:.4f} (perplexity {math.exp(best_val):.1f})"
          f" | wrote {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train the mini-GPT")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "gpt_debug.yaml")
    parser.add_argument("--resume", action="store_true", help="continue from checkpoints/.../last.pt")
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
