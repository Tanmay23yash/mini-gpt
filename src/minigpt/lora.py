"""Stage 3: LoRA fine-tuning of the open LLM for SentiMix sentiment (instruction format).

Usage::

    python -m minigpt.lora --config configs/lora.yaml           # local GTX 1650: 1 lr, 1 seed
    python -m minigpt.lora --config configs/lora_kaggle.yaml    # Kaggle T4: lr sweep + 3 seeds

Training example = the zero-shot chat prompt; target = the label word as the first answer
token. Loss = cross-entropy of the next-token logits at the last prompt position against
that token (same thing as instruction-tuning on a one-word answer). Evaluation uses the
exact label scoring of ``minigpt.llm``, so zero-shot, few-shot and LoRA are comparable.

LoRA: the base weights W stay frozen; each targeted linear layer learns a low-rank update
W + (alpha / r) * B @ A with A: (r, d_in), B: (d_out, r), B initialized to 0 (so training
starts exactly at the base model). Only A and B are trained (~1.4% of the parameters).

The optimization step is ``minigpt.train.train_step`` and the schedule ``minigpt.train.get_lr``
- the same code that pre-trained the mini-GPT.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from minigpt.data import load_sentimix
from minigpt.env import CHECKPOINT_DIR, CONFIG_DIR
from minigpt.evaluate import compute_metrics, plot_confusion_matrix, plot_training_curves
from minigpt.llm import OUT_DIR, build_messages, format_prompt, label_token_ids, load_llm, predict
from minigpt.train import get_lr, train_step
from minigpt.utils import get_device, load_config, set_seed


class LabelLoss(nn.Module):
    """Adapts a Hugging Face causal LM to ``train_step``'s ``model(x, targets=y).loss`` interface."""

    def __init__(self, model: nn.Module, label_token: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("label_token", label_token, persistent=False)   # (L,) answer token per class

    def forward(self, batch: dict[str, torch.Tensor], targets: torch.Tensor | None = None) -> SimpleNamespace:
        logits = self.model(**batch, logits_to_keep=1).logits[:, -1].float()   # (B, V) next-token logits
        loss = F.cross_entropy(logits, self.label_token[targets]) if targets is not None else None
        return SimpleNamespace(logits=logits, loss=loss)


def add_lora(model: nn.Module, cfg: dict[str, Any]) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()   # checkpointing needs grads flowing into the (frozen) embeddings
    lcfg = LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                      target_modules=cfg["target_modules"], task_type="CAUSAL_LM")
    return get_peft_model(model, lcfg)


def make_batches(prompts: list[str], targets: list[int], batch_size: int, seed: int) -> list[tuple[list[str], list[int]]]:
    """Shuffled (prompts, targets) batches; a different order every epoch via the seed."""
    order = torch.randperm(len(prompts), generator=torch.Generator().manual_seed(seed)).tolist()
    return [([prompts[i] for i in order[s:s + batch_size]], [targets[i] for i in order[s:s + batch_size]])
            for s in range(0, len(order), batch_size)]


def evaluate(model: nn.Module, tok: Any, prompts: list[str], y: list[int], ids: dict[str, list[int]],
             labels: list[str], batch_size: int) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    scores, _ = predict(model, tok, prompts, ids, batch_size)
    model.train(was_training)
    preds = scores.argmax(-1).tolist()
    m = compute_metrics(y, preds, labels)
    m["pred_distribution"] = {lab: preds.count(i) / len(preds) for i, lab in enumerate(labels)}
    return m


def run_one(cfg: dict[str, Any], lr: float, seed: int, data: dict[str, Any], labels: list[str],
            device: torch.device, save: bool) -> dict[str, Any]:
    """Train one LoRA adapter; evaluate on val after each epoch; keep the best epoch; test once."""
    t_cfg = cfg["train"]
    set_seed(seed)
    base, tok = load_llm(cfg["model_id"], cfg["dtype"], device)
    ids = label_token_ids(tok, labels)
    model = add_lora(base, cfg["lora"])
    model.train()
    wrapped = LabelLoss(model, torch.tensor([ids[lab][0] for lab in labels], device=device))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=t_cfg["weight_decay"])
    if cfg["dtype"] == "float16":   # fp16 base weights: autocast + loss scaling (Kaggle T4)
        amp_ctx, scaler = torch.autocast("cuda", dtype=torch.float16), torch.amp.GradScaler("cuda")
    else:                           # bf16 (or fp32) base weights: no loss scaling needed
        amp_ctx, scaler = nullcontext(), torch.amp.GradScaler("cuda", enabled=False)

    steps_per_epoch = -(-len(data["train_prompts"]) // t_cfg["batch_size"])
    total = t_cfg["epochs"] * steps_per_epoch
    warmup = int(t_cfg["warmup_frac"] * total)
    history, best = [], {"f1": -1.0}
    step, t0 = 0, time.perf_counter()
    for epoch in range(1, t_cfg["epochs"] + 1):
        losses = []
        for prompts, ys in make_batches(data["train_prompts"], data["train_y"], t_cfg["batch_size"], seed + epoch):
            for group in optimizer.param_groups:
                group["lr"] = get_lr(step, lr, lr * t_cfg["min_lr_ratio"], warmup, total)
            batch = tok(prompts, return_tensors="pt", padding=True).to(device)
            loss, _ = train_step(wrapped, dict(batch), torch.tensor(ys, device=device), optimizer, scaler,
                                 amp_ctx, t_cfg["grad_clip"])
            losses.append(loss)
            step += 1
            if step % t_cfg["log_interval"] == 0:
                rate = step / (time.perf_counter() - t0)
                print(f"    lr {lr:.0e} seed {seed} step {step}/{total} | loss {statistics.fmean(losses[-t_cfg['log_interval']:]):.4f}"
                      f" | {rate:.2f} steps/s | eta {(total - step) / rate / 60:.0f} min")
        val = evaluate(model, tok, data["val_prompts"], data["val_y"], ids, labels, t_cfg["eval_batch_size"])
        history.append({"epoch": epoch, "train_loss": statistics.fmean(losses), "val_loss": float("nan"),
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"]})
        print(f"    lr {lr:.0e} seed {seed} epoch {epoch} | train loss {history[-1]['train_loss']:.4f} "
              f"| val F1 {val['macro_f1']:.4f} | acc {val['accuracy']:.4f}")
        if val["macro_f1"] > best["f1"]:
            best = {"f1": val["macro_f1"], "epoch": epoch, "val": val,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}}

    model.load_state_dict(best["state"], strict=False)       # only the LoRA weights change
    test = evaluate(model, tok, data["test_prompts"], data["test_y"], ids, labels, t_cfg["eval_batch_size"])
    print(f"    lr {lr:.0e} seed {seed}: best epoch {best['epoch']} | val F1 {best['f1']:.4f} | test F1 {test['macro_f1']:.4f}")
    if save:
        out = CHECKPOINT_DIR / "stage3" / f"{cfg['run_name']}_lr{lr:g}_seed{seed}"
        model.save_pretrained(out)                           # adapter only (~40 MB), not the base model
    result = {"lr": lr, "seed": seed, "best_epoch": best["epoch"], "val": best["val"], "test": test,
              "history": history, "minutes": round((time.perf_counter() - t0) / 60, 1),
              "trainable_params": sum(p.numel() for p in trainable)}
    del model, base, wrapped, optimizer
    torch.cuda.empty_cache()
    return result


def run(cfg: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    labels = load_config(CONFIG_DIR / "data.yaml")["sentimix"]["labels"]
    device = get_device()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    system = load_config(CONFIG_DIR / "llm.yaml")["prompting"]["system_prompt"]
    data: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        rows = load_sentimix(split)[:limit]
        data[f"{split}_prompts"] = [format_prompt(tok, build_messages(r["text"], system)) for r in rows]
        data[f"{split}_y"] = [labels.index(r["label"]) for r in rows]

    t_cfg = cfg["train"]
    grid = t_cfg["lr_grid"]
    sweep = []
    if len(grid) > 1:
        print(f"[{cfg['run_name']}] lr sweep with seed {t_cfg['seeds'][0]}: {grid}")
        sweep = [run_one(cfg, lr, t_cfg["seeds"][0], data, labels, device, save=limit is None) for lr in grid]
        lr = max(sweep, key=lambda r: r["val"]["macro_f1"])["lr"]
    else:
        lr = grid[0]
    print(f"[{cfg['run_name']}] lr {lr:.0e}, seeds {t_cfg['seeds']}")
    # the sweep already trained the first seed at the chosen lr: reuse it instead of training it again
    reuse = {r["seed"]: r for r in sweep if r["lr"] == lr}
    runs = [reuse.get(seed) or run_one(cfg, lr, seed, data, labels, device, save=limit is None)
            for seed in t_cfg["seeds"]]

    out_dir = OUT_DIR / ("lora_smoke" if limit else cfg["run_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"run_name": cfg["run_name"], "model_id": cfg["model_id"], "chosen_lr": lr,
                               "lr_sweep": [{"lr": r["lr"], "val_macro_f1": r["val"]["macro_f1"]} for r in sweep],
                               "config": cfg, "seeds": [{k: v for k, v in r.items() if k != "history"} for r in runs],
                               "histories": {r["seed"]: r["history"] for r in runs}}
    for split in ("val", "test"):
        for metric in ("accuracy", "macro_f1"):
            vals = [r[split][metric] for r in runs]
            summary[f"{split}_{metric}"] = {"mean": statistics.fmean(vals),
                                            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_confusion_matrix(runs[0]["test"]["confusion_matrix"], labels, out_dir / "confusion_matrix_test.png",
                          f"{cfg['model_id'].split('/')[-1]} + LoRA - test (seed {runs[0]['seed']})")
    if len(runs[0]["history"]) > 1:
        plot_training_curves(runs[0]["history"], out_dir / "training_curves.png", f"{cfg['run_name']} (seed {runs[0]['seed']})")
    f1 = summary["test_macro_f1"]
    print(f"[{cfg['run_name']}] test macro F1 {f1['mean']:.4f} ± {f1['std']:.4f} over {len(runs)} seed(s) | wrote {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of the open LLM on SentiMix")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "lora.yaml")
    parser.add_argument("--limit", type=int, help="only the first N tweets per split (smoke test)")
    args = parser.parse_args()
    run(load_config(args.config), args.limit)


if __name__ == "__main__":
    main()
