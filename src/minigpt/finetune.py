"""Fine-tune the GPT for SentiMix sentiment classification, and compare with Stage 1.

Usage::

    python -m minigpt.finetune --config configs/finetune_pretrained.yaml   # init from gpt_small
    python -m minigpt.finetune --config configs/finetune_scratch.yaml      # same model, random init
    python -m minigpt.finetune --compare                                   # table + plot vs Stage 1

Protocol (same as Stage 1): pick the learning rate on val with the first seed, then
train every seed with it, keep the epoch with the best val macro F1, and only then
evaluate on test. The training step is ``train.train_step`` and the schedule is
``train.get_lr`` - the same code that pre-trained the model.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from minigpt.data import SPLITS, load_sentimix
from minigpt.env import CHECKPOINT_DIR, CONFIG_DIR, PROJECT_ROOT, RESULTS_DIR
from minigpt.evaluate import compute_metrics, plot_confusion_matrix, plot_training_curves
from minigpt.model.classifier import PAD_ID, GPTClassifier, encode_for_classification
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.tokenizer.bpe import BPETokenizer
from minigpt.train import configure_optimizer, get_lr, load_model, make_amp, train_step
from minigpt.utils import get_device, load_config, set_seed

OUT_ROOT = RESULTS_DIR / "stage2" / "finetune"


def make_loaders(cfg: dict[str, Any], tok: BPETokenizer, labels: list[str], seed: int) -> dict[str, DataLoader]:
    """(ids (B, T) padded with PAD_ID, labels (B,)) batches for each split."""
    label_id = {lab: i for i, lab in enumerate(labels)}

    def collate(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [torch.tensor(ids) for ids, _ in batch]
        return pad_sequence(seqs, batch_first=True, padding_value=PAD_ID), torch.tensor([y for _, y in batch])

    loaders = {}
    for split in SPLITS:
        items = [(encode_for_classification(tok, r["text"], cfg["max_len"]), label_id[r["label"]])
                 for r in load_sentimix(split)]
        loaders[split] = DataLoader(items, batch_size=cfg["batch_size"], shuffle=split == "train",
                                    collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    return loaders


def build_model(cfg: dict[str, Any], n_classes: int, device: torch.device) -> GPTClassifier:
    if cfg["init"] == "pretrained":
        gpt = load_model(PROJECT_ROOT / cfg["checkpoint"], device)
    else:  # same architecture as the checkpoint, random weights
        ckpt = torch.load(PROJECT_ROOT / cfg["checkpoint"], map_location="cpu", weights_only=False)
        gpt = GPT(GPTConfig(**ckpt["config"]["model"]))
    return GPTClassifier(gpt, n_classes, cfg["head_dropout"], cfg["pooling"]).to(device)


@torch.no_grad()
def predict(model: GPTClassifier, loader: DataLoader, device: torch.device, amp_ctx: Any) -> tuple[list[int], float]:
    model.eval()
    preds, total, n = [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with amp_ctx:
            out = model(x, targets=y)
        total += out.loss.item() * len(y)
        n += len(y)
        preds.extend(out.logits.argmax(-1).tolist())
    model.train()
    return preds, total / n


def run_one(cfg: dict[str, Any], lr: float, seed: int, tok: BPETokenizer, labels: list[str],
            device: torch.device, save: bool) -> dict[str, Any]:
    """Fine-tune once; early-stop on val macro F1; evaluate the best epoch on val and test."""
    set_seed(seed)
    loaders = make_loaders(cfg, tok, labels, seed)
    y_true = {s: [int(y) for _, ys in loaders[s] for y in ys] for s in ("val", "test")}
    model = build_model(cfg, len(labels), device)
    opt_cfg = {"weight_decay": cfg["weight_decay"], "max_lr": lr, "betas": cfg["betas"]}
    optimizer = configure_optimizer(model, opt_cfg, device)
    amp_ctx, scaler = make_amp(device, cfg["amp"])
    total_steps = cfg["epochs"] * len(loaders["train"])
    warmup = int(cfg["warmup_frac"] * total_steps)

    history, best_f1, best_state, best_epoch, bad, step = [], -1.0, None, 0, 0, 0
    model.train()
    for epoch in range(1, cfg["epochs"] + 1):
        losses = []
        for x, y in loaders["train"]:
            for group in optimizer.param_groups:
                group["lr"] = get_lr(step, lr, lr * cfg["min_lr_ratio"], warmup, total_steps)
            loss, _ = train_step(model, x.to(device), y.to(device), optimizer, scaler, amp_ctx, cfg["grad_clip"])
            losses.append(loss)
            step += 1
        preds, val_loss = predict(model, loaders["val"], device, amp_ctx)
        val = compute_metrics(y_true["val"], preds, labels)
        history.append({"epoch": epoch, "train_loss": statistics.fmean(losses), "val_loss": val_loss,
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"]})
        print(f"    lr {lr:.0e} seed {seed} epoch {epoch} | train {history[-1]['train_loss']:.4f} "
              f"| val loss {val_loss:.4f} | val F1 {val['macro_f1']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = val["macro_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    result = {"lr": lr, "seed": seed, "best_epoch": best_epoch, "history": history}
    for split in ("val", "test"):
        preds, _ = predict(model, loaders[split], device, amp_ctx)
        result[split] = compute_metrics(y_true[split], preds, labels)
    if save:
        path = CHECKPOINT_DIR / "stage2" / "finetune" / f"{cfg['run_name']}_seed{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best_state, "config": cfg, "gpt_config": model.gpt.config.__dict__,
                    "labels": labels, "val_macro_f1": best_f1}, path)
    return result


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    labels = load_config(CONFIG_DIR / "data.yaml")["sentimix"]["labels"]
    tok = BPETokenizer.load(PROJECT_ROOT / cfg["tokenizer"])
    device = get_device()
    start = time.perf_counter()

    print(f"[{cfg['run_name']}] lr sweep on val with seed {cfg['seeds'][0]}: {cfg['lr_grid']}")
    sweep = [run_one(cfg, lr, cfg["seeds"][0], tok, labels, device, save=False) for lr in cfg["lr_grid"]]
    best = max(sweep, key=lambda r: r["val"]["macro_f1"])
    lr = best["lr"]
    print(f"[{cfg['run_name']}] chosen lr {lr:.0e} (val macro F1 {best['val']['macro_f1']:.4f})")

    runs = [run_one(cfg, lr, seed, tok, labels, device, save=True) for seed in cfg["seeds"]]

    out_dir = OUT_ROOT / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "run_name": cfg["run_name"], "init": cfg["init"], "chosen_lr": lr,
        "lr_sweep": [{"lr": r["lr"], "val_macro_f1": r["val"]["macro_f1"]} for r in sweep],
        "params": sum(p.numel() for p in build_model(cfg, len(labels), torch.device("cpu")).parameters()),
        "minutes": round((time.perf_counter() - start) / 60, 1), "config": cfg,
        "seeds": [{k: v for k, v in r.items() if k != "history"} for r in runs],
    }
    for split in ("val", "test"):
        for metric in ("accuracy", "macro_f1"):
            summary[f"{split}_{metric}"] = _mean_std([r[split][metric] for r in runs])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r in runs:
        with (out_dir / f"history_seed{r['seed']}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(r["history"][0]))
            w.writeheader()
            w.writerows(r["history"])
    first = runs[0]
    plot_confusion_matrix(first["test"]["confusion_matrix"], labels, out_dir / "confusion_matrix_test.png",
                          f"{cfg['run_name']} - test (seed {first['seed']})")
    plot_training_curves(first["history"], out_dir / "training_curves.png", f"{cfg['run_name']} (seed {first['seed']})")

    f1 = summary["test_macro_f1"]
    print(f"[{cfg['run_name']}] test macro F1 {f1['mean']:.4f} ± {f1['std']:.4f} over {len(runs)} seeds "
          f"| {summary['minutes']} min | wrote {out_dir}")
    return summary


# --------------------------------------------------------------------------- comparison with Stage 1


def compare() -> list[dict[str, Any]]:
    """Collect every model's test macro F1 into results/stage2/classifier_comparison.{md,png}."""
    from minigpt.evaluate import GRID, INK, INK_MUTED, SERIES, SURFACE, _plt

    rows = []
    stage1 = RESULTS_DIR / "stage1"
    names = {"tfidf_logreg": "TF-IDF + LogReg", "bigru": "BiGRU (word)", "bilstm": "BiLSTM (word)"}
    majority = None
    for run_name, label in names.items():
        s = json.loads((stage1 / run_name / "summary.json").read_text(encoding="utf-8"))
        rows.append({"model": label, "stage": 1, "mean": s["test_macro_f1"]["mean"], "std": s["test_macro_f1"]["std"],
                     "acc": s["test_accuracy"]["mean"], "seeds": len(s["seeds"])})
        majority = s["majority_baseline_test"]
    for run_name, label in (("gpt_scratch", "mini-GPT, random init"), ("gpt_pretrained", "mini-GPT, pre-trained")):
        path = OUT_ROOT / run_name / "summary.json"
        if path.exists():
            s = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"model": label, "stage": 2, "mean": s["test_macro_f1"]["mean"],
                         "std": s["test_macro_f1"]["std"], "acc": s["test_accuracy"]["mean"], "seeds": len(s["seeds"])})
    rows.append({"model": "Majority class", "stage": 1, "mean": majority["macro_f1"], "std": 0.0,
                 "acc": majority["accuracy"], "seeds": 1})
    rows.sort(key=lambda r: r["mean"])

    lines = ["| Model | Stage | Test macro F1 | Test accuracy | Seeds |", "|---|---|---|---|---|"]
    for r in reversed(rows):
        std = f" ± {r['std']:.3f}" if r["seeds"] > 1 else ""
        lines.append(f"| {r['model']} | {r['stage']} | {r['mean']:.3f}{std} | {r['acc']:.3f} | {r['seeds']} |")
    out = RESULTS_DIR / "stage2" / "classifier_comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.2), facecolor=SURFACE)
    ys = range(len(rows))
    ax.barh(ys, [r["mean"] for r in rows], height=0.6, color=SERIES[0])
    multi = [(y, r) for y, r in zip(ys, rows) if r["seeds"] > 1]  # error bars only where there is a spread
    ax.errorbar([r["mean"] for _, r in multi], [y for y, _ in multi], xerr=[r["std"] for _, r in multi],
                fmt="none", ecolor=INK_MUTED, elinewidth=1, capsize=3)
    for y, r in zip(ys, rows):
        ax.text(r["mean"] + r["std"] + 0.01, y, f"{r['mean']:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(ys), [r["model"] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.set_xlabel("test macro F1 (mean ± std over seeds)", color=INK_MUTED, fontsize=9)
    ax.set_title("SentiMix Hinglish sentiment: Stage 1 baselines vs mini-GPT", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(out.read_text(encoding="utf-8"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the mini-GPT for sentiment")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "finetune_pretrained.yaml")
    parser.add_argument("--compare", action="store_true", help="only build the comparison table and plot")
    args = parser.parse_args()
    if args.compare:
        compare()
    else:
        run(load_config(args.config))


if __name__ == "__main__":
    main()
