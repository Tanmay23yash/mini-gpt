"""Stage 3 final comparison: every model on the same SentiMix test set.

Usage::

    python -m minigpt.compare               # ~15 min on the GTX 1650 the first time (predictions are cached)
    python -m minigpt.compare --recompute   # ignore cached predictions

1. Test predictions of one representative run per model family (the seed with the median
   test score): TF-IDF, fine-tuned mini-GPT, Qwen3.5 3-shot, Qwen3.5 + LoRA.
   Cached in results/stage3/predictions_test.csv.
2. Paired bootstrap over the 3,000 test tweets (same resample for every model): a 95% CI
   for each macro F1, and for the differences between models.
3. The full table (all stages, mean +- std over seeds), a bar chart, and example tweets.
4. Text generation on the same prompts: the mini-GPT vs Qwen3.5-0.8B (no fine-tuning).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from minigpt.data import load_sentimix
from minigpt.env import CHECKPOINT_DIR, CONFIG_DIR, PROJECT_ROOT, RESULTS_DIR
from minigpt.utils import get_device, load_config

OUT = RESULTS_DIR / "stage3"
PRED_CSV = OUT / "predictions_test.csv"
MODELS = ["tfidf", "minigpt", "qwen_3shot", "qwen_lora"]
NAMES = {"tfidf": "TF-IDF + LogReg", "minigpt": "mini-GPT (pre-trained, fine-tuned)",
         "qwen_3shot": "Qwen3.5-0.8B 3-shot", "qwen_lora": "Qwen3.5-0.8B + LoRA"}


def _median_seed(runs: list[dict[str, Any]], score: Callable[[dict[str, Any]], float], key: str = "seed") -> Any:
    ordered = sorted(runs, key=score)
    return ordered[len(ordered) // 2][key]


# --------------------------------------------------------------------------- predictions


def predict_tfidf(labels: list[str]) -> list[int]:
    from minigpt.train_baseline import load_splits, tfidf_test_predictions

    cfg = load_config(CONFIG_DIR / "stage1_tfidf.yaml")
    cfg["labels"] = labels
    c = json.loads((RESULTS_DIR / "stage1/tfidf_logreg/summary.json").read_text(encoding="utf-8"))["seeds"][0]["chosen_C"]
    return tfidf_test_predictions(cfg, load_splits(labels), c)


def predict_minigpt(labels: list[str], device: torch.device) -> tuple[list[int], int]:
    from minigpt.finetune import make_loaders, predict
    from minigpt.model.classifier import GPTClassifier
    from minigpt.model.gpt import GPT, GPTConfig
    from minigpt.tokenizer.bpe import BPETokenizer

    summary = json.loads((RESULTS_DIR / "stage2/finetune/gpt_pretrained/summary.json").read_text(encoding="utf-8"))
    seed = _median_seed(summary["seeds"], lambda r: r["test"]["macro_f1"])
    ckpt = torch.load(CHECKPOINT_DIR / f"stage2/finetune/gpt_pretrained_seed{seed}.pt", map_location="cpu",
                      weights_only=False)
    cfg = ckpt["config"]
    model = GPTClassifier(GPT(GPTConfig(**ckpt["gpt_config"])), len(labels), cfg["head_dropout"], cfg["pooling"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tok = BPETokenizer.load(PROJECT_ROOT / cfg["tokenizer"])
    preds, _ = predict(model, make_loaders(cfg, tok, labels, seed=0)["test"], device, nullcontext())
    return preds, seed


def _qwen_predict(model: Any, tok: Any, prompts: list[str], labels: list[str], batch_size: int) -> list[int]:
    from minigpt.llm import label_token_ids, predict

    scores, _ = predict(model, tok, prompts, label_token_ids(tok, labels), batch_size)
    return scores.argmax(-1).tolist()


def predict_qwen_3shot(labels: list[str], device: torch.device) -> tuple[list[int], int]:
    from minigpt.llm import build_messages, format_prompt, load_llm, select_fewshot

    cfg = load_config(CONFIG_DIR / "llm.yaml")
    runs = json.loads((OUT / "prompting/summary.json").read_text(encoding="utf-8"))["test"]["k=3"]["runs"]
    seed = _median_seed(runs, lambda r: r["raw"]["macro_f1"], key="example_seed")
    model, tok = load_llm(cfg["model_id"], cfg["dtype"], device)
    p = cfg["prompting"]
    examples = select_fewshot(load_sentimix("train"), 3, labels, seed)
    prompts = [format_prompt(tok, build_messages(r["text"], p["system_prompt"], examples)) for r in load_sentimix("test")]
    preds = _qwen_predict(model, tok, prompts, labels, p["batch_size"]["3"])
    del model
    torch.cuda.empty_cache()
    return preds, seed


def predict_qwen_lora(labels: list[str], device: torch.device) -> tuple[list[int], int]:
    from peft import PeftModel

    from minigpt.llm import build_messages, format_prompt, load_llm

    summary = json.loads((OUT / "lora_kaggle/summary.json").read_text(encoding="utf-8"))
    seed = _median_seed(summary["seeds"], lambda r: r["test"]["macro_f1"])
    lr = summary["chosen_lr"]
    cfg = load_config(CONFIG_DIR / "llm.yaml")
    # Inference in bf16 here (trained in fp16 on Kaggle); the adapter is merged into the weights.
    base, tok = load_llm(summary["model_id"], cfg["dtype"], device)
    adapter = CHECKPOINT_DIR / f"stage3/lora_kaggle_lr{lr:g}_seed{seed}"
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload().eval()
    system = cfg["prompting"]["system_prompt"]
    prompts = [format_prompt(tok, build_messages(r["text"], system)) for r in load_sentimix("test")]
    preds = _qwen_predict(model, tok, prompts, labels, 32)
    del model, base
    torch.cuda.empty_cache()
    return preds, seed


def collect_predictions(labels: list[str], recompute: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows of uid, text, gold and each model's prediction (label strings), plus the seeds used."""
    meta_path = OUT / "predictions_meta.json"
    if PRED_CSV.exists() and meta_path.exists() and not recompute:
        with PRED_CSV.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f)), json.loads(meta_path.read_text(encoding="utf-8"))
    device = get_device()
    test = load_sentimix("test")
    preds, meta = {}, {}
    print("predicting test: TF-IDF")
    preds["tfidf"] = predict_tfidf(labels)
    print("predicting test: mini-GPT")
    preds["minigpt"], meta["minigpt_seed"] = predict_minigpt(labels, device)
    print("predicting test: Qwen3.5 3-shot (~9 min)")
    preds["qwen_3shot"], meta["qwen_3shot_example_seed"] = predict_qwen_3shot(labels, device)
    print("predicting test: Qwen3.5 + LoRA (~5 min)")
    preds["qwen_lora"], meta["qwen_lora_seed"] = predict_qwen_lora(labels, device)
    rows = [{"uid": r["uid"], "text": r["text"], "gold": r["label"], **{m: labels[preds[m][i]] for m in MODELS}}
            for i, r in enumerate(test)]
    OUT.mkdir(parents=True, exist_ok=True)
    with PRED_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return rows, meta


# --------------------------------------------------------------------------- bootstrap


def macro_f1_batch(y: np.ndarray, p: np.ndarray, n_classes: int) -> np.ndarray:
    """Macro F1 for many resamples at once. y, p: (R, N) int arrays -> (R,)."""
    f1s = []
    for c in range(n_classes):
        tp = ((p == c) & (y == c)).sum(1)
        fp = ((p == c) & (y != c)).sum(1)
        fn = ((p != c) & (y == c)).sum(1)
        denom = 2 * tp + fp + fn
        f1s.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    return np.mean(f1s, axis=0)


def paired_bootstrap(y: list[int], preds: dict[str, list[int]], n_classes: int, n_resamples: int = 2000,
                     seed: int = 0) -> dict[str, np.ndarray]:
    """Macro F1 of every model on the SAME resampled test sets -> {model: (n_resamples,)}."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_resamples, len(y)))
    y_arr = np.asarray(y)[idx]
    return {m: macro_f1_batch(y_arr, np.asarray(p)[idx], n_classes) for m, p in preds.items()}


def ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- table + plot


def all_results(labels: list[str]) -> list[dict[str, Any]]:
    """Every model's test macro F1 (mean, std over seeds), accuracy and size, from the saved summaries."""
    def load(path: str) -> dict[str, Any]:
        return json.loads((RESULTS_DIR / path).read_text(encoding="utf-8"))

    rows = []
    s = load("stage1/tfidf_logreg/summary.json")
    rows.append({"key": "tfidf", "model": NAMES["tfidf"], "stage": 1, "params": s["seeds"][0]["n_features"] * len(labels),
                 "trained": "all", "f1": s["test_macro_f1"], "acc": s["test_accuracy"]["mean"], "seeds": 1})
    rows.append({"key": "majority", "model": "Majority class", "stage": 1, "params": 0, "trained": "-",
                 "f1": {"mean": s["majority_baseline_test"]["macro_f1"], "std": 0.0},
                 "acc": s["majority_baseline_test"]["accuracy"], "seeds": 1})
    for run, name in (("bigru", "BiGRU (word vocab)"), ("bilstm", "BiLSTM (word vocab)")):
        s = load(f"stage1/{run}/summary.json")
        rows.append({"key": run, "model": name, "stage": 1, "params": s["seeds"][0]["n_params"], "trained": "all",
                     "f1": s["test_macro_f1"], "acc": s["test_accuracy"]["mean"], "seeds": len(s["seeds"])})
    for run, key, name in (("gpt_pretrained", "minigpt", NAMES["minigpt"]),
                           ("gpt_scratch", "minigpt_scratch", "mini-GPT (random init, fine-tuned)")):
        s = load(f"stage2/finetune/{run}/summary.json")
        rows.append({"key": key, "model": name, "stage": 2, "params": s["params"], "trained": "all",
                     "f1": s["test_macro_f1"], "acc": s["test_accuracy"]["mean"], "seeds": len(s["seeds"])})
    p = load("stage3/prompting/summary.json")
    qwen_params = 752_000_000
    rows.append({"key": "qwen_0shot", "model": "Qwen3.5-0.8B zero-shot", "stage": 3, "params": qwen_params,
                 "trained": "none", "f1": {"mean": p["test"]["k=0"]["raw"]["macro_f1"], "std": 0.0},
                 "acc": p["test"]["k=0"]["raw"]["accuracy"], "seeds": 1})
    k3 = p["test"]["k=3"]["raw"]
    rows.append({"key": "qwen_3shot", "model": NAMES["qwen_3shot"], "stage": 3, "params": qwen_params,
                 "trained": "none", "f1": k3["macro_f1"], "acc": k3["accuracy"], "seeds": 3})
    s = load("stage3/lora_kaggle/summary.json")
    rows.append({"key": "qwen_lora", "model": NAMES["qwen_lora"], "stage": 3, "params": qwen_params,
                 "trained": s["seeds"][0]["trainable_params"], "f1": s["test_macro_f1"],
                 "acc": s["test_accuracy"]["mean"], "seeds": len(s["seeds"])})
    return sorted(rows, key=lambda r: -r["f1"]["mean"])


def _fmt_params(n: Any) -> str:
    if not isinstance(n, (int, float)) or n == 0:
        return str(n) if n else "-"
    return f"{n / 1e9:.2f}B" if n >= 1e9 * 0.5 else f"{n / 1e6:.1f}M"


def plot_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    """Horizontal bars, one color per stage, std error bars for multi-seed runs, value labels."""
    from minigpt.evaluate import GRID, INK, INK_MUTED, SURFACE, _plt

    stage_colors = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}   # categorical slots 1-3 (all-pairs validated)
    plt = _plt()
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.4), facecolor=SURFACE)
    ys = np.arange(len(rows))
    ax.barh(ys, [r["f1"]["mean"] for r in rows], height=0.62, color=[stage_colors[r["stage"]] for r in rows],
            edgecolor=SURFACE, linewidth=2)
    multi = [(y, r) for y, r in zip(ys, rows) if r["seeds"] > 1]
    ax.errorbar([r["f1"]["mean"] for _, r in multi], [y for y, _ in multi], xerr=[r["f1"]["std"] for _, r in multi],
                fmt="none", ecolor=INK_MUTED, elinewidth=1, capsize=3)
    for y, r in zip(ys, rows):
        ax.text(r["f1"]["mean"] + r["f1"]["std"] + 0.01, y, f"{r['f1']['mean']:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, [r["model"] for r in rows])
    ax.set_xlim(0, 0.85)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.set_xlabel("test macro F1 (mean ± std over seeds)", color=INK_MUTED, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in stage_colors.values()]
    ax.legend(handles, ["Stage 1: classic NLP", "Stage 2: mini-GPT from scratch", "Stage 3: open LLM"],
              frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="lower right")
    ax.set_title("SentiMix Hinglish sentiment: every model on the same test set", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- examples


# Display filter only (metrics use every tweet): keep strong slurs out of the README examples.
PROFANITY = ("chut", "bhosd", "madarch", "behench", "bhench", "gand", "andu", "randi", "lode", "lund", "lavd",
             "harami", "kutt", "fuck", "shit", "bitch", "bc ", "mc ")


def pick_examples(rows: list[dict[str, Any]], labels: list[str], per_class: int = 4, seed: int = 0) -> list[dict[str, Any]]:
    """Per gold class: tweets where the models disagree (most informative), mid-length, no slurs, fixed seed."""
    rng = random.Random(seed)
    picked = []
    for lab in labels:
        pool = [r for r in rows if r["gold"] == lab and 6 <= len(r["text"].split()) <= 22
                and len({r[m] for m in MODELS}) >= 2
                and not any(w in f" {r['text'].lower()} " for w in PROFANITY)]
        picked += rng.sample(pool, per_class)
    return picked


@torch.no_grad()
def generation_comparison(prompts: list[str], seed: int = 7) -> list[dict[str, str]]:
    """Continue the same prompts with the mini-GPT and with Qwen3.5-0.8B (plain text, no chat template)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from minigpt.generate import generate_text, load_generation_setup

    device = get_device()
    gcfg = load_config(CONFIG_DIR / "generate.yaml")
    mini, bpe = load_generation_setup(gcfg["checkpoint"], gcfg["tokenizer"], device)
    out = [{"prompt": p, "minigpt": generate_text(mini, bpe, p, 1, 30, 0.7, 50, None, seed)[0]} for p in prompts]
    del mini
    model_id = load_config(CONFIG_DIR / "llm.yaml")["model_id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    qwen = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    for row in out:
        torch.manual_seed(seed)
        ids = tok(row["prompt"], return_tensors="pt").to(device)
        gen = qwen.generate(**ids, max_new_tokens=30, do_sample=True, temperature=0.7, top_k=50)
        row["qwen"] = row["prompt"] + tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    del qwen
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 final comparison")
    parser.add_argument("--recompute", action="store_true", help="recompute cached test predictions")
    args = parser.parse_args()
    labels = load_config(CONFIG_DIR / "data.yaml")["sentimix"]["labels"]

    pred_rows, meta = collect_predictions(labels, args.recompute)
    y = [labels.index(r["gold"]) for r in pred_rows]
    preds = {m: [labels.index(r[m]) for r in pred_rows] for m in MODELS}
    boot = paired_bootstrap(y, preds, len(labels))
    point = {m: float(macro_f1_batch(np.asarray([y]), np.asarray([preds[m]]), len(labels))[0]) for m in MODELS}

    lines = ["# Stage 3: final comparison on the SentiMix test set (3,000 tweets)", "",
             "Mean ± std over seeds (training seeds, or few-shot example draws). The 95% CI is a paired bootstrap",
             "(2,000 resamples of the test tweets) of one representative run per model (the median seed).", "",
             "| Model | Stage | Params | Trained params | Test macro F1 | 95% CI (rep. run) | Test acc |",
             "|---|---|---|---|---|---|---|"]
    rows = all_results(labels)
    for r in rows:
        std = f" ± {r['f1']['std']:.3f}" if r["seeds"] > 1 else ""
        interval = "-"
        if r["key"] in boot:
            lo, hi = ci(boot[r["key"]])
            interval = f"{point[r['key']]:.3f} [{lo:.3f}, {hi:.3f}]"
        trained = r["trained"] if isinstance(r["trained"], str) else _fmt_params(r["trained"])
        if trained == "all":
            trained = _fmt_params(r["params"])
        lines.append(f"| {r['model']} | {r['stage']} | {_fmt_params(r['params'])} | {trained} | "
                     f"{r['f1']['mean']:.3f}{std} | {interval} | {r['acc']:.3f} |")

    lines += ["", "## Is the difference real? Paired bootstrap of the macro F1 difference", "",
              "| Comparison | Difference | 95% CI | Share of resamples where the first is better |", "|---|---|---|---|"]
    diffs = {}
    for a, b in (("qwen_lora", "tfidf"), ("qwen_lora", "minigpt"), ("tfidf", "minigpt"), ("qwen_lora", "qwen_3shot"),
                 ("minigpt", "qwen_3shot")):
        d = boot[a] - boot[b]
        lo, hi = ci(d)
        diffs[f"{a} - {b}"] = {"diff": point[a] - point[b], "ci": [lo, hi], "p_better": float((d > 0).mean())}
        lines.append(f"| {NAMES[a]} vs {NAMES[b]} | {point[a] - point[b]:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                     f"{(d > 0).mean():.1%} |")

    lines += ["", "## Per-class F1 (representative runs)", "", "| Model | negative | neutral | positive |", "|---|---|---|---|"]
    from minigpt.evaluate import compute_metrics
    per_class = {}
    for m in MODELS:
        pc = compute_metrics(y, preds[m], labels)["f1_per_class"]
        per_class[m] = pc
        lines.append(f"| {NAMES[m]} | " + " | ".join(f"{pc[lab]:.3f}" for lab in labels) + " |")

    examples = pick_examples(pred_rows, labels)
    lines += ["", "## Example test tweets where the models disagree", "",
              "Picked at random (fixed seed), 4 per gold label, among tweets where at least two models disagree;",
              "tweets with strong profanity are skipped for display (the metrics above use every tweet).", "",
              "| Tweet | Gold | " + " | ".join(NAMES[m] for m in MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for r in examples:
        cells = " | ".join(("✓ " if r[m] == r["gold"] else "✗ ") + r[m] for m in MODELS)
        lines.append(f"| {r['text'].replace('|', '/')} | {r['gold']} | {cells} |")

    gen = generation_comparison(load_config(CONFIG_DIR / "generate.yaml")["prompts"][1:])
    lines += ["", "## Text generation on the same prompts (T=0.7, top-k 50, 30 new tokens)", "",
              "| Prompt | mini-GPT (5.3M, Hinglish tweets only) | Qwen3.5-0.8B (no fine-tuning) |", "|---|---|---|"]
    for g in gen:
        clean = {k: v.replace("|", "/").replace("\n", " ") for k, v in g.items()}
        lines.append(f"| `{clean['prompt']}` | {clean['minigpt']} | {clean['qwen']} |")

    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_comparison(rows, OUT / "comparison.png")
    (OUT / "comparison.json").write_text(json.dumps({
        "representative_runs": meta, "point_macro_f1": point,
        "bootstrap_ci": {m: ci(boot[m]) for m in MODELS}, "differences": diffs, "per_class_f1": per_class,
        "table": [{k: v for k, v in r.items()} for r in rows]}, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
