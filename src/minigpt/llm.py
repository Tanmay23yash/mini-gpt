"""Stage 3: SentiMix sentiment with an open LLM (Qwen3.5-0.8B) - zero-shot and few-shot.

Usage::

    python -m minigpt.llm                         # zero-shot + few-shot on val (choose k) and test
    python -m minigpt.llm --limit 60              # quick smoke run on 60 tweets per split

Prediction = label scoring, not free-text generation: build the chat prompt, run ONE
forward pass, and compare the model's log-probability of each label word as the first
answer token (``positive`` / ``negative`` / ``neutral``, capitalized variants summed in).
No parsing of free text, and every tweet gets exactly one of the three labels.

Few-shot: k labeled training tweets (k/3 per class, shuffled) are given as earlier
user/assistant turns before the test tweet. k is chosen on val; test is reported over
3 different random draws of the examples to show how much the choice of examples matters.

Contextual calibration (Zhao et al., 2021): small LLMs have a built-in label bias (here:
they almost never say "neutral"). Score a few content-free inputs ("N/A", "", "[MASK]")
with the same prompt, average their label probabilities p_cf, and divide it out:
calibrated log-score = log p(label | tweet) - log p_cf(label). No labeled data needed.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from minigpt.data import load_sentimix
from minigpt.env import CONFIG_DIR, RESULTS_DIR
from minigpt.evaluate import compute_metrics, plot_confusion_matrix
from minigpt.utils import get_device, load_config, set_seed

OUT_DIR = RESULTS_DIR / "stage3"


# --------------------------------------------------------------------------- prompts


def build_messages(text: str, system_prompt: str, examples: list[tuple[str, str]] = ()) -> list[dict[str, str]]:
    """System instruction, then (tweet -> label) example turns, then the tweet to classify."""
    messages = [{"role": "system", "content": system_prompt}]
    for ex_text, ex_label in examples:
        messages += [{"role": "user", "content": ex_text}, {"role": "assistant", "content": ex_label}]
    messages.append({"role": "user", "content": text})
    return messages


def format_prompt(tok: Any, messages: list[dict[str, str]]) -> str:
    """Chat template with the generation prompt, thinking disabled (answer comes immediately)."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def select_fewshot(rows: list[dict[str, Any]], k: int, labels: list[str], seed: int,
                   min_words: int = 5, max_words: int = 25) -> list[tuple[str, str]]:
    """k examples, k // len(labels) per class, from mid-length tweets, in random order."""
    if k % len(labels):
        raise ValueError(f"k={k} must be a multiple of {len(labels)} to stay class-balanced")
    rng = random.Random(seed)
    chosen = []
    for lab in labels:
        pool = [r for r in rows if r["label"] == lab and min_words <= len(r["text"].split()) <= max_words]
        chosen += [(r["text"], lab) for r in rng.sample(pool, k // len(labels))]
    rng.shuffle(chosen)
    return chosen


def label_token_ids(tok: Any, labels: list[str]) -> dict[str, list[int]]:
    """Token ids of each label as the first answer token: [lowercase, Capitalized].
    Every variant must be a single token and no id may be shared between labels."""
    ids: dict[str, list[int]] = {}
    for lab in labels:
        variants = []
        for form in (lab, lab.capitalize()):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"label {form!r} is {len(enc)} tokens; label scoring needs single tokens")
            variants.append(enc[0])
        ids[lab] = list(dict.fromkeys(variants))   # lowercase first: LoRA trains on ids[lab][0]
    flat = [i for v in ids.values() for i in v]
    if len(flat) != len(set(flat)):
        raise ValueError("label token ids overlap")
    return ids


def label_scores(last_logits: torch.Tensor, ids: dict[str, list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """From (B, V) next-token logits: per-label log-probs (B, L) and the total probability
    mass the model puts on any label token (B,) - a check that it answers in the format."""
    logp = torch.log_softmax(last_logits.float(), dim=-1)
    per_label = torch.stack([torch.logsumexp(logp[:, v], dim=-1) for v in ids.values()], dim=-1)
    return per_label, per_label.exp().sum(-1)


# --------------------------------------------------------------------------- inference


def load_llm(model_id: str, dtype: str, device: torch.device) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"  # the last position must be the real last token for every row
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
    return model, tok


@torch.no_grad()
def predict(model: Any, tok: Any, prompts: list[str], ids: dict[str, list[int]], batch_size: int,
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-label log-probs (N, L) and label-token mass (N,) for every prompt, in input order.
    Batches are length-sorted to cut padding."""
    device = next(model.parameters()).device
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    scores, mass = torch.zeros(len(prompts), len(ids)), torch.zeros(len(prompts))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1]
        s, m = label_scores(logits, ids)
        scores[idx], mass[idx] = s.cpu(), m.cpu()
    return scores, mass


def content_free_prior(model: Any, tok: Any, cfg: dict[str, Any], ids: dict[str, list[int]],
                       examples: list[tuple[str, str]]) -> torch.Tensor:
    """Mean label distribution (L,) for content-free inputs under this exact prompt."""
    prompts = [format_prompt(tok, build_messages(t, cfg["system_prompt"], examples)) for t in cfg["content_free"]]
    scores, _ = predict(model, tok, prompts, ids, len(prompts))
    probs = torch.softmax(scores, dim=-1)                                  # renormalize over the labels
    return probs.mean(0)


def calibrate(scores: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """Contextual calibration: divide out the content-free label prior (in log space)."""
    return torch.log_softmax(scores, dim=-1) - prior.log()


def _metrics(y_true: list[int], scores: torch.Tensor, labels: list[str]) -> dict[str, Any]:
    preds = scores.argmax(-1).tolist()
    m = compute_metrics(y_true, preds, labels)
    m["pred_distribution"] = {lab: preds.count(i) / len(preds) for i, lab in enumerate(labels)}
    return m


def evaluate_prompting(model: Any, tok: Any, rows: list[dict[str, Any]], cfg: dict[str, Any], labels: list[str],
                       examples: list[tuple[str, str]], batch_size: int) -> dict[str, Any]:
    """Raw and contextually calibrated metrics for one prompt configuration."""
    ids = label_token_ids(tok, labels)
    prompts = [format_prompt(tok, build_messages(r["text"], cfg["system_prompt"], examples)) for r in rows]
    t = time.perf_counter()
    scores, mass = predict(model, tok, prompts, ids, batch_size)
    prior = content_free_prior(model, tok, cfg, ids, examples)
    y_true = [labels.index(r["label"]) for r in rows]
    return {
        "raw": _metrics(y_true, scores, labels),
        "calibrated": _metrics(y_true, calibrate(scores, prior), labels),
        "content_free_prior": dict(zip(labels, prior.tolist())),
        "label_token_mass": float(mass.mean()),
        "seconds": round(time.perf_counter() - t, 1),
    }


# --------------------------------------------------------------------------- experiment


def run(cfg: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    """Choose (k, raw|calibrated) on val, then report test: zero-shot, and the chosen k over 3 example draws."""
    set_seed(cfg["seed"])
    labels = load_config(CONFIG_DIR / "data.yaml")["sentimix"]["labels"]
    model, tok = load_llm(cfg["model_id"], cfg["dtype"], get_device())
    train_rows = load_sentimix("train")
    splits = {s: load_sentimix(s)[:limit] for s in ("val", "test")}
    p = cfg["prompting"]
    out_dir = OUT_DIR / ("smoke" if limit else "prompting")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"model_id": cfg["model_id"], "config": p, "val": {}, "test": {}}

    def show(tag: str, r: dict[str, Any]) -> None:
        for variant in ("raw", "calibrated"):
            m = r[variant]
            dist = " ".join(f"{k[:3]} {v:.0%}" for k, v in m["pred_distribution"].items())
            print(f"  {tag:<26} {variant:<10} macro F1 {m['macro_f1']:.4f} | acc {m['accuracy']:.4f} | preds {dist}")
        print(f"  {'':<26} label mass {r['label_token_mass']:.2f} | {r['seconds']:.0f}s")

    # 1. val: every k, raw and calibrated (one draw of examples)
    for k in p["k_values"]:
        examples = select_fewshot(train_rows, k, labels, p["example_seeds"][0]) if k else []
        r = evaluate_prompting(model, tok, splits["val"], p, labels, examples, p["batch_size"][str(k)])
        results["val"][f"k={k}"] = r
        show(f"val  k={k}", r)
    best_k, best_variant = max(((k, v) for k in p["k_values"] for v in ("raw", "calibrated")),
                               key=lambda kv: results["val"][f"k={kv[0]}"][kv[1]]["macro_f1"])
    results["chosen"] = {"k": best_k, "variant": best_variant}
    print(f"  chosen on val: k={best_k}, {best_variant}")

    # 2. test: zero-shot once; chosen k over every example draw
    results["test"]["k=0"] = evaluate_prompting(model, tok, splits["test"], p, labels, [], p["batch_size"]["0"])
    show("test k=0", results["test"]["k=0"])
    if best_k:
        runs = []
        for seed in p["example_seeds"]:
            examples = select_fewshot(train_rows, best_k, labels, seed)
            r = evaluate_prompting(model, tok, splits["test"], p, labels, examples, p["batch_size"][str(best_k)])
            r["example_seed"] = seed
            runs.append(r)
            show(f"test k={best_k} (ex. seed {seed})", r)
        results["test"][f"k={best_k}"] = {"runs": runs}
        for variant in ("raw", "calibrated"):
            f1s = [r[variant]["macro_f1"] for r in runs]
            results["test"][f"k={best_k}"][variant] = {
                "macro_f1": {"mean": statistics.fmean(f1s), "std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0},
                "accuracy": statistics.fmean(r[variant]["accuracy"] for r in runs)}
        plot_confusion_matrix(runs[0][best_variant]["confusion_matrix"], labels,
                              out_dir / f"confusion_matrix_test_k{best_k}_{best_variant}.png",
                              f"{cfg['model_id'].split('/')[-1]} {best_k}-shot ({best_variant}) - test")
    for variant in ("raw", "calibrated"):
        plot_confusion_matrix(results["test"]["k=0"][variant]["confusion_matrix"], labels,
                              out_dir / f"confusion_matrix_test_k0_{variant}.png",
                              f"{cfg['model_id'].split('/')[-1]} zero-shot ({variant}) - test")
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero/few-shot SentiMix with an open LLM")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "llm.yaml")
    parser.add_argument("--limit", type=int, help="only the first N tweets of each split (smoke test)")
    args = parser.parse_args()
    run(load_config(args.config), args.limit)


if __name__ == "__main__":
    main()
