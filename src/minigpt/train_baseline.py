"""Stage 1 baselines on SentiMix: TF-IDF + logistic regression, and a (Bi)LSTM/GRU.

Usage::

    python -m minigpt.train_baseline --config configs/stage1_tfidf.yaml
    python -m minigpt.train_baseline --config configs/stage1_lstm.yaml

Every run trains once per seed in the config, picks the model with the best
validation macro F1, and only then touches the test set. Outputs go to
``results/stage1/<run_name>/`` (summary.json, per-seed history CSVs, plots);
RNN checkpoints go to ``checkpoints/stage1/``, TensorBoard logs to ``runs/``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from minigpt.data import SPLITS, load_sentimix
from minigpt.env import CHECKPOINT_DIR, CONFIG_DIR, PROJECT_ROOT, RESULTS_DIR
from minigpt.evaluate import compute_metrics, plot_confusion_matrix, plot_training_curves
from minigpt.model.lstm import RNNClassifier
from minigpt.tokenizer.simple import WordTokenizer
from minigpt.utils import get_device, load_config, set_seed

Split = dict[str, list]  # {"texts": [...], "y": [...]}


def load_splits(labels: list[str]) -> dict[str, Split]:
    """Load processed SentiMix splits as texts + integer labels."""
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    out = {}
    for split in SPLITS:
        rows = load_sentimix(split)
        out[split] = {"texts": [r["text"] for r in rows], "y": [label_to_id[r["label"]] for r in rows]}
    return out


# --------------------------------------------------------------------------- TF-IDF


def tfidf_features(cfg: dict[str, Any], data: dict[str, Split], return_vectorizer: bool = False) -> Any:
    """Word 1-2 + character 2-5 n-gram TF-IDF, fit on train, applied to every split.
    With ``return_vectorizer`` also returns the fitted FeatureUnion (for the demo app)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    t = cfg["tfidf"]
    common = {"min_df": t["min_df"], "sublinear_tf": t["sublinear_tf"], "lowercase": True,
              "max_features": t["max_features"]}
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=tuple(t["word_ngrams"]),
                                 token_pattern=r"[\w']+", **common)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(t["char_ngrams"]), **common)),
    ])
    x = {s: (features.fit_transform if s == "train" else features.transform)(data[s]["texts"]) for s in SPLITS}
    return (x, features) if return_vectorizer else x


def tfidf_test_predictions(cfg: dict[str, Any], data: dict[str, Split], c: float) -> list[int]:
    """Refit the TF-IDF model with a given C (e.g. the one chosen on val) and predict test."""
    from sklearn.linear_model import LogisticRegression

    x = tfidf_features(cfg, data)
    clf = LogisticRegression(C=c, max_iter=cfg["logreg"]["max_iter"], random_state=cfg["seeds"][0])
    return clf.fit(x["train"], data["train"]["y"]).predict(x["test"]).tolist()


def run_tfidf(cfg: dict[str, Any], data: dict[str, Split]) -> dict[str, Any]:
    """Word + character n-gram TF-IDF with logistic regression; C picked on val."""
    from sklearn.linear_model import LogisticRegression

    x = tfidf_features(cfg, data)
    best: dict[str, Any] = {"f1": -1.0}
    for c in cfg["logreg"]["C_grid"]:
        clf = LogisticRegression(C=c, max_iter=cfg["logreg"]["max_iter"], random_state=cfg["seeds"][0])
        clf.fit(x["train"], data["train"]["y"])
        f1 = compute_metrics(data["val"]["y"], clf.predict(x["val"]), cfg["labels"])["macro_f1"]
        print(f"  C={c:<5} val macro F1 {f1:.4f}")
        if f1 > best["f1"]:
            best = {"f1": f1, "C": c, "clf": clf}

    clf = best["clf"]
    return {
        "seed": cfg["seeds"][0],
        "chosen_C": best["C"],
        "n_features": x["train"].shape[1],
        "val": compute_metrics(data["val"]["y"], clf.predict(x["val"]), cfg["labels"]),
        "test": compute_metrics(data["test"]["y"], clf.predict(x["test"]), cfg["labels"]),
        "history": [],
    }


# --------------------------------------------------------------------------- RNN


def make_loader(tok: WordTokenizer, split: Split, max_len: int, batch_size: int, shuffle: bool,
                seed: int) -> DataLoader:
    """DataLoader yielding (ids (B, T) right-padded, lengths (B,), labels (B,))."""
    items = [(torch.tensor(tok.encode(t, max_len)), y) for t, y in zip(split["texts"], split["y"])]

    def collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seqs, ys = zip(*batch)
        ids = pad_sequence(list(seqs), batch_first=True, padding_value=tok.pad_id)
        return ids, torch.tensor([len(s) for s in seqs]), torch.tensor(ys)

    gen = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=gen)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], float]:
    """Return predicted class ids and the mean cross-entropy loss over the loader."""
    model.eval()
    preds, total_loss, n = [], 0.0, 0
    for ids, lengths, y in loader:
        ids, y = ids.to(device), y.to(device)
        logits = model(ids, lengths)
        total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        n += len(y)
        preds.extend(logits.argmax(dim=-1).tolist())
    return preds, total_loss / n


def oov_rate(tok: WordTokenizer, texts: list[str]) -> float:
    """Share of tokens that map to <unk>."""
    counts = Counter(i == tok.unk_id for t in texts for i in tok.encode(t) if t)
    return counts[True] / max(1, sum(counts.values()))


def run_rnn(cfg: dict[str, Any], data: dict[str, Split], tok: WordTokenizer, seed: int,
            device: torch.device) -> dict[str, Any]:
    """Train one RNN with early stopping on val macro F1, then evaluate the best epoch on test."""
    set_seed(seed)
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    loaders = {s: make_loader(tok, data[s], d["max_len"], t["batch_size"], s == "train", seed) for s in SPLITS}
    model = RNNClassifier(len(tok), len(cfg["labels"]), m["embed_dim"], m["hidden_dim"], m["num_layers"],
                          m["bidirectional"], m["dropout"], m["rnn_type"], tok.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    writer = SummaryWriter(PROJECT_ROOT / "runs" / "stage1" / cfg["run_name"] / f"seed{seed}")

    history: list[dict[str, float]] = []
    best_f1, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        train_loss, n = 0.0, 0
        for ids, lengths, y in loaders["train"]:
            ids, y = ids.to(device), y.to(device)
            loss = nn.functional.cross_entropy(model(ids, lengths), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            optimizer.step()
            train_loss += loss.item() * len(y)
            n += len(y)

        preds, val_loss = predict(model, loaders["val"], device)
        val = compute_metrics(data["val"]["y"], preds, cfg["labels"])
        row = {"epoch": epoch, "train_loss": train_loss / n, "val_loss": val_loss,
               "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"]}
        history.append(row)
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)
        print(f"  seed {seed} epoch {epoch:2d} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f}"
              f" | val acc {val['accuracy']:.4f} | val F1 {val['macro_f1']:.4f}")

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val["macro_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= t["patience"]:
                print(f"  early stop: no val F1 gain for {bad_epochs} epochs")
                break
    writer.close()

    model.load_state_dict(best_state)
    ckpt_path = CHECKPOINT_DIR / "stage1" / f"{cfg['run_name']}_seed{seed}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "config": cfg, "itos": tok.itos, "epoch": best_epoch}, ckpt_path)

    val_preds, _ = predict(model, loaders["val"], device)
    test_preds, _ = predict(model, loaders["test"], device)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "n_params": sum(p.numel() for p in model.parameters()),
        "val": compute_metrics(data["val"]["y"], val_preds, cfg["labels"]),
        "test": compute_metrics(data["test"]["y"], test_preds, cfg["labels"]),
        "history": history,
    }


# --------------------------------------------------------------------------- reporting


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def summarize(cfg: dict[str, Any], runs: list[dict[str, Any]], data: dict[str, Split], extra: dict[str, Any],
              out_dir: Path) -> dict[str, Any]:
    """Aggregate seeds, write summary.json, history CSVs and plots, and print a table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"run_name": cfg["run_name"], "config": cfg, **extra, "seeds": []}
    for split in ("val", "test"):
        for metric in ("accuracy", "macro_f1"):
            summary[f"{split}_{metric}"] = _mean_std([r[split][metric] for r in runs])
    for r in runs:
        summary["seeds"].append({k: v for k, v in r.items() if k != "history"})
        if r["history"]:
            with (out_dir / f"history_seed{r['seed']}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(r["history"][0]))
                writer.writeheader()
                writer.writerows(r["history"])

    majority = Counter(data["train"]["y"]).most_common(1)[0][0]
    summary["majority_baseline_test"] = compute_metrics(data["test"]["y"], [majority] * len(data["test"]["y"]),
                                                        cfg["labels"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    first = runs[0]
    plot_confusion_matrix(first["test"]["confusion_matrix"], cfg["labels"], out_dir / "confusion_matrix_test.png",
                          f"{cfg['run_name']} - test (seed {first['seed']})")
    if first["history"]:
        plot_training_curves(first["history"], out_dir / "training_curves.png",
                             f"{cfg['run_name']} (seed {first['seed']})")

    fmt = lambda s: f"{s['mean']:.4f} ± {s['std']:.4f}"  # noqa: E731
    print(f"\n{cfg['run_name']} over {len(runs)} seed(s)")
    print(f"  val   acc {fmt(summary['val_accuracy'])} | macro F1 {fmt(summary['val_macro_f1'])}")
    print(f"  test  acc {fmt(summary['test_accuracy'])} | macro F1 {fmt(summary['test_macro_f1'])}")
    maj = summary["majority_baseline_test"]
    print(f"  majority-class test: acc {maj['accuracy']:.4f} | macro F1 {maj['macro_f1']:.4f}")
    print(f"  wrote {out_dir.relative_to(PROJECT_ROOT)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 baselines")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "stage1_lstm.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["labels"] = load_config(CONFIG_DIR / "data.yaml")["sentimix"]["labels"]
    data = load_splits(cfg["labels"])
    out_dir = RESULTS_DIR / "stage1" / cfg["run_name"]
    start = time.perf_counter()

    extra: dict[str, Any] = {}
    if cfg["model_type"] == "tfidf":
        runs = [run_tfidf(cfg, data)]
    else:
        device = get_device()
        d = cfg["data"]
        tok = WordTokenizer.build(data["train"]["texts"], d["min_freq"], d["max_vocab"], d["lowercase"])
        extra = {"vocab_size": len(tok), "oov_rate": {s: oov_rate(tok, data[s]["texts"]) for s in ("val", "test")},
                 "device": str(device)}
        print(f"device {device} | vocab {len(tok):,} | val OOV rate {extra['oov_rate']['val']:.1%}")
        runs = [run_rnn(cfg, data, tok, seed, device) for seed in cfg["seeds"]]
    extra["train_seconds"] = round(time.perf_counter() - start, 1)
    summarize(cfg, runs, data, extra, out_dir)


if __name__ == "__main__":
    main()
