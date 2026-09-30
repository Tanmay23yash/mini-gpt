"""Stage 0 data pipeline for SentiMix (labeled) and HingCorpus (unlabeled).

Usage::

    python -m minigpt.data --prepare   # download, clean, split -> data/processed/
    python -m minigpt.data --report    # print the data report, save a length plot

Outputs:
    data/processed/sentimix/{train,val,test}.csv   columns: uid,label,text,cmi
    data/processed/pretrain/{train,val}.txt        one cleaned tweet per line
    data/processed/pretrain/stats.json             corpus stats for the report
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import tarfile
import urllib.request
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import ftfy

from minigpt.env import CONFIG_DIR, PROJECT_ROOT
from minigpt.utils import load_config

SPLITS = ("train", "val", "test")

# --------------------------------------------------------------------------- cleaning

# SentiMix ships tweets pre-tokenized, so URLs/mentions/hashtags arrive split apart:
#   "https // t . co / 1TcBUH6Jp1", "@ tha _ rami", "# cwc19"
_TCO_URL_RE = re.compile(r"https?\s*:?\s*/\s*/\s*t\s*\.?\s*co\s*/\s*\w+", re.IGNORECASE)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# Tweets cut off mid-URL: "... https", "... https // t co", "... https // t. c"
_TRUNCATED_URL_RE = re.compile(r"\s*\bhttps?\b[\s:/.]*(?:t[\s.]*c?o?)?[\s/.]*$", re.IGNORECASE)
_UNDERSCORE_RE = re.compile(r"(\w)\s*(_+)\s*(\w)")
_MENTION_RE = re.compile(r"@(?:\s*_+)?\s*\w+(?:\s*_+\s*\w*)*")
_HASHTAG_RE = re.compile(r"#\s+(?=\w)")
_RT_RE = re.compile(r"^\s*RT\b\s*:?\s*")
_TRUNCATION_RE = re.compile(r"\s*…\s*$")
_APOSTROPHE_RE = re.compile(r"(\w)\s*'\s*(s|t|ve|re|ll|d|m)\b", re.IGNORECASE)  # contractions only
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:%)\]}])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{])\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def normalize_text(text: str) -> str:
    """Fix mojibake, straighten quotes and collapse whitespace."""
    text = ftfy.fix_text(text).translate(_QUOTE_MAP)
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_tweet(tokens: list[str]) -> str:
    """Rebuild a readable tweet from SentiMix tokens.

    Removes URLs, @mentions and a leading "RT" (HingCorpus has these stripped too,
    so both datasets end up in the same style), rejoins hashtags and apostrophes,
    and removes the spaces the tokenizer put before punctuation.
    """
    text = normalize_text(" ".join(tokens))
    text = _TCO_URL_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _RT_RE.sub("", text)
    text = _HASHTAG_RE.sub("#", text)
    text = _TRUNCATION_RE.sub("", text)
    text = _TRUNCATED_URL_RE.sub("", text)
    text = _TRUNCATION_RE.sub("", text)  # a "…" can sit right before the cut URL
    text = _UNDERSCORE_RE.sub(r"\1\2\3", text)
    text = _APOSTROPHE_RE.sub(r"\1'\2", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def leakage_key(text: str, n_chars: int) -> str:
    """Lowercase alphanumeric prefix used to spot eval tweets inside the corpus."""
    return re.sub(r"[^a-z0-9]", "", text.lower())[:n_chars]


def code_mixing_index(langs: list[str]) -> float:
    """Code-Mixing Index (Das & Gambäck, 2014) of one tweet, in [0, 50] for 2 languages.

    CMI = 100 * (1 - max_lang_tokens / (n - u)), where u counts language-independent
    tokens (punctuation, emoji, tags other than Hin/Eng). 0 means monolingual.
    """
    lang_counts = Counter(tag for tag in langs if tag in ("Hin", "Eng"))
    n_lang = sum(lang_counts.values())
    if n_lang == 0:
        return 0.0
    return 100.0 * (1.0 - max(lang_counts.values()) / n_lang)


# --------------------------------------------------------------------------- SentiMix


@dataclass
class Tweet:
    """One SentiMix tweet: tokens with per-token language tags and a label."""

    uid: str
    label: str | None = None
    tokens: list[str] = field(default_factory=list)
    langs: list[str] = field(default_factory=list)


def parse_conll(text: str) -> list[Tweet]:
    """Parse the SentiMix CoNLL format.

    Each tweet starts with ``meta<TAB>uid[<TAB>label]`` followed by one
    ``token<TAB>lang`` line per token; tweets are separated by blank lines.
    """
    tweets: list[Tweet] = []
    for line in text.splitlines():
        if line.startswith("meta\t"):
            parts = line.split("\t")
            tweets.append(Tweet(uid=parts[1].strip(), label=parts[2].strip() if len(parts) > 2 else None))
        elif line.strip() and tweets:
            token, _, lang = line.partition("\t")
            tweets[-1].tokens.append(token)
            tweets[-1].langs.append(lang.strip() or "O")
    return tweets


def _read_test_labels(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        return {row["Uid"].strip(): row["Sentiment"].strip() for row in csv.DictReader(f)}


def prepare_sentimix(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Download SentiMix from the Hub, clean it and write one CSV per split."""
    from huggingface_hub import hf_hub_download

    files = cfg["files"]
    fetch = lambda name: Path(hf_hub_download(cfg["hf_repo"], name, repo_type="dataset"))  # noqa: E731
    test_labels = _read_test_labels(fetch(files["test_labels"]))
    out_dir = PROJECT_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        tweets = parse_conll(fetch(files[split]).read_text(encoding="utf-8"))
        rows = []
        for tw in tweets:
            label = tw.label or test_labels.get(tw.uid)
            if label not in cfg["labels"]:
                raise ValueError(f"{split}: tweet {tw.uid} has unknown label {label!r}")
            rows.append({"uid": tw.uid, "label": label, "text": clean_tweet(tw.tokens),
                         "cmi": round(code_mixing_index(tw.langs), 2)})
        with (out_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["uid", "label", "text", "cmi"])
            writer.writeheader()
            writer.writerows(rows)
        rows_by_split[split] = rows
        print(f"sentimix/{split}: {len(rows)} tweets -> {out_dir / f'{split}.csv'}")
    return rows_by_split


def load_sentimix(split: str, cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Load a processed SentiMix split as a list of dicts (uid, label, text, cmi)."""
    cfg = cfg or load_config(CONFIG_DIR / "data.yaml")
    path = PROJECT_ROOT / cfg["sentimix"]["out_dir"] / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m minigpt.data --prepare` first")
    with path.open(encoding="utf-8", newline="") as f:
        return [{**row, "cmi": float(row["cmi"])} for row in csv.DictReader(f)]


# --------------------------------------------------------------------------- HingCorpus


def _iter_archive_lines(archive: Path, url: str) -> Iterator[str]:
    """Yield text lines from the HingCorpus tar.gz, streaming it.

    Uses the local archive if present (a partial download is fine as long as it
    holds enough lines), otherwise streams straight from the URL.
    """
    fileobj = archive.open("rb") if archive.exists() else urllib.request.urlopen(url)  # noqa: S310
    print(f"reading corpus from {archive if archive.exists() else url}")
    try:
        with fileobj, tarfile.open(fileobj=fileobj, mode="r|gz") as tf:
            for member in tf:
                if member.isfile() and member.name.endswith(".txt"):
                    for raw in tf.extractfile(member):  # type: ignore[union-attr]
                        yield raw.decode("utf-8", errors="replace")
    except (EOFError, tarfile.ReadError, zlib.error) as e:
        print(f"warning: archive ended early ({type(e).__name__}); using the lines read so far")


def prepare_pretrain(cfg: dict[str, Any], eval_texts: Iterable[str], seed: int) -> dict[str, Any]:
    """Take the first ``max_lines`` usable corpus lines, clean, dedupe, split train/val.

    Lines are dropped if too short, exact duplicates, or if they match a SentiMix
    val/test tweet (so pre-training never sees evaluation text).
    """
    n_chars = cfg["leakage_prefix_chars"]
    eval_keys = {k for k in (leakage_key(t, n_chars) for t in eval_texts) if len(k) == n_chars}
    out_dir = PROJECT_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    seen: set[int] = set()
    dropped = Counter()
    kept = Counter()
    words_per_line: Counter[int] = Counter()
    n_words = n_chars_total = 0
    with (out_dir / "train.txt").open("w", encoding="utf-8", newline="\n") as f_train, \
         (out_dir / "val.txt").open("w", encoding="utf-8", newline="\n") as f_val:
        for raw in _iter_archive_lines(PROJECT_ROOT / cfg["archive"], cfg["url"]):
            line = normalize_text(raw)
            n = len(line.split())
            if n < cfg["min_words"]:
                dropped["too_short"] += 1
                continue
            h = hash(line)
            if h in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(h)
            if leakage_key(line, n_chars) in eval_keys:
                dropped["matches_eval_tweet"] += 1
                continue
            split = "val" if rng.random() < cfg["val_fraction"] else "train"
            (f_val if split == "val" else f_train).write(line + "\n")
            kept[split] += 1
            words_per_line[n] += 1
            n_words += n
            n_chars_total += len(line)
            if sum(kept.values()) % 250_000 == 0:
                print(f"  kept {sum(kept.values()):,} lines")
            if sum(kept.values()) >= cfg["max_lines"]:
                break

    stats = {"kept": dict(kept), "dropped": dict(dropped), "words": n_words, "chars": n_chars_total,
             "words_per_line": {str(k): v for k, v in sorted(words_per_line.items())}}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"pretrain: {kept['train']:,} train / {kept['val']:,} val lines -> {out_dir}")
    if sum(kept.values()) < cfg["max_lines"]:
        print(f"warning: only {sum(kept.values()):,} of {cfg['max_lines']:,} requested lines available")
    return stats


# --------------------------------------------------------------------------- report


def _describe(lengths: list[int]) -> str:
    q = statistics.quantiles(lengths, n=20)  # q[18] = 95th percentile
    return (f"mean {statistics.fmean(lengths):5.1f} | median {statistics.median(lengths):4.0f} | "
            f"p95 {q[18]:4.0f} | max {max(lengths):4d}")


def _expand_hist(hist: dict[str, int]) -> list[int]:
    return [int(k) for k, v in hist.items() for _ in range(v)]


def save_length_plot(sentimix_lengths: list[int], corpus_lengths: list[int], path: Path) -> None:
    """Two-panel histogram of words per text: SentiMix train vs. the pre-training slice."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series, surface, ink, ink_muted, grid = "#2a78d6", "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=surface)
    panels = [(axes[0], sentimix_lengths, "SentiMix train (labeled tweets)"),
              (axes[1], corpus_lengths, "HingCorpus slice (pre-training)")]
    for ax, lengths, title in panels:
        cap = int(statistics.quantiles(lengths, n=100)[98])  # clip the long tail at p99
        ax.hist([min(x, cap) for x in lengths], bins=range(0, cap + 2), color=series,
                edgecolor=surface, linewidth=1.0)
        ax.set_facecolor(surface)
        ax.set_title(title, loc="left", color=ink, fontsize=11)
        ax.set_xlabel(f"words per text (clipped at p99 = {cap})", color=ink_muted, fontsize=9)
        ax.set_ylabel("texts", color=ink_muted, fontsize=9)
        ax.axvline(statistics.median(lengths), color=ink_muted, linewidth=1, linestyle="--")
        ax.text(statistics.median(lengths), ax.get_ylim()[1] * 0.95, f" median {statistics.median(lengths):.0f}",
                color=ink_muted, fontsize=8, va="top")
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=ink_muted, labelsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def print_report(cfg: dict[str, Any]) -> None:
    """Print sizes, class balance, examples and length stats for both datasets."""
    labels = cfg["sentimix"]["labels"]
    data = {split: load_sentimix(split, cfg) for split in SPLITS}

    print("=" * 78)
    print("SentiMix Hinglish (SemEval-2020 Task 9) - labeled sentiment")
    print("=" * 78)
    header = f"{'split':<6}{'rows':>7}" + "".join(f"{lab:>16}" for lab in labels) + f"{'empty':>7}"
    print(header)
    for split, rows in data.items():
        counts = Counter(r["label"] for r in rows)
        cells = "".join(f"{counts[lab]:>8} ({100 * counts[lab] / len(rows):4.1f}%)" for lab in labels)
        empty = sum(1 for r in rows if not r["text"])
        print(f"{split:<6}{len(rows):>7}{cells}{empty:>7}")

    print("\nWords per tweet (after cleaning):")
    for split, rows in data.items():
        print(f"  {split:<6}{_describe([len(r['text'].split()) for r in rows])}")
    print("\nCode-Mixing Index (0 = one language, 50 = even Hindi/English mix):")
    for split, rows in data.items():
        cmis = [r["cmi"] for r in rows]
        mixed = sum(1 for c in cmis if c > 0) / len(cmis)
        print(f"  {split:<6}mean CMI {statistics.fmean(cmis):5.1f} | {100 * mixed:4.1f}% of tweets are code-mixed")

    print("\nExamples (train):")
    rng = random.Random(cfg["seed"])
    for lab in labels:
        pool = [r for r in data["train"] if r["label"] == lab and len(r["text"].split()) >= 5]
        for r in rng.sample(pool, cfg["report"]["examples_per_class"]):
            print(f"  [{lab:>8}] {r['text'][:110]}")

    stats_path = PROJECT_ROOT / cfg["pretrain"]["out_dir"] / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} missing - run `python -m minigpt.data --prepare` first")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    corpus_lengths = _expand_hist(stats["words_per_line"])
    kept = stats["kept"]
    print("\n" + "=" * 78)
    print("L3Cube-HingCorpus slice - unlabeled pre-training text (CC BY-NC-SA 4.0)")
    print("=" * 78)
    print(f"  lines     train {kept.get('train', 0):,} | val {kept.get('val', 0):,}")
    print(f"  size      {stats['words'] / 1e6:.1f}M words | {stats['chars'] / 1e6:.0f}M characters")
    print("  dropped   " +" | ".join(f"{k} {v:,}" for k, v in stats["dropped"].items()))
    print(f"  words/line {_describe(corpus_lengths)}")
    val_path = PROJECT_ROOT / cfg["pretrain"]["out_dir"] / "val.txt"
    with val_path.open(encoding="utf-8") as f:
        sample = [next(f).rstrip("\n") for _ in range(4)]
    print("  examples (val):")
    for line in sample:
        print(f"    {line[:110]}")

    plot_path = PROJECT_ROOT / cfg["report"]["plot_path"]
    save_length_plot([len(r["text"].split()) for r in data["train"]], corpus_lengths, plot_path)
    print(f"\nLength plot saved to {plot_path.relative_to(PROJECT_ROOT)}")


# --------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 data pipeline")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "data.yaml")
    parser.add_argument("--prepare", action="store_true", help="download, clean and split the data")
    parser.add_argument("--report", action="store_true", help="print the data report")
    args = parser.parse_args()
    if not (args.prepare or args.report):
        parser.error("pass --prepare and/or --report")

    cfg = load_config(args.config)
    if args.prepare:
        rows = prepare_sentimix(cfg["sentimix"])
        eval_texts = [r["text"] for split in ("val", "test") for r in rows[split]]
        prepare_pretrain(cfg["pretrain"], eval_texts, seed=cfg["seed"])
    if args.report:
        print_report(cfg)


if __name__ == "__main__":
    main()
