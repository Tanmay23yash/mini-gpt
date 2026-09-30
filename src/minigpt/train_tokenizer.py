"""Train the BPE tokenizer on the pre-training corpus and encode the corpus to token ids.

Usage::

    python -m minigpt.train_tokenizer                 # learn merges, save, print stats
    python -m minigpt.train_tokenizer --encode        # also write train.bin / val.bin

``.bin`` files are flat uint16 arrays of token ids, one tweet after another, each
followed by <|endoftext|>. The GPT training loop reads random windows from them.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import time
from pathlib import Path

import numpy as np

from minigpt.env import CONFIG_DIR, PROJECT_ROOT
from minigpt.tokenizer.bpe import BPETokenizer
from minigpt.utils import load_config

ENCODE_BATCH_LINES = 100_000


def read_lines(path: Path, limit: int | None = None) -> list[str]:
    """Read up to ``limit`` lines (without the trailing newline)."""
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in itertools.islice(f, limit)]


def print_stats(tok: BPETokenizer, texts: list[str]) -> None:
    """Compression and example segmentations on held-out text."""
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    lengths = [len(tok.encode(t)) for t in texts]
    words = [w for t in texts for w in t.split()]
    single = sum(1 for w in words if len(tok.encode(" " + w)) == 1)
    print(f"vocab size        {len(tok):,} (incl. special tokens)")
    print(f"bytes per token   {n_bytes / sum(lengths):.2f}")
    print(f"tokens per tweet  mean {statistics.fmean(lengths):.1f} | p95 {statistics.quantiles(lengths, n=20)[18]:.0f}"
          f" | max {max(lengths)}")
    print(f"words that are 1 token: {single / len(words):.1%}")
    print(f"longest tokens    {sorted((tok.token_str(i) for i in tok.merges.values()), key=len)[-8:]}")
    print("examples (| marks token boundaries):")
    for text in texts[:5]:
        print("   ", "|".join(tok.token_str(i) for i in tok.encode(text))[:150])


def encode_file(tok: BPETokenizer, src: Path, dst: Path) -> int:
    """Encode every line of ``src`` into ``dst`` as uint16 ids, <|endoftext|> after each line."""
    assert len(tok) <= np.iinfo(np.uint16).max + 1, "vocab too large for uint16"
    total = 0
    with src.open(encoding="utf-8") as f_in, dst.open("wb") as f_out:
        while batch := list(itertools.islice(f_in, ENCODE_BATCH_LINES)):
            ids: list[int] = []
            for line in batch:
                ids.extend(tok.encode(line.rstrip("\n")))
                ids.append(tok.eot_id)
            np.asarray(ids, dtype=np.uint16).tofile(f_out)
            total += len(ids)
            print(f"  {dst.name}: {total:,} tokens")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE and encode the corpus")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "bpe.yaml")
    parser.add_argument("--encode", action="store_true", help="also encode train/val to .bin files")
    parser.add_argument("--skip-train", action="store_true", help="load the saved tokenizer instead of training")
    args = parser.parse_args()
    cfg = load_config(args.config)
    corpus = PROJECT_ROOT / cfg["corpus_dir"]
    out_path = PROJECT_ROOT / cfg["output"]

    if args.skip_train:
        tok = BPETokenizer.load(out_path)
    else:
        texts = read_lines(corpus / "train.txt", cfg["train_lines"])
        print(f"training BPE on {len(texts):,} lines -> vocab {cfg['vocab_size']:,}")
        start = time.perf_counter()
        tok = BPETokenizer()
        tok.train(texts, cfg["vocab_size"], verbose=True)
        print(f"trained in {(time.perf_counter() - start) / 60:.1f} min")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(out_path)
        print(f"saved {out_path.relative_to(PROJECT_ROOT)}")

    print_stats(tok, read_lines(corpus / "val.txt", 2000))

    if args.encode:
        for split in ("train", "val"):
            start = time.perf_counter()
            n = encode_file(tok, corpus / f"{split}.txt", corpus / f"{split}.bin")
            print(f"{split}.bin: {n:,} tokens in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
