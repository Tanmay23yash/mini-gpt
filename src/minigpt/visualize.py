"""Attention maps and per-head statistics for a trained GPT.

Usage::

    python -m minigpt.visualize            # uses configs/visualize.yaml -> results/stage2/attention/

Outputs:
    attention_<n>.png   every layer x head for one sentence (row = layer, column = head)
    head_stats.png      what each head does on average over validation tweets
    head_stats.csv      the same numbers

Head statistics (averaged over query positions i >= 1 and many tweets):
    prev_token   weight on position i-1      -> "previous-token heads"
    first_token  weight on position 0        -> "attention sink" (the <|endoftext|> we prepend)
    self         weight on position i itself
    distance     sum_j w_ij * (i - j)        -> how far back the head looks, in tokens
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from minigpt.env import CONFIG_DIR, PROJECT_ROOT
from minigpt.evaluate import BLUE_RAMP, INK, INK_MUTED, SURFACE, _plt
from minigpt.tokenizer.bpe import BPETokenizer
from minigpt.utils import get_device, load_config

STATS = ("prev_token", "first_token", "self", "distance")


@torch.no_grad()
def attention_for(model: nn.Module, tok: BPETokenizer, text: str, max_len: int = 64
                  ) -> tuple[list[str], np.ndarray]:
    """Tokens (with a leading <|endoftext|>, as in pre-training) and weights (L, H, T, T)."""
    device = next(model.parameters()).device
    ids = [tok.eot_id] + tok.encode(text)[: max_len - 1]
    out = model(torch.tensor([ids], device=device), return_attn=True)
    weights = torch.stack([w[0] for w in out.attentions]).float().cpu().numpy()   # (L, H, T, T)
    labels = ["<eot>"] + [tok.token_str(i).strip() or "␣" for i in ids[1:]]
    return labels, weights


def head_statistics(model: nn.Module, tok: BPETokenizer, texts: list[str], max_len: int = 64) -> dict[str, np.ndarray]:
    """Average each statistic over all query positions i >= 1 of all texts -> {stat: (L, H)}."""
    sums: dict[str, np.ndarray] = {}
    count = 0
    for text in texts:
        _, w = attention_for(model, tok, text, max_len)                  # (L, H, T, T)
        t = w.shape[-1]
        if t < 3:
            continue
        rows = np.arange(1, t)
        dist = np.arange(t)[:, None] - np.arange(t)[None, :]              # (T, T): i - j
        per_text = {
            "prev_token": w[..., rows, rows - 1].sum(-1),
            "first_token": w[..., rows, 0].sum(-1),
            "self": w[..., rows, rows].sum(-1),
            "distance": (w * dist)[..., rows, :].sum(-1).sum(-1),
        }
        for k, v in per_text.items():
            sums[k] = sums.get(k, 0) + v
        count += len(rows)
    return {k: v / count for k, v in sums.items()}


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)


def plot_attention_grid(labels: list[str], weights: np.ndarray, path: Path, title: str) -> None:
    """Row = layer, column = head. Each panel: query token (row) attends to key token (column)."""
    plt = _plt()
    n_layers, n_heads, t, _ = weights.shape
    size = max(2.4, 0.26 * t)
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(size * n_heads + 1.2, size * n_layers + 0.8),
                             facecolor=SURFACE, squeeze=False)
    cmap = _cmap()
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            im = ax.imshow(weights[layer, head], cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks(range(t))
            ax.set_yticks(range(t))
            ax.set_xticklabels(labels if layer == n_layers - 1 else [], rotation=90, fontsize=7, color=INK_MUTED)
            ax.set_yticklabels(labels if head == 0 else [], fontsize=7, color=INK_MUTED)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if layer == 0:
                ax.set_title(f"head {head}", color=INK_MUTED, fontsize=9)
            if head == 0:
                ax.set_ylabel(f"layer {layer}", color=INK_MUTED, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label("attention weight", color=INK_MUTED, fontsize=8)
    cbar.ax.tick_params(labelsize=7, colors=INK_MUTED)
    cbar.outline.set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_head_stats(stats: dict[str, np.ndarray], path: Path, n_texts: int) -> None:
    """One small layer x head heatmap per statistic, values written in each cell."""
    plt = _plt()
    titles = {"prev_token": "weight on previous token", "first_token": "weight on first token (sink)",
              "self": "weight on itself", "distance": "mean distance looked back (tokens)"}
    fig, axes = plt.subplots(1, len(STATS), figsize=(4.0 * len(STATS), 3.9), facecolor=SURFACE)
    cmap = _cmap()
    for ax, key in zip(axes, STATS):
        m = stats[key]
        vmax = 1.0 if key != "distance" else float(np.ceil(m.max()))
        ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax)
        for (layer, head), v in np.ndenumerate(m):
            ax.text(head, layer, f"{v:.2f}" if key != "distance" else f"{v:.1f}", ha="center", va="center",
                    fontsize=8, color="#ffffff" if v / vmax > 0.55 else INK)
        ax.set_xticks(range(m.shape[1]), [f"h{h}" for h in range(m.shape[1])])
        ax.set_yticks(range(m.shape[0]), [f"L{layer}" for layer in range(m.shape[0])])
        ax.tick_params(length=0, labelsize=8, colors=INK_MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(titles[key], loc="left", color=INK, fontsize=10)
    fig.suptitle(f"What each attention head does (averaged over {n_texts} validation tweets)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    from minigpt.train import load_model

    model = load_model(PROJECT_ROOT / cfg["checkpoint"], get_device()).eval()
    tok = BPETokenizer.load(PROJECT_ROOT / cfg["tokenizer"])
    out_dir = PROJECT_ROOT / cfg["output_dir"]

    for n, sentence in enumerate(cfg["sentences"], 1):
        labels, weights = attention_for(model, tok, sentence)
        plot_attention_grid(labels, weights, out_dir / f"attention_{n}.png", f'"{sentence}"')

    val_path = PROJECT_ROOT / cfg["stats_corpus"]
    with val_path.open(encoding="utf-8") as f:
        texts = [line.rstrip("\n") for _, line in zip(range(cfg["stats_n_texts"]), f)]
    stats = head_statistics(model, tok, texts)
    plot_head_stats(stats, out_dir / "head_stats.png", len(texts))
    with (out_dir / "head_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", *STATS])
        n_layers, n_heads = stats[STATS[0]].shape
        for layer in range(n_layers):
            for head in range(n_heads):
                w.writerow([layer, head, *(round(float(stats[s][layer, head]), 4) for s in STATS)])
    print(f"wrote {len(cfg['sentences'])} attention grids + head stats to {out_dir}")
    for key in STATS:
        m = stats[key]
        layer, head = np.unravel_index(m.argmax(), m.shape)
        print(f"  max {key:<12} L{layer} h{head} = {m.max():.2f}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention maps and head statistics")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "visualize.yaml")
    run(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
