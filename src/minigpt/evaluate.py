"""Classification metrics and plots shared by every stage (LSTM, GPT, LLM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Chart tokens (light surface): recessive ink and grid, categorical slots 1-2, blue ramp.
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]) -> dict[str, Any]:
    """Accuracy, macro F1, per-class F1 and the confusion matrix (rows = true class)."""
    idx = list(range(len(labels)))
    per_class = f1_score(y_true, y_pred, labels=idx, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)),
        "f1_per_class": {lab: float(f) for lab, f in zip(labels, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def plot_confusion_matrix(cm: Sequence[Sequence[int]], labels: Sequence[str], path: Path, title: str) -> None:
    """Row-normalized confusion matrix (each cell = share of the true class), counts in brackets."""
    from matplotlib.colors import LinearSegmentedColormap

    plt = _plt()
    cm_arr = np.asarray(cm, dtype=float)
    norm = cm_arr / cm_arr.sum(axis=1, keepdims=True).clip(min=1)
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=SURFACE)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "#ffffff" if norm[i, j] > 0.55 else INK
            ax.text(j, i, f"{norm[i, j]:.0%}\n({int(cm_arr[i, j])})", ha="center", va="center",
                    color=color, fontsize=9)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("true", color=INK_MUTED, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_training_curves(history: list[dict[str, float]], path: Path, title: str) -> None:
    """Two panels: train vs. val loss, and val macro F1, per epoch."""
    plt = _plt()
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)

    for key, name, color in (("train_loss", "train", SERIES[0]), ("val_loss", "val", SERIES[1])):
        values = [h[key] for h in history]
        ax_loss.plot(epochs, values, color=color, linewidth=2, marker="o", markersize=4, label=name)
        ax_loss.annotate(name, (epochs[-1], values[-1]), textcoords="offset points", xytext=(7, 0),
                         color=INK_MUTED, fontsize=8, va="center")
    ax_loss.set_title("Cross-entropy loss", loc="left", color=INK, fontsize=11)
    ax_loss.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)

    f1 = [h["val_macro_f1"] for h in history]
    best = int(np.argmax(f1))
    ax_f1.plot(epochs, f1, color=SERIES[0], linewidth=2, marker="o", markersize=4)
    ax_f1.annotate(f"best {f1[best]:.3f}", (epochs[best], f1[best]), textcoords="offset points",
                   xytext=(0, 8), ha="center", color=INK_MUTED, fontsize=8)
    ax_f1.set_title("Validation macro F1", loc="left", color=INK, fontsize=11)

    for ax in (ax_loss, ax_f1):
        _style(ax)
        ax.set_xlabel("epoch", color=INK_MUTED, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
