"""Text generation from a trained GPT: greedy, temperature, top-k and top-p (nucleus) sampling.

Usage::

    python -m minigpt.generate --prompt "bhai aaj" --n 3                  # sample with defaults
    python -m minigpt.generate --prompt "yaar" --temperature 0            # greedy
    python -m minigpt.generate --demo                                     # settings grid -> results/stage2/samples.md

Autoregressive loop: feed the tokens so far, take the logits of the LAST position,
turn them into a distribution, draw one token, append it, repeat.

- temperature T: logits / T. T < 1 sharpens (safer, repetitive), T > 1 flattens
  (more diverse, more nonsense). T = 0 means greedy argmax.
- top-k: keep only the k most likely tokens.
- top-p: keep the smallest set of tokens whose probabilities add up to >= p. Adapts
  to the model's confidence: 1-2 tokens when sure, many when unsure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from minigpt.env import CONFIG_DIR, PROJECT_ROOT
from minigpt.tokenizer.bpe import BPETokenizer
from minigpt.utils import get_device, load_config


def filter_logits(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Set logits outside the top-k / top-p set to -inf. ``logits``: (B, V). Returns a new tensor."""
    logits = logits.clone()
    if top_k is not None and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, [-1]]            # (B, 1) k-th largest
        logits[logits < kth] = float("-inf")
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        mass_before = probs.cumsum(dim=-1) - probs                         # probability of strictly better tokens
        remove_sorted = mass_before >= top_p                               # the top token always survives
        remove = remove_sorted.scatter(-1, order, remove_sorted)           # back to vocab order
        logits[remove] = float("-inf")
    return logits


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """Pick the next token for each row of ``logits`` (B, V). Returns (B, 1)."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = F.softmax(filter_logits(logits / temperature, top_k, top_p), dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None, eos_id: int | None = None,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """Extend ``idx`` (B, T) by up to ``max_new_tokens``. Sequences that emit ``eos_id`` keep
    emitting it (so rows stay the same length); stops early once every row has finished."""
    was_training = model.training
    model.eval()
    block_size = model.config.block_size
    finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:]).logits[:, -1, :]             # (B, V): last position only
        nxt = sample_next(logits.float(), temperature, top_k, top_p, generator)
        if eos_id is not None:
            nxt[finished] = eos_id
            finished |= nxt.squeeze(-1) == eos_id
        idx = torch.cat([idx, nxt], dim=1)
        if eos_id is not None and finished.all():
            break
    model.train(was_training)
    return idx


def generate_text(model: nn.Module, tok: BPETokenizer, prompt: str, n: int = 1, max_new_tokens: int = 60,
                  temperature: float = 0.8, top_k: int | None = 50, top_p: float | None = None,
                  seed: int | None = None) -> list[str]:
    """Generate ``n`` continuations of ``prompt`` (each stops at <|endoftext|>)."""
    device = next(model.parameters()).device
    ids = tok.encode(prompt) or [tok.eot_id]                              # empty prompt: start of a new tweet
    idx = torch.tensor([ids] * n, device=device)
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    out = generate(model, idx, max_new_tokens, temperature, top_k, top_p, tok.eot_id, gen)
    texts = []
    for row in out.tolist():
        new = row[len(ids):]
        if tok.eot_id in new:
            new = new[:new.index(tok.eot_id)]
        texts.append(prompt + tok.decode(new))
    return texts


def load_generation_setup(checkpoint: str | Path, tokenizer: str | Path, device: torch.device
                          ) -> tuple[nn.Module, BPETokenizer]:
    from minigpt.train import load_model

    return load_model(PROJECT_ROOT / checkpoint, device).eval(), BPETokenizer.load(PROJECT_ROOT / tokenizer)


def write_demo(cfg: dict[str, Any], model: nn.Module, tok: BPETokenizer, path: Path) -> None:
    """Every prompt x every sampling setting, as a Markdown file for the README."""
    lines = ["# Sample generations", "",
             f"Model: `{cfg['checkpoint']}`. Seed {cfg['seed']} + prompt index; each cell is one sample, "
             f"cut at <|endoftext|> or {cfg['max_new_tokens']} tokens.", ""]
    settings = cfg["settings"]
    for i, prompt in enumerate(cfg["prompts"]):
        # A different seed per prompt: with one shared seed every prompt gets the same random
        # draws, and at high temperature the same phrases show up after every prompt.
        seed = cfg["seed"] + i
        lines += [f"## `{prompt}`", "", "| setting | output |", "|---|---|"]
        for s in settings:
            text = generate_text(model, tok, prompt, 1, cfg["max_new_tokens"], s["temperature"], s.get("top_k"),
                                 s.get("top_p"), seed=seed)[0]
            cell = text.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {s['name']} | {cell} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with the mini-GPT")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "generate.yaml")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--demo", action="store_true", help="write the prompt x settings grid to results/")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model, tok = load_generation_setup(cfg["checkpoint"], cfg["tokenizer"], get_device())

    if args.demo:
        out = PROJECT_ROOT / cfg["demo_output"]
        write_demo(cfg, model, tok, out)
        print(out.read_text(encoding="utf-8"))
        return
    d = cfg["defaults"]
    texts = generate_text(
        model, tok, args.prompt, args.n, cfg["max_new_tokens"],
        d["temperature"] if args.temperature is None else args.temperature,
        d["top_k"] if args.top_k is None else args.top_k,
        d["top_p"] if args.top_p is None else args.top_p,
        cfg["seed"] if args.seed is None else args.seed,
    )
    for i, text in enumerate(texts, 1):
        print(f"[{i}] {text}")


if __name__ == "__main__":
    main()
