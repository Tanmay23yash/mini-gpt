"""Tests for sampling and generation (src/minigpt/generate.py)."""

import math
from types import SimpleNamespace

import pytest
import torch

from minigpt.generate import filter_logits, generate, sample_next
from minigpt.model.gpt import GPT, GPTConfig


def _logits_from_probs(probs: list[float]) -> torch.Tensor:
    return torch.tensor([[math.log(p) for p in probs]])


def test_top_k_keeps_k_largest() -> None:
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    out = filter_logits(logits, top_k=2)
    assert torch.isfinite(out).tolist() == [[False, True, False, True, False]]


def test_top_p_keeps_smallest_set_reaching_p() -> None:
    logits = _logits_from_probs([0.05, 0.5, 0.15, 0.3])      # sorted: 0.5, 0.3, 0.15, 0.05
    assert torch.isfinite(filter_logits(logits, top_p=0.7)).tolist() == [[False, True, False, True]]
    assert torch.isfinite(filter_logits(logits, top_p=0.9)).tolist() == [[False, True, True, True]]
    assert torch.isfinite(filter_logits(logits, top_p=0.1)).tolist() == [[False, True, False, False]]


def test_filter_does_not_modify_input() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    filter_logits(logits, top_k=1, top_p=0.5)
    assert torch.isfinite(logits).all()


def test_temperature_zero_is_greedy() -> None:
    logits = torch.tensor([[0.1, 2.0, 1.9], [3.0, 0.0, 0.0]])
    assert sample_next(logits, temperature=0).tolist() == [[1], [0]]


def test_sampling_follows_the_distribution() -> None:
    logits = _logits_from_probs([0.7, 0.2, 0.1]).repeat(4000, 1)
    draws = sample_next(logits, generator=torch.Generator().manual_seed(0)).squeeze(-1)
    freq = torch.bincount(draws, minlength=3).float() / len(draws)
    torch.testing.assert_close(freq, torch.tensor([0.7, 0.2, 0.1]), atol=0.03, rtol=0)


def _tiny_gpt() -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=30, block_size=8, d_model=16, n_layers=1, n_heads=2, dropout=0.0))


def test_generate_keeps_prompt_and_adds_tokens() -> None:
    prompt = torch.tensor([[1, 2, 3]])
    out = generate(_tiny_gpt(), prompt, max_new_tokens=5)
    assert out.shape == (1, 8)
    assert out[:, :3].tolist() == [[1, 2, 3]]


def test_generate_crops_context_to_block_size() -> None:
    out = generate(_tiny_gpt(), torch.randint(0, 30, (2, 20)), max_new_tokens=3)  # prompt > block_size 8
    assert out.shape == (2, 23)


def test_generate_is_reproducible_with_a_seed() -> None:
    model, prompt = _tiny_gpt(), torch.tensor([[1, 2]])
    a = generate(model, prompt, 10, generator=torch.Generator().manual_seed(3))
    b = generate(model, prompt, 10, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_generate_stops_at_eos_and_pads_finished_rows() -> None:
    class AlwaysEos(torch.nn.Module):
        config = SimpleNamespace(block_size=8)

        def forward(self, idx: torch.Tensor) -> SimpleNamespace:
            logits = torch.full((idx.size(0), idx.size(1), 5), -10.0)
            logits[..., 4] = 10.0                                   # token 4 = eos, always
            return SimpleNamespace(logits=logits)

    out = generate(AlwaysEos(), torch.tensor([[1], [2]]), max_new_tokens=10, temperature=0, eos_id=4)
    assert out.tolist() == [[1, 4], [2, 4]]                        # stopped after one step


def test_generate_restores_training_mode() -> None:
    model = _tiny_gpt().train()
    generate(model, torch.tensor([[1]]), 2)
    assert model.training
