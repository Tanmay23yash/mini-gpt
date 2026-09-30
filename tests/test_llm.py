"""Tests for Stage 3 prompting and label scoring (src/minigpt/llm.py). No model download needed."""

import math

import pytest
import torch

from minigpt.llm import build_messages, label_scores, label_token_ids, select_fewshot

LABELS = ["negative", "neutral", "positive"]


class FakeTokenizer:
    """Maps a few words to fixed ids; anything else becomes two tokens."""

    vocab = {"negative": 6, "Negative": 5, "neutral": 7, "Neutral": 8, "positive": 9, "Positive": 10}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab[text]] if text in self.vocab else [1, 2]


def _rows() -> list[dict]:
    words = "bhai yeh match bahut accha tha yaar kal phir"
    return [{"text": f"{words} {i}", "label": LABELS[i % 3]} for i in range(60)]


def test_build_messages_orders_system_examples_then_tweet() -> None:
    msgs = build_messages("kya scene hai", "SYS", [("ex1", "positive"), ("ex2", "negative")])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[0]["content"] == "SYS"
    assert msgs[2]["content"] == "positive"
    assert msgs[-1] == {"role": "user", "content": "kya scene hai"}
    assert len(build_messages("x", "SYS")) == 2  # zero-shot


def test_select_fewshot_is_balanced_and_reproducible() -> None:
    ex = select_fewshot(_rows(), 6, LABELS, seed=1)
    assert len(ex) == 6
    assert sorted(lab for _, lab in ex) == sorted(LABELS * 2)
    assert ex == select_fewshot(_rows(), 6, LABELS, seed=1)
    assert ex != select_fewshot(_rows(), 6, LABELS, seed=2)


def test_select_fewshot_requires_multiple_of_classes() -> None:
    with pytest.raises(ValueError):
        select_fewshot(_rows(), 4, LABELS, seed=0)


def test_label_token_ids_collects_both_casings_lowercase_first() -> None:
    ids = label_token_ids(FakeTokenizer(), LABELS)
    # "Negative" has the smaller id here; lowercase must still come first (it is the LoRA target)
    assert ids == {"negative": [6, 5], "neutral": [7, 8], "positive": [9, 10]}


def test_label_token_ids_rejects_multi_token_labels() -> None:
    with pytest.raises(ValueError):
        label_token_ids(FakeTokenizer(), ["negative", "mixed"])


def test_label_scores_sum_variant_probabilities() -> None:
    ids = {"negative": [0, 1], "neutral": [2], "positive": [3]}
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.3, 0.1]])  # token 4 is not a label
    per_label, mass = label_scores(probs.log(), ids)
    torch.testing.assert_close(per_label.exp(), torch.tensor([[0.3, 0.3, 0.3]]))
    assert mass.item() == pytest.approx(0.9)


def test_label_scores_argmax_picks_most_likely_label() -> None:
    ids = {"negative": [0], "neutral": [1], "positive": [2]}
    logits = torch.tensor([[0.0, 1.0, 3.0, 10.0], [5.0, 1.0, 0.0, 0.0]])  # token 3: not a label
    per_label, mass = label_scores(logits, ids)
    assert per_label.argmax(-1).tolist() == [2, 0]
    assert mass[0].item() < 0.01  # most mass on a non-label token
    z = math.exp(5) + math.exp(1) + 2                                  # softmax denominator of row 2
    assert mass[1].item() == pytest.approx((math.exp(5) + math.exp(1) + 1) / z)


def test_calibrate_removes_a_constant_label_bias() -> None:
    from minigpt.llm import calibrate

    # the model adds the same bias to every tweet: neutral is always 10x less likely
    true_probs = torch.tensor([[0.2, 0.5, 0.3], [0.6, 0.2, 0.2]])
    bias = torch.tensor([1.0, 0.1, 1.0])
    biased = (true_probs * bias) / (true_probs * bias).sum(-1, keepdim=True)
    prior = bias / bias.sum()                               # what content-free inputs would show
    assert biased[0].argmax() != 1                          # biased: neutral lost
    fixed = calibrate(biased.log(), prior)
    assert fixed.argmax(-1).tolist() == true_probs.argmax(-1).tolist()
