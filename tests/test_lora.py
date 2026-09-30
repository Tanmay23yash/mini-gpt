"""Tests for LoRA fine-tuning (src/minigpt/lora.py) on a tiny random model - no download needed."""

from contextlib import nullcontext

import pytest
import torch

pytest.importorskip("peft")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from minigpt.lora import LabelLoss, add_lora, make_batches  # noqa: E402
from minigpt.train import train_step  # noqa: E402

LORA_CFG = {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": "all-linear", "gradient_checkpointing": False}


def _tiny_llm() -> LlamaForCausalLM:
    torch.manual_seed(0)
    # initializer_range 0.5 (not 0.02): after the final RMSNorm the hidden vector has a fixed norm,
    # so a tiny random LM head caps the logit gap (~0.6) and no adapter could push the loss down.
    # Trained LLMs have large, meaningful LM-head weights; this mimics that.
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
                      initializer_range=0.5)
    return LlamaForCausalLM(cfg)


def _batch() -> dict[str, torch.Tensor]:
    ids = torch.randint(3, 64, (4, 10))
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_lora_starts_as_the_base_model() -> None:
    """B is initialized to zero, so the adapted model's output equals the base model's."""
    base, batch = _tiny_llm().eval(), _batch()
    with torch.no_grad():
        before = base(**batch).logits.clone()
    model = add_lora(base, LORA_CFG).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**batch).logits, before)


def test_only_lora_weights_are_trainable() -> None:
    model = add_lora(_tiny_llm(), LORA_CFG)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_train < 0.25 * sum(p.numel() for p in model.parameters())


def test_label_loss_uses_last_position_logits() -> None:
    model = _tiny_llm().eval()
    wrapped = LabelLoss(model, torch.tensor([10, 20, 30]))
    batch, y = _batch(), torch.tensor([0, 1, 2, 0])
    out = wrapped(batch, targets=y)
    logits = model(**batch).logits[:, -1]
    torch.testing.assert_close(out.logits, logits)
    torch.testing.assert_close(out.loss, torch.nn.functional.cross_entropy(logits, torch.tensor([10, 20, 30, 10])))


def test_train_step_learns_the_label_and_keeps_base_frozen() -> None:
    model = add_lora(_tiny_llm(), LORA_CFG)
    model.train()
    frozen_before = model.base_model.model.model.layers[0].self_attn.q_proj.base_layer.weight.clone()
    wrapped = LabelLoss(model, torch.tensor([10, 20, 30]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch, y = _batch(), torch.tensor([0, 1, 2, 0])
    first, _ = train_step(wrapped, batch, y, opt, scaler, nullcontext(), 1.0)
    for _ in range(40):
        loss, _ = train_step(wrapped, batch, y, opt, scaler, nullcontext(), 1.0)
    assert loss < 0.2 * first
    torch.testing.assert_close(model.base_model.model.model.layers[0].self_attn.q_proj.base_layer.weight, frozen_before)


def test_make_batches_covers_everything_once_and_reshuffles() -> None:
    prompts, ys = [f"p{i}" for i in range(10)], list(range(10))
    b1 = make_batches(prompts, ys, 4, seed=1)
    assert [len(p) for p, _ in b1] == [4, 4, 2]
    assert sorted(y for _, y_list in b1 for y in y_list) == ys
    assert all(int(p[1:]) == y for ps, ys_ in b1 for p, y in zip(ps, ys_))   # pairs stay aligned
    assert b1 != make_batches(prompts, ys, 4, seed=2)
