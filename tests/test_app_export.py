"""The exported demo models must reproduce the published test results exactly.

Skipped unless both the exported models (python -m minigpt.export_app) and the processed
data (python -m minigpt.data --prepare) exist, e.g. on a fresh clone.
"""

import json

import pytest
import torch

from minigpt.env import PROJECT_ROOT, RESULTS_DIR

MODELS = PROJECT_ROOT / "app" / "models"
DATA = PROJECT_ROOT / "data" / "processed" / "sentimix" / "test.csv"
pytestmark = pytest.mark.skipif(not (MODELS / "model_info.json").exists() or not DATA.exists(),
                                reason="needs app/models and processed SentiMix data")


def _test_rows():
    from minigpt.data import load_sentimix

    return load_sentimix("test")


def test_tfidf_export_reproduces_stage1_score() -> None:
    import joblib

    from minigpt.evaluate import compute_metrics

    bundle = joblib.load(MODELS / "tfidf.joblib")
    rows = _test_rows()
    preds = bundle["clf"].predict(bundle["vectorizer"].transform([r["text"] for r in rows]))
    y = [bundle["labels"].index(r["label"]) for r in rows]
    expected = json.loads((RESULTS_DIR / "stage1/tfidf_logreg/summary.json").read_text())["test_macro_f1"]["mean"]
    assert compute_metrics(y, preds.tolist(), bundle["labels"])["macro_f1"] == pytest.approx(expected, abs=1e-9)


def test_minigpt_classifier_export_reproduces_its_seed_score() -> None:
    from safetensors.torch import load_file

    from minigpt.evaluate import compute_metrics
    from minigpt.model.classifier import GPTClassifier, encode_for_classification
    from minigpt.model.gpt import GPT, GPTConfig
    from minigpt.tokenizer.bpe import BPETokenizer

    cfg = json.loads((MODELS / "minigpt_classifier.json").read_text())
    model = GPTClassifier(GPT(GPTConfig(**cfg["gpt_config"])), 3, cfg["head_dropout"], cfg["pooling"])
    missing, unexpected = model.load_state_dict(load_file(MODELS / "minigpt_classifier.safetensors"), strict=False)
    assert missing == ["gpt.lm_head.weight"] and not unexpected
    assert model.gpt.lm_head.weight is model.gpt.tok_emb.weight  # still tied after loading
    model.eval()
    tok = BPETokenizer.load(MODELS / "tokenizer.json")
    rows = _test_rows()
    preds = []
    with torch.no_grad():
        for r in rows:
            ids = encode_for_classification(tok, r["text"], cfg["max_len"])
            preds.append(int(model(torch.tensor([ids])).logits.argmax()))
    y = [cfg["labels"].index(r["label"]) for r in rows]
    summary = json.loads((RESULTS_DIR / "stage2/finetune/gpt_pretrained/summary.json").read_text())
    expected = next(s for s in summary["seeds"] if s["seed"] == cfg["seed"])["test"]["macro_f1"]
    assert compute_metrics(y, preds, cfg["labels"])["macro_f1"] == pytest.approx(expected, abs=2e-3)


def test_minigpt_lm_export_has_its_validation_loss() -> None:
    import numpy as np
    from safetensors.torch import load_file

    from minigpt.model.gpt import GPT, GPTConfig

    val_bin = PROJECT_ROOT / "data/processed/pretrain/val.bin"
    if not val_bin.exists():
        pytest.skip("needs the encoded pre-training corpus")
    cfg = json.loads((MODELS / "minigpt_lm.json").read_text())
    model = GPT(GPTConfig(**cfg["model"])).eval()
    model.load_state_dict(load_file(MODELS / "minigpt_lm.safetensors"), strict=False)
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    t = cfg["model"]["block_size"]
    starts = np.arange(0, 64 * t * 50, t * 50)[:64]
    x = torch.from_numpy(np.stack([data[s:s + t] for s in starts]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[s + 1:s + t + 1] for s in starts]).astype(np.int64))
    with torch.no_grad():
        loss = model(x, targets=y).loss.item()
    assert abs(loss - cfg["val_loss"]) < 0.15   # different windows than training-time eval; same ballpark
