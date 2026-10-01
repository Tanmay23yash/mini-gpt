"""Package the trained models for the Gradio demo and build a Hugging Face Space bundle.

Usage::

    python -m minigpt.export_app            # -> app/models/ (for running app/app.py locally)
    python -m minigpt.export_app --space    # also -> dist/space/ (upload this folder as the Space)

What gets exported (weights only, no optimizer state):
    minigpt_lm.safetensors + minigpt_lm.json     pre-trained mini-GPT (text generation, attention maps)
    minigpt_classifier.safetensors + .json       fine-tuned mini-GPT sentiment classifier (median seed)
    tokenizer.json                               the BPE tokenizer
    tfidf.joblib                                 TF-IDF + logistic regression (C chosen on val)
    qwen_lora/                                   LoRA adapter for Qwen3.5-0.8B (median seed)
    comparison.png, comparison.md, model_info.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from minigpt.env import CHECKPOINT_DIR, CONFIG_DIR, PROJECT_ROOT, RESULTS_DIR
from minigpt.utils import load_config

APP_DIR = PROJECT_ROOT / "app"
MODELS_DIR = APP_DIR / "models"
SPACE_DIR = PROJECT_ROOT / "dist" / "space"


def _contiguous(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # safetensors refuses shared memory; the tied lm_head / tok_emb weight is rebuilt by GPT.__init__
    return {k: v.contiguous().clone() for k, v in state.items() if not k.endswith("lm_head.weight")}


def export_models(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    labels = load_config(CONFIG_DIR / "data.yaml")["sentimix"]["labels"]
    info: dict[str, Any] = {"labels": labels}

    # 1. pre-trained mini-GPT language model
    ckpt = torch.load(CHECKPOINT_DIR / "stage2/gpt_small/best.pt", map_location="cpu", weights_only=False)
    save_file(_contiguous(ckpt["model"]), out / "minigpt_lm.safetensors")
    (out / "minigpt_lm.json").write_text(json.dumps({"model": ckpt["config"]["model"], "step": ckpt["step"],
                                                     "val_loss": ckpt["best_val_loss"]}, indent=2), encoding="utf-8")
    info["minigpt_lm"] = {"val_loss": ckpt["best_val_loss"], "tokens_seen": (ckpt["step"] + 1) * 64 * 128}

    # 2. fine-tuned mini-GPT classifier (median-seed run, the one used in the comparison)
    meta = json.loads((RESULTS_DIR / "stage3/predictions_meta.json").read_text(encoding="utf-8"))
    seed = meta["minigpt_seed"]
    ckpt = torch.load(CHECKPOINT_DIR / f"stage2/finetune/gpt_pretrained_seed{seed}.pt", map_location="cpu",
                      weights_only=False)
    save_file(_contiguous(ckpt["model"]), out / "minigpt_classifier.safetensors")
    (out / "minigpt_classifier.json").write_text(json.dumps({
        "gpt_config": ckpt["gpt_config"], "labels": ckpt["labels"], "max_len": ckpt["config"]["max_len"],
        "pooling": ckpt["config"]["pooling"], "head_dropout": ckpt["config"]["head_dropout"], "seed": seed},
        indent=2), encoding="utf-8")

    # 3. tokenizer
    shutil.copy(PROJECT_ROOT / "artifacts/tokenizer/bpe_8192.json", out / "tokenizer.json")

    # 4. TF-IDF + logistic regression, refit with the C chosen on val
    import joblib
    from sklearn.linear_model import LogisticRegression

    from minigpt.train_baseline import load_splits, tfidf_features

    t_cfg = load_config(CONFIG_DIR / "stage1_tfidf.yaml")
    t_cfg["labels"] = labels
    c = json.loads((RESULTS_DIR / "stage1/tfidf_logreg/summary.json").read_text(encoding="utf-8"))["seeds"][0]["chosen_C"]
    data = load_splits(labels)
    x, vectorizer = tfidf_features(t_cfg, data, return_vectorizer=True)
    clf = LogisticRegression(C=c, max_iter=t_cfg["logreg"]["max_iter"], random_state=t_cfg["seeds"][0])
    clf.fit(x["train"], data["train"]["y"])
    joblib.dump({"vectorizer": vectorizer, "clf": clf, "labels": labels}, out / "tfidf.joblib", compress=3)

    # 5. Qwen3.5 LoRA adapter (median seed)
    lora = json.loads((RESULTS_DIR / "stage3/lora_kaggle/summary.json").read_text(encoding="utf-8"))
    lora_seed = meta["qwen_lora_seed"]
    src = CHECKPOINT_DIR / f"stage3/lora_kaggle_lr{lora['chosen_lr']:g}_seed{lora_seed}"
    shutil.copytree(src, out / "qwen_lora", dirs_exist_ok=True)
    llm_cfg = load_config(CONFIG_DIR / "llm.yaml")
    info["qwen"] = {"model_id": lora["model_id"], "system_prompt": llm_cfg["prompting"]["system_prompt"],
                    "lora_seed": lora_seed, "lora_lr": lora["chosen_lr"]}

    # 6. results for the Results tab
    comparison = json.loads((RESULTS_DIR / "stage3/comparison.json").read_text(encoding="utf-8"))
    info["test_macro_f1"] = {r["key"]: r["f1"] for r in comparison["table"]}
    shutil.copy(RESULTS_DIR / "stage3/comparison.png", out / "comparison.png")
    (out / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.stat().st_size / 1e6:7.1f} MB  {f.relative_to(out)}")
    return info


def build_space() -> None:
    """dist/space = app.py, requirements.txt, README.md (Space card), the minigpt package, models/."""
    if SPACE_DIR.exists():
        shutil.rmtree(SPACE_DIR)
    SPACE_DIR.mkdir(parents=True)
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy(APP_DIR / name, SPACE_DIR / name)
    shutil.copytree(PROJECT_ROOT / "src/minigpt", SPACE_DIR / "minigpt",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(MODELS_DIR, SPACE_DIR / "models")
    (SPACE_DIR / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
                                              "*.joblib filter=lfs diff=lfs merge=lfs -text\n"
                                              "*.png filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    total = sum(f.stat().st_size for f in SPACE_DIR.rglob("*") if f.is_file())
    print(f"Space bundle: {SPACE_DIR} ({total / 1e6:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models for the Gradio app / HF Space")
    parser.add_argument("--space", action="store_true", help="also build dist/space for upload")
    args = parser.parse_args()
    print(f"exporting models to {MODELS_DIR}")
    export_models(MODELS_DIR)
    if args.space:
        build_space()


if __name__ == "__main__":
    main()
