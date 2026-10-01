"""mini-gpt demo: Hinglish text generation, sentiment with three models, attention maps.

Run locally (from the project root)::

    python -m minigpt.export_app        # writes app/models/
    python app/app.py                   # http://127.0.0.1:7860

On Hugging Face Spaces this file runs from dist/space (``python -m minigpt.export_app --space``).
Environment variables: ENABLE_QWEN=0 skips the 0.8B LLM (faster start, less RAM);
MINIGPT_MODELS points to another models folder.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import minigpt  # noqa: F401  - first: loads .env (GRADIO_TEMP_DIR etc.) before gradio is imported

import gradio as gr
import torch
from safetensors.torch import load_file

from minigpt.generate import generate_text
from minigpt.model.classifier import GPTClassifier, encode_for_classification
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.tokenizer.bpe import BPETokenizer
from minigpt.visualize import attention_for, plot_attention_grid

HERE = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("MINIGPT_MODELS", HERE / "models"))
ENABLE_QWEN = os.environ.get("ENABLE_QWEN", "1") != "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP_DIR = os.environ.get("GRADIO_TEMP_DIR")  # None -> system temp (fine on Spaces)

INFO = json.loads((MODELS / "model_info.json").read_text(encoding="utf-8"))
LABELS: list[str] = INFO["labels"]
TOK = BPETokenizer.load(MODELS / "tokenizer.json")


# --------------------------------------------------------------------------- models


def _load_weights(module: torch.nn.Module, path: Path) -> None:
    """Load safetensors weights; the only key allowed to be missing is the tied LM head."""
    missing, unexpected = module.load_state_dict(load_file(path), strict=False)
    if unexpected or any(not k.endswith("lm_head.weight") for k in missing):
        raise RuntimeError(f"bad checkpoint {path.name}: missing {missing}, unexpected {unexpected}")


def load_lm() -> GPT:
    cfg = json.loads((MODELS / "minigpt_lm.json").read_text(encoding="utf-8"))
    model = GPT(GPTConfig(**cfg["model"]))
    _load_weights(model, MODELS / "minigpt_lm.safetensors")
    return model.to(DEVICE).eval()


def load_minigpt_classifier() -> tuple[GPTClassifier, dict]:
    cfg = json.loads((MODELS / "minigpt_classifier.json").read_text(encoding="utf-8"))
    model = GPTClassifier(GPT(GPTConfig(**cfg["gpt_config"])), len(LABELS), cfg["head_dropout"], cfg["pooling"])
    _load_weights(model, MODELS / "minigpt_classifier.safetensors")
    return model.to(DEVICE).eval(), cfg


def load_tfidf() -> dict:
    import joblib

    return joblib.load(MODELS / "tfidf.joblib")


LM = load_lm()
CLF, CLF_CFG = load_minigpt_classifier()
TFIDF = load_tfidf()

# Qwen3.5-0.8B + LoRA is loaded before the server starts (downloads ~1.5 GB the first time).
# Loading it in a background thread deadlocked on Windows (imports racing the server start-up).
QWEN: dict = {"model": None, "tok": None, "status": "disabled (ENABLE_QWEN=0)"}


def load_qwen() -> None:
    try:
        from peft import PeftModel

        from minigpt.llm import label_token_ids, load_llm

        t0 = time.perf_counter()
        dtype = "bfloat16" if DEVICE.type == "cuda" else "float32"
        base, tok = load_llm(INFO["qwen"]["model_id"], dtype, DEVICE)
        model = PeftModel.from_pretrained(base, MODELS / "qwen_lora").merge_and_unload().eval()
        QWEN.update(model=model, tok=tok, ids=label_token_ids(tok, LABELS), status="ready")
        print(f"Qwen3.5-0.8B + LoRA ready on {DEVICE} ({dtype}) in {time.perf_counter() - t0:.0f} s")
    except Exception as e:  # keep the rest of the demo usable
        QWEN["status"] = f"failed to load: {type(e).__name__}: {e}"
        print(QWEN["status"])


if ENABLE_QWEN:
    load_qwen()


# --------------------------------------------------------------------------- tab functions


@torch.inference_mode()  # per call: grad mode is thread-local and Gradio uses worker threads
def classify(text: str) -> tuple[dict, dict, dict | None, str]:
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type a tweet first.")
    # TF-IDF
    probs = TFIDF["clf"].predict_proba(TFIDF["vectorizer"].transform([text]))[0]
    tfidf = {lab: float(p) for lab, p in zip(LABELS, probs)}
    # mini-GPT classifier
    ids = encode_for_classification(TOK, text, CLF_CFG["max_len"])
    logits = CLF(torch.tensor([ids], device=DEVICE)).logits[0]
    minigpt = {lab: float(p) for lab, p in zip(LABELS, torch.softmax(logits, -1).tolist())}
    # Qwen3.5 + LoRA
    if QWEN["status"] != "ready":
        return tfidf, minigpt, None, f"Qwen3.5-0.8B + LoRA: {QWEN['status']}"
    from minigpt.llm import build_messages, format_prompt, label_scores

    tok = QWEN["tok"]
    t0 = time.perf_counter()
    prompt = format_prompt(tok, build_messages(text, INFO["qwen"]["system_prompt"]))
    enc = tok([prompt], return_tensors="pt").to(DEVICE)
    scores, mass = label_scores(QWEN["model"](**enc, logits_to_keep=1).logits[:, -1], QWEN["ids"])
    qwen = {lab: float(p) for lab, p in zip(LABELS, torch.softmax(scores[0], -1).tolist())}
    note = f"Qwen3.5-0.8B + LoRA: {time.perf_counter() - t0:.1f} s on {DEVICE.type}"
    return tfidf, minigpt, qwen, note


@torch.inference_mode()
def generate(prompt: str, n: int, max_new_tokens: int, temperature: float, top_k: int, top_p: float,
             seed: int) -> str:
    texts = generate_text(LM, TOK, prompt or "", int(n), int(max_new_tokens), float(temperature),
                          int(top_k) or None, float(top_p) if top_p < 1.0 else None, int(seed))
    return "\n\n".join(f"**{i}.** {t}" for i, t in enumerate(texts, 1))


@torch.inference_mode()
def attention(text: str, layer: str) -> str:
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type a sentence first.")
    labels, weights = attention_for(LM, TOK, text, max_len=24)
    if layer != "all layers":
        idx = int(layer.split()[-1])
        weights = weights[idx: idx + 1]
    path = Path(tempfile.mkdtemp(dir=TEMP_DIR)) / "attention.png"
    plot_attention_grid(labels, weights, path, f'"{text}" ({layer})')
    return str(path)


# --------------------------------------------------------------------------- UI

F1 = INFO["test_macro_f1"]
RESULTS_MD = f"""
All models are evaluated on the same **3,000 SentiMix test tweets** (SemEval-2020 Task 9, Hinglish).
Macro F1 is the mean of the per-class F1 scores, so always predicting one class scores badly.

| Model | Params | Test macro F1 |
|---|---|---|
| Qwen3.5-0.8B + LoRA (10.8M trained params) | 0.75B | **{F1['qwen_lora']['mean']:.3f} ± {F1['qwen_lora']['std']:.3f}** |
| TF-IDF + logistic regression | 0.3M | {F1['tfidf']['mean']:.3f} |
| mini-GPT, pre-trained on 98M Hinglish tokens, fine-tuned | 5.3M | {F1['minigpt']['mean']:.3f} ± {F1['minigpt']['std']:.3f} |
| BiGRU | 1.8M | {F1['bigru']['mean']:.3f} |
| mini-GPT, random init, fine-tuned | 5.3M | {F1['minigpt_scratch']['mean']:.3f} |
| Qwen3.5-0.8B, 3 examples in the prompt | 0.75B | {F1['qwen_3shot']['mean']:.3f} |
| Qwen3.5-0.8B, zero-shot | 0.75B | {F1['qwen_0shot']['mean']:.3f} |

- Pre-training the mini-GPT on unlabeled Hinglish tweets adds about +4 F1 over the same model from random init.
- Without fine-tuning, the 0.8B LLM almost never answers "neutral"; LoRA fixes that and gives the best model.
- LoRA vs TF-IDF: +0.018 F1, 95% bootstrap CI [+0.002, +0.035]: a real but small win.
"""

CLASSIFY_EXAMPLES = [
    "yeh movie bilkul bakwas thi, paisa barbaad ho gaya",
    "bhai aaj ka match ekdum zabardast tha, maza aa gaya 🔥",
    "kal se office 10 baje khulega, sab log time pe aana",
    "itna traffic roz roz, pata nahi kab sudhrega yeh sheher",
    "Happy birthday bhai, bhagwan tujhe hamesha khush rakhe ❤️",
    "sarkar ne naye scheme ka announcement kiya hai, details kal aayengi",
]

with gr.Blocks(title="mini-gpt: Hinglish GPT from scratch") as demo:
    gr.Markdown(
        "# mini-gpt: a Hinglish GPT from scratch vs a LoRA-tuned open LLM\n"
        "A 5.3M-parameter GPT written from scratch in PyTorch and pre-trained on Hinglish (Hindi-English "
        "code-mixed) tweets, compared with classic NLP and with Qwen3.5-0.8B fine-tuned with LoRA.")
    with gr.Tab("Classify sentiment"):
        inp = gr.Textbox(label="Hinglish tweet", lines=2, placeholder="yaar aaj ka din bahut accha tha")
        btn = gr.Button("Classify", variant="primary")
        with gr.Row():
            out_tfidf = gr.Label(label="TF-IDF + LogReg", num_top_classes=3)
            out_mini = gr.Label(label="mini-GPT (5.3M, from scratch)", num_top_classes=3)
            out_qwen = gr.Label(label="Qwen3.5-0.8B + LoRA", num_top_classes=3)
        note = gr.Markdown()
        gr.Examples(CLASSIFY_EXAMPLES, inputs=inp)
        btn.click(classify, inp, [out_tfidf, out_mini, out_qwen, note], api_name="classify")
        inp.submit(classify, inp, [out_tfidf, out_mini, out_qwen, note], api_name=False)
    with gr.Tab("Generate Hinglish"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt (empty = start a new tweet)", value="bhai aaj ka match")
                n = gr.Slider(1, 5, value=3, step=1, label="Samples")
                max_new = gr.Slider(10, 100, value=40, step=5, label="Max new tokens")
                temp = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature (0 = greedy)")
                top_k = gr.Slider(0, 200, value=50, step=5, label="Top-k (0 = off)")
                top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-p (1 = off)")
                seed = gr.Number(value=7, precision=0, label="Seed")
                gen_btn = gr.Button("Generate", variant="primary")
            gen_out = gr.Markdown()
        gen_btn.click(generate, [prompt, n, max_new, temp, top_k, top_p, seed], gen_out)
    with gr.Tab("Attention maps"):
        gr.Markdown("What each attention head of the mini-GPT looks at. Row = token doing the looking, "
                    "column = token being looked at. Layer 2, head 3 is a previous-token head.")
        with gr.Row():
            att_text = gr.Textbox(label="Sentence", value="bhai aaj ka match bahut accha tha", scale=3)
            att_layer = gr.Dropdown(["layer 0", "layer 1", "layer 2", "layer 3", "all layers"],
                                    value="layer 2", label="Layer", scale=1)
        att_btn = gr.Button("Show attention", variant="primary")
        att_img = gr.Image(type="filepath", label="Attention weights")
        att_btn.click(attention, [att_text, att_layer], att_img)
    with gr.Tab("Results"):
        gr.Markdown(RESULTS_MD)
        gr.Image(str(MODELS / "comparison.png"), label="Test macro F1 by model", show_label=False)
    gr.Markdown(
        "<small>The mini-GPT is pre-trained on L3Cube-HingCorpus (CC BY-NC-SA 4.0), so this demo is "
        "non-commercial. Sentiment labels: SemEval-2020 SentiMix. Qwen3.5-0.8B: Apache-2.0. "
        "Predictions can be wrong and reflect biases in Twitter data.</small>")

if __name__ == "__main__":
    demo.launch()
