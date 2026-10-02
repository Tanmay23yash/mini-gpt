"""Build the static project page (free Hugging Face Static Space) from the saved results.

Usage::

    python -m minigpt.export_app        # once: the app models the predictions come from
    python -m minigpt.build_site        # -> dist/site/ (index.html, assets/, README.md)

No model runs in the page: predictions for the example tweets and the sample generations are
computed here with the exact demo-app models and embedded as JSON.
"""

from __future__ import annotations

import csv
import html
import importlib.util
import json
import shutil
from pathlib import Path
from string import Template
from typing import Any

from minigpt.env import CONFIG_DIR, PROJECT_ROOT, RESULTS_DIR
from minigpt.utils import load_config

SITE = PROJECT_ROOT / "dist" / "site"
ASSETS = {
    "comparison.png": RESULTS_DIR / "stage3/comparison.png",
    "loss_curve.png": RESULTS_DIR / "stage2/gpt_small/loss_curve.png",
    "attention.png": RESULTS_DIR / "stage2/attention/attention_1.png",
    "head_stats.png": RESULTS_DIR / "stage2/attention/head_stats.png",
}
# Written for this page (not from the dataset).
OWN_EXAMPLES = [
    "yeh movie bilkul bakwas thi, paisa barbaad ho gaya",
    "bhai aaj ka match ekdum zabardast tha, maza aa gaya 🔥",
    "kal se office 10 baje khulega, sab log time pe aana",
    "itna traffic roz roz, pata nahi kab sudhrega yeh sheher",
    "Happy birthday bhai, bhagwan tujhe hamesha khush rakhe ❤️",
    "sarkar ne naye scheme ka announcement kiya hai, details kal aayengi",
]


def _load_app() -> Any:
    spec = importlib.util.spec_from_file_location("demo_app", PROJECT_ROOT / "app/app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def collect(app: Any) -> dict[str, Any]:
    """Predictions for the example tweets and mini-GPT generations, using the demo models."""
    from minigpt.compare import pick_examples

    labels = load_config(CONFIG_DIR / "data.yaml")["sentimix"]["labels"]
    with (RESULTS_DIR / "stage3/predictions_test.csv").open(encoding="utf-8", newline="") as f:
        test_examples = pick_examples(list(csv.DictReader(f)), labels)
    examples = [{"text": t, "gold": None, "source": "written for this page"} for t in OWN_EXAMPLES]
    examples += [{"text": r["text"], "gold": r["gold"], "source": "SentiMix test set"} for r in test_examples]
    for ex in examples:
        tfidf, mini, qwen, _ = app.classify(ex["text"])
        ex["pred"] = {"tfidf": tfidf, "minigpt": mini, "qwen_lora": qwen}

    gcfg = load_config(CONFIG_DIR / "generate.yaml")
    generations = [{"prompt": p, "samples": [s.split("** ", 1)[1] for s in
                                             app.generate(p, 3, 30, 0.7, 50, 1.0, 7 + i).split("\n\n")]}
                   for i, p in enumerate(gcfg["prompts"][1:])]
    return {"labels": labels, "examples": examples, "generations": generations}


def table_rows() -> str:
    comparison = json.loads((RESULTS_DIR / "stage3/comparison.json").read_text(encoding="utf-8"))
    ci = comparison["bootstrap_ci"]
    rows = []
    for r in comparison["table"]:
        std = f" ± {r['f1']['std']:.3f}" if r["seeds"] > 1 else ""
        interval = f"[{ci[r['key']][0]:.3f}, {ci[r['key']][1]:.3f}]" if r["key"] in ci else ""
        best = ' class="best"' if r["key"] == "qwen_lora" else ""
        rows.append(f"<tr{best}><td>{html.escape(r['model'])}</td><td>{r['stage']}</td>"
                    f"<td class=num>{r['f1']['mean']:.3f}{std}</td><td class=num>{interval}</td>"
                    f"<td class=num>{r['acc']:.3f}</td></tr>")
    return "\n".join(rows)


def build() -> Path:
    app = _load_app()
    data = collect(app)
    comparison = json.loads((RESULTS_DIR / "stage3/comparison.json").read_text(encoding="utf-8"))
    f1 = {r["key"]: r["f1"]["mean"] for r in comparison["table"]}
    lora_vs_tfidf = comparison["differences"]["qwen_lora - tfidf"]

    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "assets").mkdir(parents=True)
    for name, src in ASSETS.items():
        shutil.copy(src, SITE / "assets" / name)

    page = Template((Path(__file__).parent / "site_template.html").read_text(encoding="utf-8")).substitute(
        best_f1=f"{f1['qwen_lora']:.3f}", minigpt_f1=f"{f1['minigpt']:.3f}", tfidf_f1=f"{f1['tfidf']:.3f}",
        pretrain_gain=f"+{(f1['minigpt'] - f1['minigpt_scratch']) * 100:.1f}",
        zero_shot_f1=f"{f1['qwen_0shot']:.3f}", few_shot_f1=f"{f1['qwen_3shot']:.3f}",
        diff=f"+{lora_vs_tfidf['diff']:.3f}", diff_lo=f"+{lora_vs_tfidf['ci'][0]:.3f}",
        diff_hi=f"+{lora_vs_tfidf['ci'][1]:.3f}", table_rows=table_rows(),
        data_json=json.dumps(data, ensure_ascii=False).replace("</", "<\\/"),
    )
    (SITE / "index.html").write_text(page, encoding="utf-8")
    (SITE / "README.md").write_text(
        "---\ntitle: mini-gpt Hinglish\nemoji: 🗣️\ncolorFrom: blue\ncolorTo: green\nsdk: static\n"
        "license: cc-by-nc-sa-4.0\nshort_description: Hinglish GPT from scratch vs LoRA-tuned Qwen3.5 sentiment\n---\n\n"
        "Project page for mini-gpt: a GPT written from scratch, pre-trained on Hinglish tweets, compared with "
        "classic NLP and a LoRA-tuned Qwen3.5-0.8B on SemEval-2020 SentiMix sentiment. Static page, no live model.\n",
        encoding="utf-8")
    total = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file())
    print(f"site: {SITE} ({total / 1e6:.1f} MB, {len(data['examples'])} example tweets)")
    return SITE


if __name__ == "__main__":
    build()
