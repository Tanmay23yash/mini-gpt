---
title: mini-gpt Hinglish
emoji: 🗣️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.29.0
python_version: "3.12"
app_file: app.py
license: cc-by-nc-sa-4.0
short_description: Hinglish GPT from scratch vs LoRA-tuned Qwen3.5 sentiment
---

# mini-gpt: a Hinglish GPT from scratch vs a LoRA-tuned open LLM

- **Classify sentiment** of a Hinglish (Hindi-English code-mixed) tweet with three models side by side:
  TF-IDF + logistic regression, a 5.3M-parameter GPT written from scratch and pre-trained on Hinglish
  tweets, and Qwen3.5-0.8B fine-tuned with LoRA.
- **Generate** Hinglish text with the from-scratch GPT (temperature, top-k, top-p).
- **Inspect attention** heads of the from-scratch GPT.
- **Results** on the SemEval-2020 SentiMix test set (best: Qwen3.5-0.8B + LoRA, macro F1 0.714).

Non-commercial: the from-scratch GPT is trained on L3Cube-HingCorpus (CC BY-NC-SA 4.0).
The Qwen3.5-0.8B base model (Apache-2.0) is downloaded from the Hub at startup; this Space only
contains the LoRA adapter. Runs on the free CPU tier; the LLM answer takes a few seconds.
