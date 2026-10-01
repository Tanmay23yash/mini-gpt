# Experiments

One entry per run: date, config file, key hyperparameters, metrics, notes.

| Date | Stage | Config | Model | Params | Val loss / Acc | Macro F1 | Notes |
|------|-------|--------|-------|--------|----------------|----------|-------|
| 2026-09-30 | 1 | `configs/stage1_tfidf.yaml` | TF-IDF word 1-2 + char 2-5 grams, LogReg (C=0.25 on val) | - | test acc 0.691 (val 0.623) | test **0.696** (val 0.628) | Best Stage 1 model. Char n-grams handle spelling variants. |
| 2026-09-30 | 1 | `configs/stage1_gru.yaml` | BiGRU, word vocab 12.9k, emb 128, hid 128 | 1.85M | test acc 0.651 ± 0.003 (val 0.598) | test 0.654 ± 0.003 (val 0.600) | 3 seeds. Best epoch 4-5, then overfits. |
| 2026-09-30 | 1 | `configs/stage1_lstm.yaml` | BiLSTM, word vocab 12.9k, emb 128, hid 128 | 1.91M | test acc 0.638 ± 0.016 (val 0.595) | test 0.639 ± 0.018 (val 0.596) | 3 seeds. Val loss rises from epoch 3-4. Val OOV 10.5%. |
| 2026-09-30 | 1 | - | Majority class (neutral) | - | test acc 0.367 | test 0.179 | Floor any model must beat. |

### Stage 1 notes

- **Bag-of-n-grams beats the RNNs** (0.696 vs 0.654 test macro F1). With 14k labeled tweets
  and randomly initialized embeddings, the RNNs overfit within ~4 epochs; TF-IDF's char
  n-grams also share signal across spellings (`nahi` / `nhi` / `nai`) that a word vocab
  treats as unrelated or `<unk>` (10.5% of val tokens). Both problems are what pre-training
  on HingCorpus + subword BPE (Stage 2) should address.
- **Neutral is the hardest class** for every model (test F1 0.54-0.63); most errors are
  negative/positive tweets predicted as neutral, rarely negative <-> positive.
- **Test scores are ~5-7 points above val for every model**, so the official test split is
  simply easier than the dev split. Model selection used val only; the gap is a dataset
  property, not leakage. Compare models on the same split, never val-vs-test.
- Reference: best SemEval-2020 SentiMix Hinglish system reported 75.0 F1 (BERT-based).
| 2026-09-30 | 2 | `configs/bpe.yaml` | Byte-level BPE, 8192 vocab (+`<\|endoftext\|>`), trained on 50k corpus lines | - | 3.74 bytes/token on val | - | 31 min to train (pure Python), 0.8 min to encode 2M lines (chunk cache). 54.5M train / 0.55M val tokens. 77% of words are 1 token. SentiMix: 27 tokens/tweet mean, 99.5% <= 64 tokens. |

### Tokenizer notes

- Frequent spelling variants (`nahi` / `nhi` / `nai`, `accha` / `acha` / `achha`) each get their
  own token; BPE does not merge them. Their similarity must come from pre-training (similar
  contexts -> similar embeddings). Rare variants split into shared pieces (`bhaiya|aaa`).
- No `<unk>` at all (byte fallback) vs 10.5% OOV for the Stage 1 word vocab.
| 2026-09-30 | 2 | `configs/gpt_debug.yaml` | GPT 2 layers, d=64, T=64 | 0.63M | val loss 7.42 (ppl 1672) | - | 300 steps, 0.3M tokens, 6 s. Pipeline check only. |
| 2026-09-30 | 2 | `configs/gpt_small.yaml` | GPT 4 layers, d=256, 4 heads, T=128, fp32 | 5.29M | **val loss 5.009 (ppl 149.7)** | - | 12k steps x 8192 tok = 98M tokens (~1.8 epochs), 40.1 min on GTX 1650, ~42.5k tok/s. Init loss 9.07 = ln(8193). |

### Pre-training notes (gpt_small)

- Train and val loss stay together the whole run (5.05 vs 5.01 at the end), so there is
  **no overfitting**: the model is capacity-limited, not data-limited. A larger model (Kaggle GPU)
  should lower the loss further. (Val sits slightly below train because train loss is measured
  with dropout on, on single noisy batches.)
- Most of the gain happens in the first ~10M tokens (9.07 -> 5.7); the last 50M tokens buy ~0.15.
- fp16 AMP was 4.6x slower than fp32 on the GTX 1650 (no tensor cores), so this run is fp32.
- Samples (top-k 50, temperature 0.8): "modi ji ne kaha tha ki bjp ka koi fayda nahi hai, ab kya hai?",
  "yaar mood khrab ho gaya ..". Grammatical Hinglish with natural code-switching and emoji.
| 2026-09-30 | 2 | `configs/finetune_scratch.yaml` (v1: 8 epochs, lr grid 1e-4..1e-3) | mini-GPT classifier, random init | 5.29M | test acc 0.630 | test 0.633 ± 0.002 | Chosen lr at grid edge; best epoch 2-3, then overfits. Superseded. |
| 2026-09-30 | 2 | `configs/finetune_pretrained.yaml` (v1: 8 epochs, lr grid 1e-4..1e-3) | mini-GPT classifier, pre-trained | 5.29M | test acc 0.667 | test 0.670 ± 0.006 | Best epoch 1 for every seed, lr at grid edge. Superseded. |
| 2026-09-30 | 2 | `configs/finetune_scratch.yaml` | mini-GPT classifier, random init, lr 1e-4, 3 epochs | 5.29M | test acc 0.636 (val 0.606) | test 0.640 ± 0.006 | Control: same model and training, no pre-training. |
| 2026-09-30 | 2 | `configs/finetune_pretrained.yaml` | mini-GPT classifier, pre-trained gpt_small, lr 3e-5, 3 epochs | 5.29M | test acc 0.676 (val 0.628) | **test 0.679 ± 0.002** | +3.9 F1 over random init. Still 1.7 below TF-IDF (0.696). ~6 min for sweep + 3 seeds. |

### Fine-tuning notes

- **Pre-training is worth +3.9 macro F1** (0.679 vs 0.640, same architecture, same protocol,
  3 seeds each, std <= 0.006), and it lifts the mini-GPT from BiLSTM level to clearly above the BiGRU.
- **TF-IDF + LogReg still wins (0.696).** The pre-trained model beats it on positive tweets
  (F1 0.752 vs 0.744) but loses on negative (0.699 vs 0.718) and neutral (0.587 vs 0.625).
  With 5M parameters pre-trained on ~100M tokens, the model's language knowledge is limited
  (val perplexity 150); character n-grams on 14k labeled tweets are a strong baseline for
  short, noisy, code-mixed text. Stage 3 tests whether a real pre-trained LLM closes the gap.
- **What didn't work (v1):** 8 epochs with warmup over the first 0.8 epochs. The model overfit
  after 1-2 epochs while the lr was still near its peak, and the lr sweep picked the grid edge.
  Short runs (3 epochs) with the lr decaying over the whole run fixed both: +0.9 F1 pre-trained,
  +0.7 random init.
- The pre-trained model needs a ~3x lower lr than the random-init one (3e-5 vs 1e-4): large
  updates overwrite what pre-training learned. The final sweep includes 1e-5 so 3e-5 is an
  interior optimum (val F1 1e-5: 0.609, 3e-5: 0.631, 1e-4: 0.617).
- Re-running the pre-trained sweep reproduced every test number to 4 decimals (fixed seeds,
  deterministic cuDNN).
| 2026-09-30 | 3 | `configs/llm.yaml` | Qwen3.5-0.8B zero-shot (label scoring, bf16) | 0.75B | test acc 0.511 (val 0.484) | test 0.415 (val 0.396) | Never predicts neutral (0%). Contextual calibration: test 0.430. |
| 2026-09-30 | 3 | `configs/llm.yaml` | Qwen3.5-0.8B 3-shot (k chosen on val from 0/3/6) | 0.75B | test acc 0.565 | **test 0.508 ± 0.026** | 3 draws of the examples. 5-8% neutral predictions. ~9 min per 3k tweets on the GTX 1650. |

### Zero/few-shot notes (Qwen3.5-0.8B)

- Label scoring works as intended: 97-99% of the next-token probability goes to the three
  label words, so the model answers in format; the problem is *which* label.
- The base model almost never says **neutral** (0% zero-shot, 4-8% few-shot, vs 37% of tweets).
  Examples help (+9 F1 with 3 shots) but 6 shots is worse than 3 on val (0.459 vs 0.486).
- **Contextual calibration backfires with examples in the prompt.** Its content-free inputs
  ("N/A", "") get labeled *neutral* (prior 0.57-0.73 with 3-6 shots), which is the correct answer
  for them, so dividing by that prior removes neutral even more. It only helps zero-shot
  (+1.5 F1), where the prior is positive-heavy. The method assumes content-free = uniform,
  which does not hold for a task with a neutral class.
- The choice of the 3 examples moves test macro F1 by up to 5 points (0.484-0.535).
- Even few-shot (0.508) is far below the 5M-param mini-GPT fine-tuned on SentiMix (0.679):
  task-specific labels matter more than 150x more parameters without them.
| 2026-10-01 | 3 | `configs/lora_kaggle.yaml` | Qwen3.5-0.8B + LoRA (r=16, all-linear), fp16 on Kaggle T4, lr 1e-4, 2 epochs | 0.75B (10.8M trained) | test acc 0.710 (val 0.655) | **test 0.714 ± 0.007** | Best model overall. lr sweep (val): 1e-4 0.653, 2e-4 0.646, 5e-4 0.641. ~47 min per run on a T4. |

### LoRA notes (Qwen3.5-0.8B)

- LoRA fixes the base model's neutral problem: neutral predictions go from 5-8% (3-shot) to
  35-42% (true share 37%), neutral F1 0.61-0.65. Negative 0.73, positive 0.76-0.78.
- **First model to beat TF-IDF (0.696)**, by 1.8 macro F1 (seed std 0.007).
- Limitation: the chosen lr (1e-4) is the lowest in the grid and every seed's best epoch was the
  last (2 of 2), so the model may be slightly under-tuned; a follow-up with lr 5e-5 and 3 epochs
  could add a little. The lr differences are small (0.641-0.653 val F1) but monotonic.
- fp16 GradScaler: the default init scale (2^16) overflows on this model; measured locally that
  steps only succeed below ~64, so the scaler starts at 2^10.

### Stage 3 final comparison (`python -m minigpt.compare`, results/stage3/comparison.md)

| Model | Test macro F1 | 95% CI (paired bootstrap, rep. run) |
|---|---|---|
| Qwen3.5-0.8B + LoRA | **0.714 ± 0.007** | [0.699, 0.730] |
| TF-IDF + LogReg | 0.696 | [0.680, 0.712] |
| mini-GPT, pre-trained + fine-tuned | 0.679 ± 0.002 | [0.663, 0.696] |
| Qwen3.5-0.8B 3-shot | 0.508 ± 0.026 | [0.490, 0.522] |

- LoRA vs TF-IDF: +0.018, 95% CI [+0.002, +0.035] (LoRA better in 98.6% of resamples): a real
  but small win. TF-IDF vs mini-GPT: +0.015, CI [-0.001, +0.032]: **not** significant at 95%.
- Per class, LoRA is best on all three (neg 0.737 / neu 0.647 / pos 0.759).
- Representative-run point estimates match the training-time numbers (LoRA seed 44: 0.7165 on
  Kaggle in fp16 vs 0.714 here in bf16 - the small gap is inference precision).
- Generation: the mini-GPT writes short, tweet-like Hinglish; Qwen3.5-0.8B (a chat/thinking
  model) continues raw text with chat artifacts (`<think>`), repetition or drifts into
  translation mode. Domain pre-training gives the small model the right *style*; the LLM
  gives better *understanding* once fine-tuned.
