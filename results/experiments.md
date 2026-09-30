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
