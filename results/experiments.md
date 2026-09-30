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
