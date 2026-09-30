# mini-gpt

A transformer language model written from scratch in PyTorch, trained on Hinglish
(Hindi-English code-mixed) text, then compared against a LoRA-tuned open LLM on
sentiment classification.

Work in progress. See `results/experiments.md` for measured results.

## Setup

```powershell
E:\pyenv\Scripts\activate
copy .env.example .env   # edit cache paths if needed
pip install -r requirements.txt
pip install -e .
pytest
```

## Data

| Dataset | Use | Size | License |
|---|---|---|---|
| [SemEval-2020 Task 9 SentiMix (Hinglish)](https://arxiv.org/abs/2008.04277), via the [RTT1/SentiMix](https://huggingface.co/datasets/RTT1/SentiMix) mirror | Sentiment labels (negative / neutral / positive), official 14k / 3k / 3k split | 20k tweets | Research use (SemEval task data) |
| [L3Cube-HingCorpus](https://github.com/l3cube-pune/code-mixed-nlp) (roman) | Unlabeled pre-training text | First 2M cleaned lines of 52.9M (~39M words) | CC BY-NC-SA 4.0 |

Because HingCorpus is **non-commercial, share-alike**, the pre-trained mini-GPT weights
and the demo are released under CC BY-NC-SA 4.0 as well.

```powershell
python -m minigpt.data --prepare   # download + clean + split (~3 min)
python -m minigpt.data --report    # sizes, class balance, lengths, code-mixing index
```

Cleaning: SentiMix tweets are rebuilt from CoNLL tokens, URLs / @mentions / `RT` are
removed (HingCorpus already has them stripped), mojibake is fixed with `ftfy`.
Corpus lines are deduplicated, and lines matching a SentiMix val/test tweet are
dropped so pre-training never sees evaluation text.
