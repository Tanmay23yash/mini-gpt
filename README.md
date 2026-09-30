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
