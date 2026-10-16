# LLM fine-tuning (CS197 Lecture 4)

Replicate the lecture fine-tuning exercise: fine-tune a small pretrained causal LM (e.g. DistilGPT-2) on a standard dataset such as SQuAD.

## Goal

- Start from a **pretrained** checkpoint (not random weights)
- Fine-tune with Hugging Face `Trainer`
- Evaluate (e.g. loss / perplexity) and try generation

## Setup

```bash
conda create -n lec4 python=3.13
conda activate lec4
pip install torch transformers datasets accelerate jupyter
```

## Layout

```
notebooks/   # fine-tuning notebook(s)
data/        # local data if needed (gitignored)
models/      # checkpoints (gitignored)
```

## Note

The from-scratch group-chat LM exercise lives on the `adam` branch.
