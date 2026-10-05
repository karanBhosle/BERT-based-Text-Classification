# BERT-based Text Classification

**Author:** Karan Bhosle  
**Contact:** [LinkedIn — Karan Bhosle](https://www.linkedin.com/in/karanbhosle/)

## Overview

This repository is a **hands-on learning project** for fine-tuning a pre-trained **BERT** model (`bert-base-uncased`) on a small custom dataset for **binary text classification** (labels `0` and `1`).

The main artifact is the Jupyter notebook:

`BERT_TEXT_CLASSIFICATION/BERT_TEXT_CLASSIFICATION.ipynb`

Each section includes **markdown notes** (what to learn and why) plus **commented code** for that step—install → data → tokenization → training → evaluation.

## What you will learn

| Topic | Takeaway |
|--------|----------|
| Tokenization | Text → `input_ids` + `attention_mask` for BERT |
| `Dataset` / `DataLoader` | Batching and shuffling for PyTorch training |
| Fine-tuning | Forward pass, cross-entropy loss, AdamW updates |
| Inference | `model.eval()`, `torch.no_grad()`, `argmax` on logits |
| Metrics | Accuracy with scikit-learn (and when to use other metrics) |

## Quick start

### Requirements

- Python 3.9+
- GPU optional (CUDA speeds up training; CPU works for this tiny demo)

### Run the notebook

```bash
pip install transformers torch scikit-learn jupyter
jupyter notebook BERT_TEXT_CLASSIFICATION/BERT_TEXT_CLASSIFICATION.ipynb
```

On **Google Colab**, open the notebook from this repo or upload it; the first cell installs dependencies.

### Hugging Face (optional)

Public models download without a token. For gated models or higher rate limits, set `HF_TOKEN` in your environment ([Hugging Face settings](https://huggingface.co/settings/tokens)).

## Project flow (matches the notebook)

1. **Dependencies** — `transformers`, `torch`, `scikit-learn`
2. **Data** — example `texts` and binary `labels`
3. **Hyperparameters** — `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE` (typical BERT LR ≈ `2e-5`)
4. **Device** — CPU or CUDA
5. **Model** — `BertTokenizer` + `BertForSequenceClassification(num_labels=2)`
6. **Custom dataset** — tokenize in `__getitem__` with `max_length=128`
7. **Training** — standard loop with average loss per epoch
8. **Evaluation** — batch inference and **accuracy** on a small test list

## Important notes

- The training set has only **10** examples. Results illustrate the **workflow**, not state-of-the-art accuracy.
- For real tasks: use a proper train/validation/test split, more data, class-imbalance metrics (F1, etc.), and save checkpoints (`model.save_pretrained`).
- The notebook training loop was cleaned up from an earlier version (removed duplicate nested loops, fixed device placement for `input_ids`, and prints **accuracy** explicitly).

## Libraries

- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [scikit-learn](https://scikit-learn.org/) (`accuracy_score`)

## References

1. Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. [Hugging Face — Fine-tuning a pretrained model](https://huggingface.co/docs/transformers/training)
3. [PyTorch — Custom Dataset & DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

## License

Use and adapt this project for learning. If you extend it, consider adding a `requirements.txt` and your own dataset instructions.
