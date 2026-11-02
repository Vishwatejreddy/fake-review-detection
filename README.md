# Fake Review Detection

A text-classification project for identifying fake and genuine product reviews. The project report's extracted code explores traditional machine-learning models, deep-learning models, and a DeBERTa transformer.

## Models represented in the source

- TF-IDF + Logistic Regression
- TF-IDF + Random Forest
- TF-IDF + Multinomial Naive Bayes
- CNN
- BiLSTM
- DeBERTa-v3-small (transformer)

The cleaned starter script in `src/train_deberta.py` focuses on the DeBERTa model, which is the transformer approach highlighted in the project. The original extracted code contained repeated notebook cells, machine-specific Windows paths, and inconsistent column-name examples; those have been consolidated into a configurable script.

## Dataset

The dataset is **not included** in this repository. Place your CSV file in a local `data/` folder (which is ignored by Git), or provide its path when running the script.

The script expects:
- a review-text column (by default `text_`)
- a label column (by default `label`)

The source code maps `CG` to `0` and `OR` to `1`. It also accepts common text labels such as `fake`/`real` and `genuine`. Check that this mapping matches your dataset before training.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python src/train_deberta.py --data data/fake_reviews_dataset.csv
```

If your CSV uses different column names:

```bash
python src/train_deberta.py --data path/to/reviews.csv --text-column review_text --label-column category
```

The script creates model outputs under `outputs/` and prints evaluation accuracy. The first run downloads the pretrained `microsoft/deberta-v3-small` model and tokenizer, so it requires an internet connection and sufficient memory.

## Important notes

- No dataset or trained model weights are committed.
- The original report's claimed accuracy should not be treated as independently verified by this cleaned script. Run the training and record the actual evaluation results before publishing metrics.
- The extracted report code contains multiple experimental sections. This repository currently provides a consolidated DeBERTa training entry point rather than claiming every experiment is a polished, independently reproducible pipeline.
