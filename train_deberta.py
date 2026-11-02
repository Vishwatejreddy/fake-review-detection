"""Train a DeBERTa-v3-small classifier for fake-review detection.

Example:
    python src/train_deberta.py --data data/fake_reviews_dataset.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "microsoft/deberta-v3-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DeBERTa fake-review classifier.")
    parser.add_argument("--data", required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--text-column", default="text_", help="CSV column containing review text.")
    parser.add_argument("--label-column", default="label", help="CSV column containing labels.")
    parser.add_argument("--output-dir", default="outputs/deberta", help="Directory for model outputs.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def encode_labels(labels: pd.Series) -> pd.Series:
    """Map source labels to integers: fake=0 and genuine/real=1."""
    mapping = {
        "CG": 0,
        "OR": 1,
        "fake": 0,
        "real": 1,
        "genuine": 1,
    }
    normalized = labels.astype(str).str.strip().str.lower()
    normalized_mapping = {key.lower(): value for key, value in mapping.items()}
    encoded = normalized.map(normalized_mapping)

    unknown = sorted(normalized[encoded.isna()].unique().tolist())
    if unknown:
        raise ValueError(
            "Unrecognized label value(s): "
            + ", ".join(unknown)
            + ". Update encode_labels() to match your dataset."
        )
    return encoded.astype(int)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_csv(data_path)
    for column in (args.text_column, args.label_column):
        if column not in frame.columns:
            raise ValueError(
                f"Column '{column}' was not found. Available columns: {frame.columns.tolist()}"
            )

    frame = frame[[args.text_column, args.label_column]].copy()
    frame = frame.rename(columns={args.text_column: "text", args.label_column: "labels"})
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["labels"] = encode_labels(frame["labels"])
    frame = frame[frame["text"].str.len() > 0].reset_index(drop=True)

    if frame["labels"].nunique() != 2:
        raise ValueError("The dataset must contain both classes after label mapping.")

    train_frame, test_frame = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=42,
        stratify=frame["labels"],
    )
    train_dataset = Dataset.from_pandas(train_frame.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_frame.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    def compute_metrics(eval_prediction) -> dict:
        logits, true_labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="binary", zero_division=0
        )
        return {
            "accuracy": accuracy_score(true_labels, predictions),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.4f}")

    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=-1)
    print("\nClassification report (0=fake, 1=genuine):")
    print(
        classification_report(
            predictions.label_ids,
            predicted_labels,
            target_names=["fake", "genuine"],
            digits=4,
            zero_division=0,
        )
    )

    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    print(f"\nSaved model and tokenizer to: {output_dir / 'final_model'}")


if __name__ == "__main__":
    main()
