"""
Fine-tune DistilBERT on scraped Reddit comments.
Tracks experiments with both W&B and MLflow.

Usage:
    poetry run python -m sentiment_analysis.training.train
"""

import json
import mlflow
import wandb
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

TRAINING_DATA_PATH = "data/training_data.jsonl"
MODEL_OUTPUT_PATH = "models/finetuned-sentiment"
BASE_MODEL = "distilbert-base-uncased"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def load_data(path: str) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            if sample["label"] in LABEL2ID:
                texts.append(sample["text"])
                labels.append(LABEL2ID[sample["label"]])
    return texts, labels


def tokenize(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading training data...")
    texts, labels = load_data(TRAINING_DATA_PATH)
    print(f"Loaded {len(texts)} samples")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

    # ── Tokenize ──────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_dataset = train_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── W&B + MLflow ──────────────────────────────────────────────────────────
    wandb.init(
        project="reddit-sentiment",
        name="distilbert-finetuned",
        config={
            "base_model": BASE_MODEL,
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
            "max_length": 128,
        },
    )

    mlflow.set_experiment("reddit-sentiment")
    mlflow.start_run(run_name="distilbert-finetuned")
    mlflow.log_params({
        "base_model": BASE_MODEL,
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
    })

    # ── Training ──────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_PATH,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to=["wandb"],
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Evaluating...")
    results = trainer.evaluate()
    print(f"\nFinal results: {results}")

    # Log to MLflow
    mlflow.log_metrics({
        "val_accuracy": results["eval_accuracy"],
        "val_f1_macro": results["eval_f1_macro"],
    })

    # Full classification report
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    report = classification_report(
        val_labels,
        pred_labels,
        target_names=["negative", "neutral", "positive"]
    )
    print(f"\nClassification Report:\n{report}")
    mlflow.log_text(report, "classification_report.txt")

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.save_model(MODEL_OUTPUT_PATH)
    tokenizer.save_pretrained(MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")

    mlflow.end_run()
    wandb.finish()


if __name__ == "__main__":
    main()