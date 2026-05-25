"""
Evaluate and compare three models:
  1. VADER (baseline)
  2. Pretrained RoBERTa (our current model)
  3. Fine-tuned DistilBERT (our trained model)

Usage:
    poetry run python -m sentiment_analysis.training.evaluate
"""

import json
import mlflow
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentiment_analysis.training.comparison import SentimentComparator

FINETUNED_MODEL_PATH = "models/finetuned-sentiment"

# Hand-labeled ground truth test set
# These are examples where we KNOW the correct answer
GROUND_TRUTH = [
    # (text, true_label)
    ("I love this, it works perfectly", "positive"),
    ("Best decision I ever made", "positive"),
    ("This is absolutely incredible", "positive"),
    ("This track goes so hard", "positive"),
    ("That presentation was fire", "positive"),
    ("The new update actually slaps", "positive"),
    ("I am so happy with how this turned out", "positive"),
    ("Not bad at all, actually really enjoyed it", "positive"),

    ("This is a complete disaster", "negative"),
    ("Absolutely terrible, would not recommend", "negative"),
    ("Wow, what a surprise. The train is late again.", "negative"),
    ("Oh great, another Monday morning meeting.", "negative"),
    ("I can't believe how badly this was handled", "negative"),
    ("Total waste of time and money", "negative"),
    ("Nothing works and support is useless", "negative"),

    ("The meeting is on Thursday at 2pm", "neutral"),
    ("The package will arrive in 3-5 business days", "neutral"),
    ("The update changed the default settings", "neutral"),
    ("There are 12 items in the list", "neutral"),
    ("The report was submitted yesterday", "neutral"),
]


def evaluate_vader(comparator, texts, true_labels):
    preds = []
    for text in texts:
        result = comparator.compare(text)
        preds.append(result.vader_sentiment)
    return preds


def evaluate_roberta(comparator, texts, true_labels):
    preds = []
    for text in texts:
        result = comparator.compare(text)
        preds.append(result.roberta_sentiment)
    return preds


def evaluate_finetuned(texts):
    pipe = pipeline(
        "text-classification",
        model=FINETUNED_MODEL_PATH,
        truncation=True,
        max_length=128,
    )
    results = pipe(texts)
    return [r["label"].lower() for r in results]


def print_model_report(name: str, true_labels: list, pred_labels: list):
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.2%}  |  F1 Macro: {f1:.2%}")
    print(classification_report(true_labels, pred_labels, zero_division=0))


def main():
    texts = [t for t, _ in GROUND_TRUTH]
    true_labels = [l for _, l in GROUND_TRUTH]

    print("Loading models...")
    comparator = SentimentComparator()

    mlflow.set_experiment("reddit-sentiment-evaluation")

    with mlflow.start_run(run_name="model-comparison"):

        # ── VADER ─────────────────────────────────────────────────────────────
        print("Evaluating VADER...")
        vader_preds = evaluate_vader(comparator, texts, true_labels)
        print_model_report("VADER (baseline)", true_labels, vader_preds)
        mlflow.log_metrics({
            "vader_accuracy": accuracy_score(true_labels, vader_preds),
            "vader_f1_macro": f1_score(true_labels, vader_preds, average="macro", zero_division=0),
        })

        # ── Pretrained RoBERTa ────────────────────────────────────────────────
        print("Evaluating pretrained RoBERTa...")
        roberta_preds = evaluate_roberta(comparator, texts, true_labels)
        print_model_report("Pretrained RoBERTa", true_labels, roberta_preds)
        mlflow.log_metrics({
            "roberta_accuracy": accuracy_score(true_labels, roberta_preds),
            "roberta_f1_macro": f1_score(true_labels, roberta_preds, average="macro", zero_division=0),
        })

        # ── Fine-tuned DistilBERT ─────────────────────────────────────────────
        try:
            print("Evaluating fine-tuned DistilBERT...")
            finetuned_preds = evaluate_finetuned(texts)
            print_model_report("Fine-tuned DistilBERT", true_labels, finetuned_preds)
            mlflow.log_metrics({
                "finetuned_accuracy": accuracy_score(true_labels, finetuned_preds),
                "finetuned_f1_macro": f1_score(true_labels, finetuned_preds, average="macro", zero_division=0),
            })
        except Exception:
            print("\nFine-tuned model not found — run train.py first.")

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"VADER accuracy:    {accuracy_score(true_labels, vader_preds):.2%}")
        print(f"RoBERTa accuracy:  {accuracy_score(true_labels, roberta_preds):.2%}")


if __name__ == "__main__":
    main()