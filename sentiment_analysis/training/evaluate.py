"""
Two-tier evaluation:
  1. tweet_eval benchmark — standard NLP benchmark, comparable to published results
  2. Hard examples diagnostic set — 20 curated cases exposing specific failure modes

Models compared:
  - VADER (rule-based baseline, 2014)
  - Pretrained RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)
  - Fine-tuned DistilBERT (if available)

Usage:
    poetry run python -m sentiment_analysis.training.evaluate
"""

import os
import mlflow
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentiment_analysis.training.comparison import SentimentComparator

FINETUNED_MODEL_PATH = "models/finetuned-sentiment"
TWEET_EVAL_SAMPLE_SIZE = 12284  # use all test examples from tweet_eval

# tweet_eval labels: 0=negative, 1=neutral, 2=positive
TWEETEVAL_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# ── Hard examples diagnostic set ──────────────────────────────────────────────
GROUND_TRUTH = [
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


def load_tweet_eval(sample_size: int):
    """Load full tweet_eval sentiment test split."""
    print("Loading tweet_eval full test split...")
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")
    texts = [item["text"] for item in dataset]
    labels = [TWEETEVAL_ID2LABEL[item["label"]] for item in dataset]
    print(f"  Loaded {len(texts)} examples")
    return texts, labels


def evaluate_vader(comparator, texts):
    return [comparator.compare(t).vader_sentiment for t in texts]


def evaluate_roberta(comparator, texts):
    return [comparator.compare(t).roberta_sentiment for t in texts]


def evaluate_finetuned(texts):
    from transformers import pipeline
    pipe = pipeline(
        "text-classification",
        model=FINETUNED_MODEL_PATH,
        truncation=True,
        max_length=128,
    )
    results = pipe(texts)
    return [r["label"].lower() for r in results]


def print_report(name: str, true_labels: list, pred_labels: list):
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"  Accuracy: {acc:.2%}   F1 Macro: {f1:.2%}")
    print(f"{'='*55}")
    print(classification_report(true_labels, pred_labels, zero_division=0))
    return acc, f1


def run_tier(name, true_labels, vader_preds, roberta_preds, finetuned_preds=None):
    print(f"\n{'#'*55}")
    print(f"  TIER: {name}")
    print(f"{'#'*55}")

    v_acc, v_f1     = print_report("VADER (baseline)", true_labels, vader_preds)
    r_acc, r_f1     = print_report("Pretrained RoBERTa", true_labels, roberta_preds)

    ft_acc, ft_f1 = None, None
    if finetuned_preds:
        ft_acc, ft_f1 = print_report("Fine-tuned DistilBERT", true_labels, finetuned_preds)

    print(f"\n  SUMMARY — {name}")
    print(f"  {'Model':<28} {'Accuracy':>10} {'F1 Macro':>10}")
    print(f"  {'-'*50}")
    print(f"  {'VADER':<28} {v_acc:>9.1%} {v_f1:>9.1%}")
    print(f"  {'Pretrained RoBERTa':<28} {r_acc:>9.1%} {r_f1:>9.1%}")
    if ft_acc:
        print(f"  {'Fine-tuned DistilBERT':<28} {ft_acc:>9.1%} {ft_f1:>9.1%}")

    return {
        "vader_accuracy": v_acc, "vader_f1": v_f1,
        "roberta_accuracy": r_acc, "roberta_f1": r_f1,
    }


def main():
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    print("Loading models...")
    comparator = SentimentComparator()

    mlflow.set_experiment("reddit-sentiment-evaluation")

    with mlflow.start_run(run_name="two-tier-evaluation"):

        # ── Tier 1: tweet_eval benchmark ──────────────────────────────────────
        tweet_texts, tweet_labels = load_tweet_eval(TWEET_EVAL_SAMPLE_SIZE)

        print("Running VADER on tweet_eval...")
        tweet_vader = evaluate_vader(comparator, tweet_texts)

        print("Running RoBERTa on tweet_eval...")
        tweet_roberta = evaluate_roberta(comparator, tweet_texts)

        tweet_finetuned = None
        try:
            print("Running fine-tuned DistilBERT on tweet_eval...")
            tweet_finetuned = evaluate_finetuned(tweet_texts)
        except Exception:
            print("  Fine-tuned model not found — skipping.")

        tweet_metrics = run_tier(
            "tweet_eval benchmark (n=12284)",
            tweet_labels, tweet_vader, tweet_roberta, tweet_finetuned
        )

        mlflow.log_metrics({
            "tweet_eval_vader_accuracy":   tweet_metrics["vader_accuracy"],
            "tweet_eval_vader_f1":         tweet_metrics["vader_f1"],
            "tweet_eval_roberta_accuracy": tweet_metrics["roberta_accuracy"],
            "tweet_eval_roberta_f1":       tweet_metrics["roberta_f1"],
        })

        # ── Tier 2: hard examples diagnostic ─────────────────────────────────
        hard_texts  = [t for t, _ in GROUND_TRUTH]
        hard_labels = [l for _, l in GROUND_TRUTH]

        hard_vader   = evaluate_vader(comparator, hard_texts)
        hard_roberta = evaluate_roberta(comparator, hard_texts)

        hard_finetuned = None
        try:
            hard_finetuned = evaluate_finetuned(hard_texts)
        except Exception:
            pass

        hard_metrics = run_tier(
            "Hard examples — slang, sarcasm, negation (n=20)",
            hard_labels, hard_vader, hard_roberta, hard_finetuned
        )

        mlflow.log_metrics({
            "hard_vader_accuracy":   hard_metrics["vader_accuracy"],
            "hard_vader_f1":         hard_metrics["vader_f1"],
            "hard_roberta_accuracy": hard_metrics["roberta_accuracy"],
            "hard_roberta_f1":       hard_metrics["roberta_f1"],
        })

        # ── Final summary ─────────────────────────────────────────────────────
        print(f"\n{'#'*55}")
        print("  FINAL SUMMARY ACROSS BOTH TIERS")
        print(f"{'#'*55}")
        print(f"\n  {'Model':<28} {'tweet_eval acc':>15} {'hard examples acc':>18}")
        print(f"  {'-'*63}")
        print(f"  {'VADER':<28} {tweet_metrics['vader_accuracy']:>14.1%} {hard_metrics['vader_accuracy']:>17.1%}")
        print(f"  {'Pretrained RoBERTa':<28} {tweet_metrics['roberta_accuracy']:>14.1%} {hard_metrics['roberta_accuracy']:>17.1%}")
        print()


if __name__ == "__main__":
    main()