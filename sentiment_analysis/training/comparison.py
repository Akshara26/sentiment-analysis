"""
Baseline comparison: VADER vs RoBERTa sentiment models.

VADER is a rule-based sentiment analyzer designed for social media text (2014).
It's the industry default "quick and easy" solution.
This module compares it against our RoBERTa-based approach on hard examples.
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dataclasses import dataclass
from sentiment_analysis.model.inference import SentimentAnalyzer, SentimentResult

nltk.download("vader_lexicon", quiet=True)


@dataclass
class ComparisonResult:
    text: str
    # VADER
    vader_sentiment: str
    vader_score: float
    # RoBERTa
    roberta_sentiment: str
    roberta_score: float
    roberta_emotion: str
    # Agreement
    models_agree: bool


class SentimentComparator:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.roberta = SentimentAnalyzer()

    def _vader_predict(self, text: str) -> tuple[str, float]:
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            return "positive", round(compound, 4)
        elif compound <= -0.05:
            return "negative", round(abs(compound), 4)
        else:
            return "neutral", round(1 - abs(compound), 4)

    def compare(self, text: str) -> ComparisonResult:
        vader_sentiment, vader_score = self._vader_predict(text)
        roberta_result: SentimentResult = self.roberta.analyze(text)

        return ComparisonResult(
            text=text,
            vader_sentiment=vader_sentiment,
            vader_score=vader_score,
            roberta_sentiment=roberta_result.sentiment,
            roberta_score=roberta_result.sentiment_score,
            roberta_emotion=roberta_result.emotion,
            models_agree=vader_sentiment == roberta_result.sentiment,
        )

    def compare_batch(self, texts: list[str]) -> list[ComparisonResult]:
        return [self.compare(t) for t in texts]

    def run_hard_examples(self) -> list[ComparisonResult]:
        """
        Curated set of examples that expose VADER's weaknesses.
        These are the cases that matter for Reddit-style text.
        """
        hard_examples = [
            # Hyperbole as enthusiasm
            "This is insanely good, I can't believe how well it works",
            "I am literally dying this is so funny",
            "This movie absolutely destroyed me in the best way",

            # Sarcasm and irony
            "Oh great, another Monday morning meeting. Just what I needed.",
            "Wow, what a surprise. The train is late again.",
            "Sure, because that always works out perfectly.",

            # Negation
            "Not bad at all, actually really enjoyed it",
            "I wouldn't say this is terrible",
            "Can't complain, everything went smoothly",

            # Internet slang / enthusiasm
            "This track goes so hard",
            "That presentation was fire",
            "The new update actually slaps",

            # Clearly positive (both should get right)
            "I love this product, it works perfectly",
            "Best decision I ever made, highly recommend",

            # Clearly negative (both should get right)
            "This is a complete disaster, nothing works",
            "Absolutely terrible experience, would not recommend",
        ]
        return self.compare_batch(hard_examples)


def print_comparison_table(results: list[ComparisonResult]) -> None:
    print(f"\n{'='*100}")
    print(f"{'TEXT':<45} {'VADER':<12} {'RoBERTa':<12} {'EMOTION':<12} {'AGREE'}")
    print(f"{'='*100}")
    for r in results:
        text_preview = r.text[:42] + "..." if len(r.text) > 42 else r.text
        agree_symbol = "✅" if r.models_agree else "❌"
        print(
            f"{text_preview:<45} "
            f"{r.vader_sentiment:<12} "
            f"{r.roberta_sentiment:<12} "
            f"{r.roberta_emotion:<12} "
            f"{agree_symbol}"
        )
    print(f"{'='*100}")

    agreements = sum(1 for r in results if r.models_agree)
    print(f"\nModels agreed on {agreements}/{len(results)} examples ({agreements/len(results)*100:.0f}%)")
    print(f"Disagreements (where RoBERTa likely wins): {len(results) - agreements}\n")


if __name__ == "__main__":
    print("Loading models...")
    comparator = SentimentComparator()
    print("Running hard examples...\n")
    results = comparator.run_hard_examples()
    print_comparison_table(results)