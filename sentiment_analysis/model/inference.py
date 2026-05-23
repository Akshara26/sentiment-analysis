from transformers import pipeline
from dataclasses import dataclass


@dataclass
class SentimentResult:
    text: str
    sentiment: str        # positive / negative / neutral
    sentiment_score: float
    emotion: str          # joy, anger, sadness, fear, surprise, disgust
    emotion_score: float


class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipe = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            max_length=512,
        )
        self.emotion_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            truncation=True,
            max_length=512,
        )

    def analyze(self, text: str) -> SentimentResult:
        text = text[:1000]
        sentiment = self.sentiment_pipe(text)[0]
        emotion = self.emotion_pipe(text)[0]
        return SentimentResult(
            text=text,
            sentiment=sentiment["label"],
            sentiment_score=round(sentiment["score"], 4),
            emotion=emotion["label"],
            emotion_score=round(emotion["score"], 4),
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.analyze(t) for t in texts]