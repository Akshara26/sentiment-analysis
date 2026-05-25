from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentiment_analysis.model.inference import SentimentAnalyzer, SentimentResult
from sentiment_analysis.training.comparison import SentimentComparator, ComparisonResult

app = FastAPI(
    title="Reddit Sentiment API",
    description="Analyze sentiment and emotion in Reddit posts/comments",
    version="1.0.0",
)

# Load models once at startup
analyzer = SentimentAnalyzer()
comparator = SentimentComparator()


class TextInput(BaseModel):
    text: str


class BatchInput(BaseModel):
    texts: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=SentimentResult)
def analyze(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return analyzer.analyze(body.text)


@app.post("/analyze/batch", response_model=list[SentimentResult])
def analyze_batch(body: BatchInput):
    if not body.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    if len(body.texts) > 50:
        raise HTTPException(status_code=400, detail="Max 50 texts per batch")
    return analyzer.analyze_batch(body.texts)


@app.post("/compare", response_model=ComparisonResult)
def compare(body: TextInput):
    """Compare VADER vs RoBERTa on a single text."""
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return comparator.compare(body.text)


@app.get("/compare/hard-examples", response_model=list[ComparisonResult])
def hard_examples():
    """Run the curated hard examples test set and return comparison results."""
    return comparator.run_hard_examples()