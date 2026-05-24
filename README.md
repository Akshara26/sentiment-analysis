# 🧠 Reddit Sentiment Analyzer

A real-time sentiment and emotion analysis platform for Reddit. Paste any text or point it at a subreddit to get instant sentiment (positive/negative/neutral) and emotion (joy, anger, sadness, fear, surprise, disgust) breakdowns — powered by fine-tuned transformer models.

---

## Architecture

```
Reddit API (PRAW)
      ↓
Data Pipeline (scraping → S3 → Lambda)
      ↓
ML Inference (RoBERTa + DistilRoBERTa)
      ↓
FastAPI REST API ←→ Streamlit Dashboard
```

---

## Features

- **Dual-model inference** — sentiment via `cardiffnlp/twitter-roberta-base-sentiment-latest` (trained on 124M tweets, handles informal Reddit language) + emotion via `j-hartmann/emotion-english-distilroberta-base` (6-class: joy, anger, sadness, fear, surprise, disgust)
- **REST API** — FastAPI endpoint with auto-generated OpenAPI docs at `/docs`
- **Live dashboard** — Streamlit UI for real-time subreddit analysis with sentiment breakdown bar and per-post results
- **Data pipeline** — Reddit scraping → AWS Kinesis Firehose → S3 → Lambda → Supabase
- **CI/CD** — GitHub Actions running `ruff` linting and `pytest` on every push

---

## Quickstart

### Prerequisites
- Python 3.12
- Poetry

```bash
git clone https://github.com/Akshara26/sentiment-analysis.git
cd sentiment-analysis
poetry install
```

Create a `.env` file in the project root:

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=sentiment-bot/1.0
```

---

## Running Locally

### ML Inference (smoke test)
```bash
poetry run python -c "
from sentiment_analysis.model.inference import SentimentAnalyzer
analyzer = SentimentAnalyzer()
print(analyzer.analyze('This is absolutely incredible!'))
"
```

### FastAPI
```bash
poetry run uvicorn sentiment_analysis.api.app:app --reload
```
→ API docs at http://localhost:8000/docs

### Streamlit Dashboard
```bash
poetry run streamlit run sentiment_analysis/dashboard/app.py
```
→ Dashboard at http://localhost:8501

---

## API Reference

### `POST /analyze`
Analyze a single piece of text.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I just got my dream job, this is incredible!"}'
```

```json
{
  "text": "I just got my dream job, this is incredible!",
  "sentiment": "positive",
  "sentiment_score": 0.9954,
  "emotion": "joy",
  "emotion_score": 0.9112
}
```

### `POST /analyze/batch`
Analyze up to 50 texts in one request.

```bash
curl -X POST http://localhost:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This is great!", "I am so frustrated right now."]}'
```

### `GET /health`
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Running Tests

```bash
poetry run pytest tests/ -v
```

```
tests/test_inference.py::test_positive PASSED
tests/test_inference.py::test_negative PASSED
tests/test_inference.py::test_neutral  PASSED
tests/test_inference.py::test_emotion_joy PASSED
tests/test_inference.py::test_batch    PASSED
5 passed in 11.46s
```

---

## Project Structure

```
sentiment_analysis/
├── model/
│   └── inference.py       # SentimentAnalyzer — dual-model inference
├── api/
│   └── app.py             # FastAPI REST API
├── dashboard/
│   └── app.py             # Streamlit dashboard
├── scrapping/             # Reddit data scraping (PRAW)
├── data_transformation/   # ML preprocessing pipeline
└── storage/               # Supabase + S3 integration

tests/
└── test_inference.py      # Unit tests for the ML layer

.github/
└── workflows/
    └── ci.yml             # GitHub Actions — ruff + pytest on every push
```

---

## Model Choice

| Model | Purpose | Why |
|-------|---------|-----|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment | Trained on 124M tweets — handles slang, abbreviations, and informal Reddit language far better than models trained on formal text |
| `j-hartmann/emotion-english-distilroberta-base` | Emotion | 6-class emotion classifier that goes beyond positive/negative to explain *why* people feel the way they do |

---

## Known Limitations

- The emotion classifier can misclassify profanity used as emphasis (e.g. "this shit is awesome" → disgust instead of joy) — a known weakness of models without enough slang context in training data
- Models are not fine-tuned on Reddit-specific data; performance may degrade on highly domain-specific subreddits

---

## Roadmap

- [ ] Deploy to HuggingFace Spaces (public demo link)
- [ ] Docker + Docker Compose for one-command local setup
- [ ] Sentiment spike alerting via Slack/email
- [ ] Data drift detection with Evidently AI
- [ ] `mypy` type checking in CI

---

## Built On

- [PRAW](https://praw.readthedocs.io/) — Reddit API wrapper
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — model inference
- [FastAPI](https://fastapi.tiangolo.com/) — REST API
- [Streamlit](https://streamlit.io/) — dashboard
- [Poetry](https://python-poetry.org/) — dependency management
- Original pipeline architecture by [RedhaWassim](https://github.com/RedhaWassim/Sentiment-Analysis)
