# 🧠 Reddit Sentiment Analyzer

A real-time sentiment and emotion analysis platform for Reddit. Analyze any subreddit or paste any text to get instant sentiment (positive/negative/neutral) and emotion (joy, anger, sadness, fear, surprise, disgust) breakdowns — powered by fine-tuned transformer models.

Built as a demonstration of why model selection matters: VADER scores **65% accuracy** on informal internet language. This project's RoBERTa-based approach scores **95%** on the same test set.

---

## Architecture

```
Reddit API (PRAW)
      ↓
Data Pipeline (scraping → AWS Kinesis Firehose → S3 → Lambda)
      ↓
ML Inference (RoBERTa + DistilRoBERTa)
      ↓
FastAPI REST API ←→ Streamlit Dashboard
      ↓
Evaluation Pipeline (VADER vs RoBERTa, tracked in MLflow + W&B)
```

---

## Features

- **Dual-model inference** — sentiment via `cardiffnlp/twitter-roberta-base-sentiment-latest` (trained on 124M tweets) + emotion via `j-hartmann/emotion-english-distilroberta-base` (6-class: joy, anger, sadness, fear, surprise, disgust)
- **VADER baseline comparison** — `/compare` endpoints run VADER and RoBERTa side by side so you can see exactly where rule-based models fail on modern internet language
- **REST API** — FastAPI with auto-generated OpenAPI docs at `/docs`
- **Live dashboard** — Streamlit UI for real-time subreddit analysis with sentiment breakdown and per-post results
- **Evaluation pipeline** — ground truth test set with MLflow experiment tracking
- **Docker** — one-command local deployment via Docker Compose
- **CI/CD** — GitHub Actions running `ruff` linting and `pytest` on every push

---

## Model Performance

Evaluated on 20 curated hard examples (slang, sarcasm, negation, internet vernacular):

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| VADER (2014 baseline) | 65% | 66% |
| Pretrained RoBERTa | **95%** | **94%** |
| Fine-tuned DistilBERT | *in progress* | *in progress* |

VADER fails on modern slang ("fire", "slaps", "goes hard") and sarcasm ("oh great, another Monday meeting"). RoBERTa handles both because it was trained on 124M tweets from 2020 — a distribution much closer to how people actually write on Reddit today.

---

## Quickstart

### Option 1: Docker (recommended)

```bash
git clone https://github.com/Akshara26/sentiment-analysis.git
cd sentiment-analysis

# Create .env file
cp .env.example .env  # then fill in your Reddit API credentials

docker compose up --build
```

- API → http://localhost:8000/docs
- Dashboard → http://localhost:8501

### Option 2: Local

```bash
git clone https://github.com/Akshara26/sentiment-analysis.git
cd sentiment-analysis

poetry install
```

Create a `.env` file:
```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=sentiment-bot/1.0
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

Run the API:
```bash
poetry run uvicorn sentiment_analysis.api.app:app --reload
```

Run the dashboard (new terminal):
```bash
poetry run streamlit run sentiment_analysis/dashboard/app.py
```

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

### `POST /compare`
Compare VADER vs RoBERTa on a single text — useful for understanding where models disagree.

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"text": "That presentation was fire"}'
```

```json
{
  "text": "That presentation was fire",
  "vader_sentiment": "negative",
  "vader_score": 0.5106,
  "roberta_sentiment": "positive",
  "roberta_score": 0.9187,
  "roberta_emotion": "anger",
  "models_agree": false
}
```

### `GET /compare/hard-examples`
Run the full curated test set and return all comparison results.

### `GET /health`
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```
---

## Project Structure

```
sentiment_analysis/
├── model/
│   └── inference.py              # SentimentAnalyzer — dual-model inference
├── api/
│   └── app.py                    # FastAPI REST API
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── training/
│   ├── comparison.py             # VADER vs RoBERTa side-by-side
│   ├── scrape_training_data.py   # Reddit scraping + weak supervision labeling
│   ├── train.py                  # DistilBERT fine-tuning with W&B + MLflow
│   └── evaluate.py               # Ground truth evaluation across all models
├── scrapping/                    # Reddit data scraping (PRAW)
├── data_transformation/          # ML preprocessing pipeline
└── storage/                      # Supabase + S3 integration

tests/
└── test_inference.py             # Unit tests for the ML layer

.github/workflows/
└── ci.yml                        # GitHub Actions — ruff + pytest on every push

Dockerfile                        # Container for API and dashboard
docker-compose.yml                # Runs API + dashboard together
```

---

## Tech Stack

- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — model inference and fine-tuning
- [FastAPI](https://fastapi.tiangolo.com/) — REST API
- [Streamlit](https://streamlit.io/) — dashboard
- [NLTK VADER](https://www.nltk.org/api/nltk.sentiment.vader.html) — baseline comparison
- [MLflow](https://mlflow.org/) + [Weights & Biases](https://wandb.ai/) — experiment tracking
- [Poetry](https://python-poetry.org/) — dependency management
- [Docker](https://www.docker.com/) — containerization
