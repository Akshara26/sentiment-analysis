# 🧠 Sentiment Analyzer

A real-time sentiment and emotion tweet analysis platform. Analyze tweets or paste any text to get instant sentiment (positive/negative/neutral) and emotion (joy, anger, sadness, fear, surprise, disgust) breakdowns — powered by fine-tuned transformer models.

Built as a demonstration of why model selection matters: VADER scores **53% accuracy** on the full tweet_eval benchmark (12,284 examples). This project's RoBERTa-based approach scores **72%** on the same benchmark — and **95%** on a curated diagnostic set of slang, sarcasm, and negation examples.

---

## Model Performance

Two-tier evaluation — standard benchmark + targeted diagnostic set:

### Tier 1: tweet_eval benchmark (n=12,284, full test split)

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| VADER (2014 baseline) | 53.0% | 52.6% |
| Pretrained RoBERTa | **72.2%** | **72.4%** |
| Fine-tuned DistilBERT | *in progress* | *in progress* |

VADER at 53% is barely above random on a 3-class problem. RoBERTa at 72% is consistent with published results for this model on this benchmark.

### Tier 2: Hard examples diagnostic set (n=20)
Curated cases targeting slang, sarcasm, and negation — the specific failure modes of rule-based models:

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| VADER (2014 baseline) | 65.0% | 66.1% |
| Pretrained RoBERTa | **95.0%** | **94.4%** |

The gap is larger on hard examples (30 points) than on tweet_eval (19 points) — confirming the diagnostic set isolates exactly the cases where VADER breaks down.

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
Evaluation Pipeline (VADER vs RoBERTa on tweet_eval + hard examples, tracked in MLflow + W&B)
```

---

## Features

- **Dual-model inference** — sentiment via `cardiffnlp/twitter-roberta-base-sentiment-latest` (trained on 124M tweets) + emotion via `j-hartmann/emotion-english-distilroberta-base` (6-class: joy, anger, sadness, fear, surprise, disgust)
- **VADER baseline comparison** — `/compare` endpoints run VADER and RoBERTa side by side
- **Two-tier evaluation pipeline** — tweet_eval benchmark (12,284 examples) + curated diagnostic set, logged to MLflow
- **REST API** — FastAPI with auto-generated OpenAPI docs at `/docs`
- **Live dashboard** — Streamlit UI for real-time subreddit analysis
- **Docker** — one-command local deployment via Docker Compose
- **CI/CD** — GitHub Actions running `ruff` and `pytest` on every push

---

## Quickstart

### Option 1: Docker (recommended)

```bash
git clone https://github.com/Akshara26/sentiment-analysis.git
cd sentiment-analysis
cp .env.example .env  # fill in Reddit API credentials
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
Compare VADER vs RoBERTa side by side.
```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"text": "That presentation was fire"}'
```
```json
{
  "vader_sentiment": "negative",
  "vader_score": 0.5106,
  "roberta_sentiment": "positive",
  "roberta_score": 0.9187,
  "models_agree": false
}
```

### `GET /compare/hard-examples`
Run the full diagnostic test set and return all comparison results.

### `GET /health`
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Running the Evaluation

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
poetry run python -m sentiment_analysis.training.evaluate
```

View results in MLflow:
```bash
poetry run mlflow ui
# open http://localhost:5000
```

---

## Running Tests

```bash
poetry run pytest tests/ -v
```
```
5 passed in 11.46s
```

---

## Project Structure

```
sentiment_analysis/
├── model/
│   └── inference.py              # Dual-model inference (RoBERTa + DistilRoBERTa)
├── api/
│   └── app.py                    # FastAPI REST API
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── training/
│   ├── comparison.py             # VADER vs RoBERTa side-by-side
│   ├── scrape_training_data.py   # Reddit scraping + weak supervision labeling
│   ├── train.py                  # DistilBERT fine-tuning with W&B + MLflow
│   └── evaluate.py               # Two-tier evaluation pipeline
├── scrapping/                    # Reddit data scraping (PRAW)
├── data_transformation/          # ML preprocessing pipeline
└── storage/                      # Supabase + S3 integration

tests/
└── test_inference.py

.github/workflows/ci.yml          # ruff + pytest on every push
Dockerfile
docker-compose.yml
```

---

## Known Limitations

- RoBERTa's 72% on tweet_eval leaves room for improvement — fine-tuning on domain-specific data is the next step
- Neither model reliably detects sarcasm; fine-tuning on the SARC corpus is on the roadmap
- Emotion classifier can misread hyperbolic enthusiasm as fear (out-of-distribution failure on informal language)

---

## Tech Stack

- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — model inference and fine-tuning
- [FastAPI](https://fastapi.tiangolo.com/) — REST API
- [Streamlit](https://streamlit.io/) — dashboard
- [NLTK VADER](https://www.nltk.org/api/nltk.sentiment.vader.html) — baseline comparison
- [tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) — benchmark dataset (12,284 examples)
- [MLflow](https://mlflow.org/) + [Weights & Biases](https://wandb.ai/) — experiment tracking
- [Poetry](https://python-poetry.org/) — dependency management
- [Docker](https://www.docker.com/) — containerization
- Original pipeline architecture by [RedhaWassim](https://github.com/RedhaWassim/Sentiment-Analysis)