FROM python:3.12-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY . .

RUN pip install -e .

RUN python -c "\
from transformers import pipeline; \
pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-sentiment-latest', truncation=True, max_length=512); \
pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', truncation=True, max_length=512)"

EXPOSE 8000

CMD ["uvicorn", "sentiment_analysis.api.app:app", "--host", "0.0.0.0", "--port", "8000"]