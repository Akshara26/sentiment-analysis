# Reddit Data Engineering Pipeline

A production-ready data engineering pipeline that ingests Reddit posts and comments with Python, delivers batches to AWS Kinesis Firehose, lands raw data in S3, publishes processing events with SNS, transforms with AWS Lambda, and stores curated datasets in Supabase.

<img width="1031" height="726" alt="image" src="https://github.com/user-attachments/assets/f103f0cf-d948-4b18-831b-d266b73c1cc0" />

## End-to-End Architecture

Reddit → Python Ingestion → AWS Kinesis Firehose → AWS S3 → AWS SNS → AWS Lambda → Supabase

- Reddit: data source (subreddits feed)
- Python Ingestion: PRAW-based scrapers with batching and retry/backoff
- Kinesis Firehose: reliable delivery stream to S3
- S3: raw data lake storage (partitioned)
- SNS: event notifications to trigger downstream processing
- Lambda: stateless validation, transformation, enrichment
- Supabase: relational storage for curated posts and comments

## Repository Layout (actual)

```
Notebook/
README.md
makefile
pyproject.toml
poetry.lock
mypy.ini

sentiment_analysis/
  __init__.py
  logging_config.py            # project-wide logger
  exception.py                 # custom exception utilities

  scrapping/
    __init__.py
    data_scrapping.py          # Reddit ingestion (PRAW): posts + comments, batching

  AWS_processing/
    __init__.py
    kinesis_firehose/
      __init__.py
      kinesis_firehose.py      # Firehose client and PutRecordBatch wrapper
    lambda_function/
      __init__.py
      lambda_function.py       # SNS-triggered Lambda: read S3 → transform ( basic ETL )  → load

  data_ingestion/              # ingest data from the supabase for machine learning purpose
  data_transformation/         # data transformation for machine learning purpose 
  storage/
    db_schema.sql              # Supabase schema (posts, comments)
  utils/                       # (reserved for helpers)
```

## What Each Module Does

- sentiment_analysis/scrapping/data_scrapping.py
  - Authenticates with Reddit using PRAW
  - Pulls posts/comments by topic (best/new/trending)
  - Batches and prepares payloads for delivery

- sentiment_analysis/AWS_processing/kinesis_firehose/kinesis_firehose.py
  - Thin client around Firehose
  - Handles PutRecord/PutRecordBatch with basic retries
  - Expects JSON payloads (posts/comments)

- sentiment_analysis/AWS_processing/lambda_function/lambda_function.py
  - SNS-triggered entrypoint
  - Reads new S3 objects (from Firehose)
  - Validates/cleans/transforms records
  - Upserts curated rows into Supabase

- sentiment_analysis/storage/db_schema.sql
  - SQL schema for Supabase tables: posts and comments
  - Indexes for common query patterns
- sentiment_analysis/data_ingestion
  - ML-focused data ingestion modules for model training workflows
  - Contains utilities for data staging and preprocessing
- sentiment_analysis/data_transformation
  - ML-focused data transformation utilities for feature engineering
  - Handles data normalization and preprocessing for machine learning models

## Getting Started

### Prerequisites
- Python 3.10+
- Poetry
- AWS access (Kinesis Firehose, S3, SNS, Lambda)
- Supabase project (database URL + API key)

### Install
```bash
poetry install
```

### Configure
Create an .env file (see your own secrets store) with at least:
- Reddit credentials (PRAW)
- AWS credentials (profile/keys or role)
- Supabase URL + Key

### Run Ingestion (local)
```bash
poetry run python sentiment_analysis/scrapping/data_scrapping.py
```
This will fetch batches from Reddit and deliver them to Kinesis Firehose, which writes to S3.

### Lambda Processing (cloud)
- Deploy `sentiment_analysis/AWS_processing/lambda_function/lambda_function.py` as a Lambda
- Configure SNS to trigger the Lambda when new S3 objects land
- Lambda reads the S3 object, transforms, and loads into Supabase using the schema in `storage/db_schema.sql`

## Data Flow Details

1. Scraper pulls posts/comments → batches to Firehose
2. Firehose delivers to S3 under time-based prefixes
3. S3 PUT event → SNS notification
4. SNS triggers Lambda
5. Lambda reads object, validates/transforms, writes to Supabase

## Development

- make lint — ruff/mypy/black checks
- make test — run tests (add tests under tests/)
- Notebooks in Notebook/ for exploration only

## License
MIT
