"""
Scrape Reddit comments and auto-label them using the existing RoBERTa models.
This is weak supervision — we use our best current model to generate training labels.

Usage:
    poetry run python -m sentiment_analysis.training.scrape_training_data
"""

import os
import json
import praw
from dotenv import load_dotenv
from datetime import datetime
from sentiment_analysis.model.inference import SentimentAnalyzer

load_dotenv()

# Subreddits chosen for diversity of language and sentiment
SUBREDDITS = [
    "technology",       # neutral/mixed tech discussions
    "worldnews",        # negative-leaning news reactions
    "happy",            # positive content
    "rant",             # negative content
    "mildlyinteresting",# neutral/positive
    "tifu",             # negative/self-deprecating
    "gaming",           # mixed, lots of slang
    "investing",        # mixed, financial language
]

COMMENTS_PER_SUBREDDIT = 100
OUTPUT_PATH = "data/training_data.jsonl"


def get_reddit_client():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "sentiment-bot/1.0"),
    )


def scrape_comments(reddit, subreddit_name: str, limit: int) -> list[str]:
    comments = []
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.hot(limit=20):
        post.comments.replace_more(limit=0)
        for comment in post.comments.list():
            text = comment.body.strip()
            # Filter out low quality comments
            if len(text) > 20 and len(text) < 500 and text != "[deleted]" and text != "[removed]":
                comments.append(text)
            if len(comments) >= limit:
                break
        if len(comments) >= limit:
            break
    return comments[:limit]


def main():
    os.makedirs("data", exist_ok=True)

    print("Loading sentiment analyzer...")
    analyzer = SentimentAnalyzer()

    print("Connecting to Reddit...")
    reddit = get_reddit_client()

    all_samples = []

    for subreddit_name in SUBREDDITS:
        print(f"Scraping r/{subreddit_name}...")
        try:
            comments = scrape_comments(reddit, subreddit_name, COMMENTS_PER_SUBREDDIT)
            print(f"  Got {len(comments)} comments, labeling...")

            for comment in comments:
                result = analyzer.analyze(comment)
                # Only keep high-confidence labels for training quality
                if result.sentiment_score >= 0.80:
                    all_samples.append({
                        "text": comment,
                        "label": result.sentiment,
                        "confidence": result.sentiment_score,
                        "emotion": result.emotion,
                        "subreddit": subreddit_name,
                        "scraped_at": datetime.utcnow().isoformat(),
                    })

            print(f"  Kept {len([s for s in all_samples if s['subreddit'] == subreddit_name])} high-confidence samples")

        except Exception as e:
            print(f"  Failed r/{subreddit_name}: {e}")
            continue

    # Save to JSONL
    with open(OUTPUT_PATH, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\nDone! Saved {len(all_samples)} samples to {OUTPUT_PATH}")

    # Print label distribution
    from collections import Counter
    labels = Counter(s["label"] for s in all_samples)
    print(f"Label distribution: {dict(labels)}")


if __name__ == "__main__":
    main()