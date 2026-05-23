import streamlit as st
import praw
import os
from dotenv import load_dotenv
from sentiment_analysis.model.inference import SentimentAnalyzer

load_dotenv()

st.set_page_config(
    page_title="Reddit Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .metric-card {
        background: #0f0f0f;
        border: 1px solid #222;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .sentiment-positive { color: #00ff88; font-weight: 700; font-size: 1.4rem; }
    .sentiment-negative { color: #ff4466; font-weight: 700; font-size: 1.4rem; }
    .sentiment-neutral  { color: #aaaaaa; font-weight: 700; font-size: 1.4rem; }

    .emotion-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .post-card {
        background: #111;
        border: 1px solid #1e1e1e;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

EMOTION_COLORS = {
    "joy":      "#FFD700",
    "anger":    "#FF4444",
    "sadness":  "#4488FF",
    "fear":     "#AA44FF",
    "surprise": "#FF8800",
    "disgust":  "#44CC88",
    "neutral":  "#888888",
}

SENTIMENT_CLASS = {
    "positive": "sentiment-positive",
    "negative": "sentiment-negative",
    "neutral":  "sentiment-neutral",
}

# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_analyzer():
    return SentimentAnalyzer()

# ── Reddit client ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_reddit():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "sentiment-bot/1.0"),
    )

def fetch_posts(subreddit_name: str, limit: int = 15):
    reddit = get_reddit()
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=limit):
        posts.append({
            "title": post.title,
            "score": post.score,
            "url": f"https://reddit.com{post.permalink}",
            "text": post.title + (" " + post.selftext if post.selftext else ""),
        })
    return posts

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧠 Reddit Sentiment Analyzer")
st.markdown("*Real-time sentiment & emotion analysis on any subreddit*")
st.divider()

col_input, col_options = st.columns([3, 1])
with col_input:
    subreddit = st.text_input("Subreddit", value="technology", placeholder="e.g. worldnews, stocks, gaming")
with col_options:
    post_limit = st.slider("Posts to analyze", 5, 25, 10)

# ── Single text mode ──────────────────────────────────────────────────────────
st.markdown("#### Or analyze any text directly")
custom_text = st.text_area("Paste any text", height=80, placeholder="Paste a Reddit post, tweet, or any text...")

analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

if analyze_btn:
    analyzer = load_analyzer()

    if custom_text.strip():
        # Single text mode
        with st.spinner("Analyzing..."):
            result = analyzer.analyze(custom_text)

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sentiment", result.sentiment.upper())
        c2.metric("Confidence", f"{result.sentiment_score * 100:.1f}%")
        c3.metric("Emotion", result.emotion.capitalize())
        c4.metric("Emotion Score", f"{result.emotion_score * 100:.1f}%")

    else:
        # Subreddit mode
        with st.spinner(f"Fetching r/{subreddit} posts..."):
            try:
                posts = fetch_posts(subreddit, post_limit)
            except Exception as e:
                st.error(f"Could not fetch r/{subreddit}: {e}")
                st.stop()

        with st.spinner("Running sentiment analysis..."):
            results = analyzer.analyze_batch([p["text"] for p in posts])

        # ── Summary metrics ───────────────────────────────────────────────────
        st.divider()
        st.subheader(f"r/{subreddit} — {len(results)} posts analyzed")

        counts = {"positive": 0, "negative": 0, "neutral": 0}
        emotion_counts: dict[str, int] = {}
        avg_score = 0.0

        for r in results:
            counts[r.sentiment] = counts.get(r.sentiment, 0) + 1
            emotion_counts[r.emotion] = emotion_counts.get(r.emotion, 0) + 1
            avg_score += r.sentiment_score

        avg_score /= len(results)
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("✅ Positive", counts["positive"])
        m2.metric("❌ Negative", counts["negative"])
        m3.metric("➖ Neutral",  counts["neutral"])
        m4.metric("🎭 Top Emotion", dominant_emotion.capitalize())

        # ── Sentiment bar ─────────────────────────────────────────────────────
        total = len(results)
        pos_pct = counts["positive"] / total
        neg_pct = counts["negative"] / total
        neu_pct = counts["neutral"]  / total

        st.markdown("**Sentiment breakdown**")
        bar_html = f"""
        <div style="display:flex; height:12px; border-radius:6px; overflow:hidden; margin-bottom:16px;">
            <div style="width:{pos_pct*100:.1f}%; background:#00ff88;"></div>
            <div style="width:{neu_pct*100:.1f}%; background:#555555;"></div>
            <div style="width:{neg_pct*100:.1f}%; background:#ff4466;"></div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

        # ── Individual posts ──────────────────────────────────────────────────
        st.subheader("Post breakdown")
        for post, result in zip(posts, results):
            color = EMOTION_COLORS.get(result.emotion, "#888")
            s_class = SENTIMENT_CLASS.get(result.sentiment, "sentiment-neutral")
            st.markdown(f"""
            <div class="post-card">
                <div style="margin-bottom:8px;">
                    <a href="{post['url']}" target="_blank" style="color:#ddd; text-decoration:none; font-size:0.95rem;">
                        {post['title'][:120]}{'...' if len(post['title']) > 120 else ''}
                    </a>
                </div>
                <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
                    <span class="{s_class}">{result.sentiment.upper()}</span>
                    <span style="color:#888; font-size:0.85rem;">{result.sentiment_score*100:.0f}% confident</span>
                    <span class="emotion-tag" style="background:{color}22; color:{color}; border:1px solid {color}44;">
                        {result.emotion.capitalize()}
                    </span>
                    <span style="color:#555; font-size:0.8rem;">⬆ {post['score']:,}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)