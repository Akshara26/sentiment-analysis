from sentiment_analysis.model.inference import SentimentAnalyzer

def test_positive():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("This is absolutely amazing, I love it!")
    assert result.sentiment == "positive"
    assert result.sentiment_score > 0.7

def test_negative():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("This is terrible and I hate everything about it.")
    assert result.sentiment == "negative"

def test_neutral():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("The meeting is on Thursday at 2pm.")
    assert result.sentiment == "neutral"

def test_emotion_joy():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("I just got my dream job, this is the best day ever!")
    assert result.emotion == "joy"

def test_batch():
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_batch(["great!", "awful..."])
    assert len(results) == 2