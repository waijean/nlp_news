from textblob import TextBlob


def calculate_polarity(text: str) -> float:
    _text = TextBlob(text)
    return _text.sentiment.polarity


def calculate_subjectivity(text: str) -> float:
    _text = TextBlob(text)
    return _text.sentiment.subjectivity
