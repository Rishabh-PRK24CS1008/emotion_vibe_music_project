
from textblob import TextBlob

def analyze_text(text):

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "positive"

    elif polarity < 0:
        return "negative"

    else:
        return "neutral"
