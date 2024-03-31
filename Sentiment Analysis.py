from textblob import TextBlob


def analyze_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)

    # Get polarity score
    polarity = blob.sentiment.polarity

    # Determine sentiment based on polarity score
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment


# Example usage
text = "I love this tool, it's amazing!"
sentiment = analyze_sentiment(text)
print("Sentiment:", sentiment)
