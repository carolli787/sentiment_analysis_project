"""Shared test helpers."""

from sentiment.classifier import SentimentClassifier

POSITIVE_TEXTS = [
    "i love this so much",
    "what a great day",
    "this is awesome and fun",
    "love it, best thing ever",
    "happy and excited today",
    "great job, love the result",
]
NEGATIVE_TEXTS = [
    "i hate this so much",
    "what a terrible day",
    "this is awful and sad",
    "hate it, worst thing ever",
    "sad and tired today",
    "bad job, hate the result",
]


def train_tiny_classifier() -> SentimentClassifier:
    """A classifier trained in milliseconds on a handful of obvious examples.

    Lets tests exercise the real pipeline without the full data set or a saved model.
    """
    texts = POSITIVE_TEXTS + NEGATIVE_TEXTS
    labels = ["positive"] * len(POSITIVE_TEXTS) + ["negative"] * len(NEGATIVE_TEXTS)
    return SentimentClassifier(regularization=10.0, min_df=1).train(texts, labels)
