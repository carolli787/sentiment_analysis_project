"""Twitter sentiment classifier: data loading, text preprocessing, training and inference.

This package has no Django dependency, so the notebook, the training script and
the API all share exactly the same code.
"""

from sentiment.classifier import Prediction, SentimentClassifier

__all__ = ["Prediction", "SentimentClassifier"]
