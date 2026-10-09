"""The sentiment classifier: a TF-IDF + logistic regression pipeline with train/evaluate methods."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment.preprocessing import normalize_text

# Words of any length (so "i" and "u" count), plus "!" and "?" which carry tone.
TOKEN_PATTERN = r"(?u)\b\w+\b|[!?]"


@dataclass(frozen=True)
class Prediction:
    """The result of classifying one text.

    `confidence` is the model's probability for `sentiment`, so it is always
    at least 0.5. Values near 0.5 mean the model is unsure.
    """

    sentiment: str
    confidence: float


class SentimentClassifier:
    """Binary (positive/negative) sentiment classifier for tweets.

    Text is normalized with `normalize_text`, turned into TF-IDF features over
    word unigrams and bigrams (bigrams capture short phrases like "not good"),
    and classified with logistic regression. Preprocessing is part of the
    pipeline, so training and inference can't drift apart.

    Typical use:
        classifier = SentimentClassifier().train(texts, labels)
        classifier.evaluate("I love this!")  # Prediction(sentiment="positive", confidence=0.9...)
        classifier.save("models/sentiment_model.joblib")
    """

    def __init__(
        self,
        regularization: float = 1.0,
        max_features: int | None = 500_000,
        min_df: int = 2,
    ) -> None:
        """Set hyperparameters. Nothing is fitted until `train` is called.

        Args:
            regularization: Inverse regularization strength (logistic regression's `C`).
                Smaller values give a simpler, more strongly regularized model.
            max_features: Keep only this many of the most frequent unigrams/bigrams.
                Bounds the model's size and memory use. None keeps all of them.
            min_df: Ignore terms that appear in fewer than this many tweets.
        """
        self.regularization = regularization
        self.max_features = max_features
        self.min_df = min_df
        self._pipeline: Pipeline | None = None

    def train(self, texts: Sequence[str], labels: Sequence[str]) -> "SentimentClassifier":
        """Fit the classifier on raw tweet texts and their labels. Returns self."""
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=normalize_text,
                token_pattern=TOKEN_PATTERN,
                ngram_range=(1, 2),
                min_df=self.min_df,
                max_features=self.max_features,
                sublinear_tf=True,
                dtype=np.float32,
            )),
            ("model", LogisticRegression(C=self.regularization, max_iter=1000)),
        ])
        pipeline.fit(list(texts), list(labels))
        self._pipeline = pipeline
        return self

    def evaluate(self, text: str) -> Prediction:
        """Classify one text and return its sentiment with a confidence score."""
        return self.predict([text])[0]

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Classify many texts at once (much faster than calling `evaluate` in a loop)."""
        pipeline = self._fitted_pipeline()
        probabilities = pipeline.predict_proba(list(texts))
        classes = pipeline.classes_
        best = probabilities.argmax(axis=1)
        return [
            Prediction(sentiment=str(classes[i]), confidence=float(row[i]))
            for row, i in zip(probabilities, best)
        ]

    def save(self, path: str | Path) -> None:
        """Save the fitted classifier to a file, creating parent directories if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "SentimentClassifier":
        """Load a classifier saved with `save`.

        Only load files you created yourself: joblib files can run code when loaded.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Train one first with: python -m sentiment.train"
            )
        classifier = joblib.load(path)
        if not isinstance(classifier, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return classifier

    def _fitted_pipeline(self) -> Pipeline:
        if self._pipeline is None:
            raise RuntimeError("The classifier has not been trained yet. Call train() first.")
        return self._pipeline
