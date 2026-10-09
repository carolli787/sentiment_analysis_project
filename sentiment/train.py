"""Train the sentiment classifier, report its metrics and save it to disk.

Usage (from the project root):
    python -m sentiment.train
    python -m sentiment.train --sample-size 100000   # quick run on a subsample

Steps:
  1. Load and clean the training data (drop duplicates and conflicting labels).
  2. Split it into train and validation sets (stratified, fixed seed).
  3. Train one model per regularization value and pick the best on validation.
  4. Score the chosen model once on the manual test set (positive/negative rows only).
  5. Save the model and a JSON file with its metrics and data details.
"""

import argparse
import json
import logging
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sentiment.classifier import SentimentClassifier
from sentiment.data import (
    BINARY_LABELS,
    PROJECT_ROOT,
    TEST_PATH,
    TRAIN_PATH,
    binary_test_rows,
    clean_training_data,
    load_tweets,
    split_train_validation,
)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.joblib"
DEFAULT_REGULARIZATION_GRID = (0.25, 0.5, 1.0, 2.0)

logger = logging.getLogger(__name__)


def score(classifier: SentimentClassifier, df: pd.DataFrame) -> dict:
    """Accuracy, macro F1 and confusion matrix of `classifier` on a labeled DataFrame."""
    predicted = [p.sentiment for p in classifier.predict(df["text"])]
    matrix = confusion_matrix(df["label"], predicted, labels=list(BINARY_LABELS))
    return {
        "rows": len(df),
        "accuracy": round(accuracy_score(df["label"], predicted), 4),
        "macro_f1": round(f1_score(df["label"], predicted, average="macro"), 4),
        # Rows are the true label, columns the predicted label, both in BINARY_LABELS order.
        "confusion_matrix": {"labels": list(BINARY_LABELS), "matrix": matrix.tolist()},
    }


def run(
    output_path: Path,
    regularization_grid: tuple[float, ...],
    validation_fraction: float,
    seed: int,
    sample_size: int | None,
) -> dict:
    """Run the full training procedure and return the saved metadata."""
    logger.info("Loading %s", TRAIN_PATH.name)
    raw_df = load_tweets(TRAIN_PATH)
    clean_df, cleaning = clean_training_data(raw_df)
    logger.info("Cleaning: %s", cleaning)

    if sample_size is not None and sample_size < len(clean_df):
        # Stratified so a small sample still has both labels in equal measure
        clean_df = clean_df.groupby("label").sample(n=sample_size // 2, random_state=seed)

    train_df, validation_df = split_train_validation(clean_df, validation_fraction, seed)
    test_df = binary_test_rows(load_tweets(TEST_PATH))
    logger.info("Split sizes: train=%d validation=%d test=%d", len(train_df), len(validation_df), len(test_df))

    # Model selection uses the validation set only. The test set is used once, at the end.
    candidates = []
    for regularization in regularization_grid:
        started = time.perf_counter()
        classifier = SentimentClassifier(regularization=regularization)
        classifier.train(train_df["text"], train_df["label"])
        validation_metrics = score(classifier, validation_df)
        logger.info(
            "regularization=%s: validation accuracy=%.4f macro F1=%.4f (%.0fs)",
            regularization,
            validation_metrics["accuracy"],
            validation_metrics["macro_f1"],
            time.perf_counter() - started,
        )
        candidates.append((validation_metrics["accuracy"], regularization, classifier, validation_metrics))

    _, best_regularization, best_classifier, best_validation = max(candidates, key=lambda c: c[0])
    test_metrics = score(best_classifier, test_df)
    logger.info(
        "Chose regularization=%s. Test accuracy=%.4f macro F1=%.4f",
        best_regularization,
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
    )

    best_classifier.save(output_path)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "type": "TF-IDF (word 1-2 grams) + logistic regression",
            "regularization": best_regularization,
            "max_features": best_classifier.max_features,
            "min_df": best_classifier.min_df,
        },
        "data": {
            "train_file": TRAIN_PATH.name,
            "test_file": TEST_PATH.name,
            "cleaning": asdict(cleaning),
            "sample_size": sample_size,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "split_sizes": {"train": len(train_df), "validation": len(validation_df), "test": len(test_df)},
        },
        "metrics": {
            "validation_by_regularization": {str(c[1]): c[0] for c in candidates},
            "validation": best_validation,
            "test": test_metrics,
        },
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Saved model to %s and metadata to %s", output_path, metadata_path)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tweet sentiment classifier.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="Where to save the model.")
    parser.add_argument(
        "--regularization",
        type=float,
        nargs="+",
        default=list(DEFAULT_REGULARIZATION_GRID),
        help="Logistic regression C values to compare on the validation set.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Train on a balanced random subsample of this many tweets (for quick experiments).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    metadata = run(
        output_path=args.output,
        regularization_grid=tuple(args.regularization),
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        sample_size=args.sample_size,
    )
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
