"""Loading, cleaning and splitting the Sentiment140 data set."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "trainingandtestdata"
TRAIN_PATH = DATA_DIR / "training.1600000.processed.noemoticon.csv"
TEST_PATH = DATA_DIR / "testdata.manual.2009.06.14.csv"

COLUMNS = ["polarity", "id", "date", "query", "user", "text"]
POLARITY_TO_LABEL = {0: "negative", 2: "neutral", 4: "positive"}
# The classifier is binary: the training data has no neutral examples.
BINARY_LABELS = ("negative", "positive")


def load_tweets(path: Path) -> pd.DataFrame:
    """Load a Sentiment140 CSV and add a readable `label` column.

    The files have no header row and are latin-1 encoded.
    """
    df = pd.read_csv(path, header=None, names=COLUMNS, encoding="latin-1")
    df["label"] = df["polarity"].map(POLARITY_TO_LABEL)
    return df


@dataclass(frozen=True)
class CleaningReport:
    """Row counts before and after cleaning, saved with the model for traceability."""

    rows_in: int
    conflicting_rows_dropped: int
    duplicate_rows_dropped: int
    rows_out: int


def clean_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove duplicate texts from the training data.

    - A text that appears with both labels is dropped entirely: at least one
      copy is mislabeled and there's no way to tell which.
    - Of the remaining duplicates, only the first copy is kept, so repeated
      (often spam) tweets don't count many times, and the same tweet can't land
      in both the training and validation splits.
    """
    labels_per_text = df.groupby("text")["label"].nunique()
    conflicting_texts = labels_per_text[labels_per_text > 1].index
    without_conflicts = df[~df["text"].isin(conflicting_texts)]
    deduplicated = without_conflicts.drop_duplicates(subset="text", keep="first")

    report = CleaningReport(
        rows_in=len(df),
        conflicting_rows_dropped=len(df) - len(without_conflicts),
        duplicate_rows_dropped=len(without_conflicts) - len(deduplicated),
        rows_out=len(deduplicated),
    )
    return deduplicated.reset_index(drop=True), report


def binary_test_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the positive and negative rows of the manual test set (drops neutral)."""
    return df[df["label"].isin(BINARY_LABELS)].reset_index(drop=True)


def split_train_validation(
    df: pd.DataFrame, validation_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle and split into train and validation sets with the same label balance.

    Shuffling matters because the raw training file is sorted by label.
    """
    train_df, validation_df = train_test_split(
        df,
        test_size=validation_fraction,
        stratify=df["label"],
        shuffle=True,
        random_state=seed,
    )
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True)
