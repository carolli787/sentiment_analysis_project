import unittest

import pandas as pd

from sentiment.data import binary_test_rows, clean_training_data, split_train_validation


def make_df(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["text", "label"])


class CleanTrainingDataTests(unittest.TestCase):
    def test_drops_every_copy_of_a_text_with_conflicting_labels(self):
        df = make_df([("meh", "positive"), ("meh", "negative"), ("meh", "positive"), ("yay", "positive")])
        cleaned, report = clean_training_data(df)
        self.assertEqual(cleaned["text"].tolist(), ["yay"])
        self.assertEqual(report.conflicting_rows_dropped, 3)

    def test_keeps_one_copy_of_a_duplicate_with_consistent_labels(self):
        df = make_df([("yay", "positive"), ("yay", "positive"), ("boo", "negative")])
        cleaned, report = clean_training_data(df)
        self.assertEqual(sorted(cleaned["text"]), ["boo", "yay"])
        self.assertEqual(report.duplicate_rows_dropped, 1)
        self.assertEqual(report.rows_in, 3)
        self.assertEqual(report.rows_out, 2)


class BinaryTestRowsTests(unittest.TestCase):
    def test_drops_neutral_rows(self):
        df = make_df([("a", "positive"), ("b", "neutral"), ("c", "negative")])
        self.assertEqual(binary_test_rows(df)["label"].tolist(), ["positive", "negative"])


class SplitTrainValidationTests(unittest.TestCase):
    def test_both_splits_get_both_labels_even_when_input_is_sorted_by_label(self):
        # Sorted like the raw file: all negatives first, then all positives
        df = make_df([(f"n{i}", "negative") for i in range(50)] + [(f"p{i}", "positive") for i in range(50)])
        train_df, validation_df = split_train_validation(df, validation_fraction=0.2, seed=0)
        self.assertEqual(len(validation_df), 20)
        self.assertEqual(validation_df["label"].value_counts().to_dict(), {"negative": 10, "positive": 10})
        self.assertEqual(train_df["label"].value_counts().to_dict(), {"negative": 40, "positive": 40})

    def test_same_seed_gives_same_split(self):
        df = make_df([(f"t{i}", "negative" if i % 2 else "positive") for i in range(40)])
        first, _ = split_train_validation(df, 0.25, seed=7)
        second, _ = split_train_validation(df, 0.25, seed=7)
        self.assertEqual(first["text"].tolist(), second["text"].tolist())


if __name__ == "__main__":
    unittest.main()
