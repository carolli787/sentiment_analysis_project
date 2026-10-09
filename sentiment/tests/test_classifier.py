import tempfile
import unittest
from pathlib import Path

from sentiment.classifier import Prediction, SentimentClassifier
from sentiment.tests.helpers import train_tiny_classifier
from sentiment.train import DEFAULT_MODEL_PATH


class SentimentClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = train_tiny_classifier()

    def test_evaluate_returns_expected_labels(self):
        self.assertEqual(self.classifier.evaluate("I love this, great").sentiment, "positive")
        self.assertEqual(self.classifier.evaluate("I hate this, awful").sentiment, "negative")

    def test_confidence_is_probability_of_returned_label(self):
        prediction = self.classifier.evaluate("love")
        self.assertIsInstance(prediction, Prediction)
        self.assertGreaterEqual(prediction.confidence, 0.5)
        self.assertLessEqual(prediction.confidence, 1.0)

    def test_text_with_no_known_words_still_gets_a_prediction(self):
        prediction = self.classifier.evaluate("@someone http://example.com")
        self.assertIn(prediction.sentiment, ("positive", "negative"))

    def test_predict_matches_evaluate(self):
        texts = ["love it", "hate it"]
        self.assertEqual(self.classifier.predict(texts), [self.classifier.evaluate(t) for t in texts])

    def test_evaluate_before_train_raises(self):
        with self.assertRaises(RuntimeError):
            SentimentClassifier().evaluate("hello")

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "model.joblib"
            self.classifier.save(path)
            loaded = SentimentClassifier.load(path)
        self.assertEqual(loaded.evaluate("great day"), self.classifier.evaluate("great day"))

    def test_load_missing_file_explains_how_to_train(self):
        with self.assertRaisesRegex(FileNotFoundError, "sentiment.train"):
            SentimentClassifier.load("/nonexistent/model.joblib")


@unittest.skipUnless(DEFAULT_MODEL_PATH.exists(), "no trained model; run `python -m sentiment.train`")
class TrainedModelAcceptanceTests(unittest.TestCase):
    """Acceptance checks from the spec, run against the real trained model."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = SentimentClassifier.load(DEFAULT_MODEL_PATH)

    def test_spec_example_sentences(self):
        self.assertEqual(self.classifier.evaluate("I love this!").sentiment, "positive")
        self.assertEqual(self.classifier.evaluate("This is the worst day ever").sentiment, "negative")


if __name__ == "__main__":
    unittest.main()
