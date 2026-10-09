from unittest import mock

from django.test import SimpleTestCase
from rest_framework.test import APIClient

from sentiment.tests.helpers import train_tiny_classifier
from sentiment_api.serializers import MAX_TEXT_LENGTH


class ApiTestCase(SimpleTestCase):
    """Runs the API against a tiny in-memory classifier, so no trained model file is needed."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.classifier = train_tiny_classifier()

    def setUp(self):
        patcher = mock.patch("sentiment_api.views.get_classifier", return_value=self.classifier)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = APIClient()

    def post_json(self, body):
        return self.client.post("/evaluate", body, format="json")


class EvaluateSuccessTests(ApiTestCase):
    def test_returns_sentiment_and_confidence(self):
        response = self.post_json({"text": "I love this, great day"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sentiment"], "positive")
        self.assertGreaterEqual(response.json()["confidence"], 0.5)
        self.assertEqual(set(response.json()), {"sentiment", "confidence"})

    def test_negative_text(self):
        response = self.post_json({"text": "I hate this, awful day"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sentiment"], "negative")

    def test_text_at_length_limit_is_accepted(self):
        self.assertEqual(self.post_json({"text": "a" * MAX_TEXT_LENGTH}).status_code, 200)


class EvaluateValidationTests(ApiTestCase):
    def assert_bad_request(self, response, field="text"):
        self.assertEqual(response.status_code, 400)
        self.assertIn(field, response.json())

    def test_missing_text(self):
        self.assert_bad_request(self.post_json({}))

    def test_empty_text(self):
        self.assert_bad_request(self.post_json({"text": ""}))

    def test_whitespace_only_text(self):
        self.assert_bad_request(self.post_json({"text": "   \n\t"}))

    def test_null_text(self):
        self.assert_bad_request(self.post_json({"text": None}))

    def test_list_text(self):
        self.assert_bad_request(self.post_json({"text": ["hello"]}))

    def test_text_over_length_limit(self):
        self.assert_bad_request(self.post_json({"text": "a" * (MAX_TEXT_LENGTH + 1)}))

    def test_body_is_not_an_object(self):
        self.assert_bad_request(self.post_json(["hello"]), field="non_field_errors")

    def test_invalid_json(self):
        response = self.client.post("/evaluate", "{not json", content_type="application/json")
        self.assert_bad_request(response, field="detail")


class EvaluateProtocolTests(ApiTestCase):
    def test_unknown_url_is_json_404(self):
        response = self.client.get("/nope")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Not found."})

    def test_non_json_content_type_is_415(self):
        response = self.client.post("/evaluate", "text=hello", content_type="application/x-www-form-urlencoded")
        self.assertEqual(response.status_code, 415)

    def test_get_is_405(self):
        response = self.client.get("/evaluate")
        self.assertEqual(response.status_code, 405)
        self.assertEqual(response["Allow"], "POST, OPTIONS")

    def test_unexpected_error_is_json_500_without_details(self):
        broken = mock.Mock()
        broken.evaluate.side_effect = ValueError("secret internal detail")
        with mock.patch("sentiment_api.views.get_classifier", return_value=broken), \
                self.assertLogs("sentiment_api.exceptions", level="ERROR"):
            response = self.post_json({"text": "hello"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Internal server error."})


class LoggingTests(ApiTestCase):
    def test_logs_status_and_label_but_never_the_text(self):
        with self.assertLogs("sentiment_api.middleware", level="INFO") as logs:
            self.post_json({"text": "my private love note"})
        line = logs.output[0]
        self.assertIn("status=200", line)
        self.assertIn("sentiment=positive", line)
        self.assertNotIn("private", line)


class HealthTests(ApiTestCase):
    def test_health(self):
        with mock.patch("sentiment_api.views.get_classifier", return_value=self.classifier):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
