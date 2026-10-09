"""Request and response shapes for the sentiment API."""

from rest_framework import serializers

from sentiment.data import BINARY_LABELS

MAX_TEXT_LENGTH = 1000


class EvaluateRequestSerializer(serializers.Serializer):
    """Body of POST /evaluate.

    CharField trims surrounding whitespace and then rejects blank text, so
    empty and whitespace-only input both fail with 400.
    """

    text = serializers.CharField(max_length=MAX_TEXT_LENGTH)


class EvaluateResponseSerializer(serializers.Serializer):
    """Body of a successful POST /evaluate response."""

    sentiment = serializers.ChoiceField(choices=BINARY_LABELS)
    confidence = serializers.FloatField(min_value=0.5, max_value=1.0)
