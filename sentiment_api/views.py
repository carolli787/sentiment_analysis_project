"""HTTP endpoints. Views only handle HTTP concerns; all ML logic lives in the `sentiment` package."""

from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from sentiment_api.serializers import EvaluateRequestSerializer, EvaluateResponseSerializer
from sentiment_api.services import get_classifier


class EvaluateView(APIView):
    """POST /evaluate: return the sentiment of one sentence."""

    def post(self, request: Request) -> Response:
        request_serializer = EvaluateRequestSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)  # invalid input -> 400

        prediction = get_classifier().evaluate(request_serializer.validated_data["text"])

        response_serializer = EvaluateResponseSerializer({
            "sentiment": prediction.sentiment,
            "confidence": round(prediction.confidence, 4),
        })
        return Response(response_serializer.data)


class HealthView(APIView):
    """GET /health: report that the server is up and the model is loaded."""

    def get(self, request: Request) -> Response:
        get_classifier()
        return Response({"status": "ok"})
