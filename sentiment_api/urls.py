from django.urls import path

from sentiment_api.views import EvaluateView, HealthView

urlpatterns = [
    # The assignment names this endpoint /evaluate. A strictly noun-based design
    # would call it something like /sentiments; see the README.
    path("evaluate", EvaluateView.as_view(), name="evaluate"),
    path("health", HealthView.as_view(), name="health"),
]
