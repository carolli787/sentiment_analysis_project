"""Root URL configuration."""

from django.urls import include, path

urlpatterns = [
    path("", include("sentiment_api.urls")),
]

# Unknown URLs get a JSON 404 instead of an HTML page (only applies when DEBUG is off).
handler404 = "sentiment_api.exceptions.not_found"
