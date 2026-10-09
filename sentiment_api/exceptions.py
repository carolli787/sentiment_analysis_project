"""API-wide error handling."""

import logging

from django.http import HttpRequest, JsonResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

logger = logging.getLogger(__name__)


def exception_handler(exc: Exception, context: dict) -> Response:
    """Use DRF's handling for API errors, and return a JSON 500 for anything unexpected.

    Without this, unexpected errors would produce Django's HTML error page.
    The details go to the log, never to the client.
    """
    response = drf_exception_handler(exc, context)
    if response is not None:
        return response
    logger.exception("Unhandled error in %s", context["view"].__class__.__name__)
    return Response({"detail": "Internal server error."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def not_found(request: HttpRequest, exception: Exception) -> JsonResponse:
    """JSON 404 for unknown URLs (Django's default is an HTML page)."""
    return JsonResponse({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
