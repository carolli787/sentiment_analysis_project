"""Request logging."""

import logging
import time
from collections.abc import Callable

from django.http import HttpRequest, HttpResponse

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware:
    """Log the method, path, status code, latency and predicted sentiment of every request.

    The request body is never logged: tweets can contain personal information.
    """

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        started = time.perf_counter()
        response = self.get_response(request)
        latency_ms = (time.perf_counter() - started) * 1000

        data = getattr(response, "data", None)  # set on DRF responses
        sentiment = data.get("sentiment") if isinstance(data, dict) else None
        logger.info(
            "%s %s status=%d latency_ms=%.1f sentiment=%s",
            request.method,
            request.path,
            response.status_code,
            latency_ms,
            sentiment or "-",
        )
        return response
