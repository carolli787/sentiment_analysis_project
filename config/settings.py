"""Django settings for the sentiment API.

Values that differ between environments are read from environment variables.
The app needs no database: it only serves predictions from a model file.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# The fallback key is only for local development. Set DJANGO_SECRET_KEY anywhere else.
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "django-insecure-local-development-only")
# Off by default so error responses never include stack traces.
DEBUG = os.environ.get("DJANGO_DEBUG", "false").lower() == "true"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = [
    "rest_framework",
    "sentiment_api",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "sentiment_api.middleware.RequestLoggingMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

DATABASES: dict = {}

USE_TZ = True

REST_FRAMEWORK = {
    # JSON in, JSON out. Other request content types get 415 Unsupported Media Type.
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    # The API is public and has no users, so authentication is switched off entirely.
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "UNAUTHENTICATED_USER": None,
    "EXCEPTION_HANDLER": "sentiment_api.exceptions.exception_handler",
}

SENTIMENT_MODEL_PATH = Path(
    os.environ.get("SENTIMENT_MODEL_PATH", BASE_DIR / "models" / "sentiment_model.joblib")
)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"simple": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
    "loggers": {"sentiment_api": {"handlers": ["console"], "level": "INFO"}},
}
