"""Access to the trained classifier, loaded once per process."""

import logging
from functools import cache

from django.conf import settings

from sentiment import SentimentClassifier

logger = logging.getLogger(__name__)


@cache
def get_classifier() -> SentimentClassifier:
    """Return the trained classifier, loading it from SENTIMENT_MODEL_PATH on first use.

    Called at server startup (see config/wsgi.py), so requests never pay the load cost.
    """
    path = settings.SENTIMENT_MODEL_PATH
    logger.info("Loading sentiment model from %s", path)
    return SentimentClassifier.load(path)
