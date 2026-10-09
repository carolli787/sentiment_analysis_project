"""WSGI entry point, used by `manage.py runserver` and production servers such as gunicorn.

The model is loaded here, once, when the server starts. A missing or broken
model file therefore stops the server at startup with a clear error, instead
of failing on the first request. (Loading it in AppConfig.ready() would also
run for every management command, including `manage.py test`, which doesn't
need a trained model.)
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

application = get_wsgi_application()

from sentiment_api.services import get_classifier  # noqa: E402  (needs Django set up first)

get_classifier()
