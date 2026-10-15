"""ASGI entrypoint: `uvicorn gateway.main:app --port 8080`."""

import logging

from gateway.app import create_app
from gateway.config import Settings

logging.basicConfig(level=Settings().log_level.upper(),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = create_app()
