import logging
import sys


def setup_logging() -> None:
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    loggers = ["", "uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "starlette"]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
