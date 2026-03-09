"""
uses structlog for JSON output in production, pretty console output in development
"""

import logging
import sys
from typing import Any
import structlog


def setup_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    """
    configures structlog + stdlib logging
    args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        json_logs: If True, emit JSON lines (good for log aggregators)
                   If False, emit human-readable coloured output
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        # i.e. in production: machine-readable JSON lines
        renderer = structlog.processors.JSONRenderer()
    else:
        # else in development: coloured key=value output coz we likey farbe
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level.upper())

    # we silence the noisy third-party loggers
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """return a bound structlog logger for the given module name"""
    return structlog.get_logger(name)
