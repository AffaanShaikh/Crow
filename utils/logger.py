"""
uses structlog for JSON output in production, pretty console output in development
"""

import logging
import os
import sys
from typing import Any
import structlog 

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    """
    configures structlog + stdlib logging
    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        json_logs: If True, emit JSON lines (good for log aggregators)
                   If False, emit human-readable coloured output
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,                                         # [light green, for 'info'] 
        structlog.stdlib.add_logger_name,                                       # [dark blue, file name]
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),#(fmt="iso"), # grey, time
        structlog.processors.StackInfoRenderer(),                               # attaches error stack trace if true
        structlog.processors.format_exc_info,                                   # ensures proper stack trace in JSON logs
    ] 

    file_renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors
        + [ # 'ProcessorFormatter.wrap_for_formatter' tells structlog not to render the event yet, let Python logging handlers render it
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # formatters (console formatter later)
    file_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            file_renderer,
        ],
    )
 
    # logs everything to 2 places during development: stdout (console) & logs/app.log
    # logs everything to 1 place during deployment:  json logs in logs/app.log 

    # in deployment and dev., json logs in app.log
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"))
    file_handler.setFormatter(file_formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [file_handler]

    if not json_logs: # i.e. in dev., app.log and console
        console_renderer = structlog.dev.ConsoleRenderer(colors=True)
        console_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                console_renderer,
            ],
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        root_logger.handlers.insert(0, console_handler)

    root_logger.setLevel(log_level.upper())

    # we silence the noisy third-party loggers
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """return a bound structlog logger for the given module name"""
    return structlog.get_logger(name)