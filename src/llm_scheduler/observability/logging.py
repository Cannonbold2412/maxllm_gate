"""Structured logging configuration."""

import logging
import sys

import structlog


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    
    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


def get_logger() -> structlog.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger()


class RequestLogger:
    """Context manager for request logging."""
    
    def __init__(self, request_id: str, **context):
        self.request_id = request_id
        self.context = context
        self.logger = get_logger()
    
    def __enter__(self):
        structlog.contextvars.bind_contextvars(
            request_id=self.request_id,
            **self.context,
        )
        return self.logger
    
    def __exit__(self, *args):
        structlog.contextvars.unbind_contextvars(
            "request_id",
            *self.context.keys(),
        )
