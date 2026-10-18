"""Structured logging configuration (structlog)."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", *, json: bool = False) -> None:
    """Configure structlog and the stdlib root logger.

    Args:
        level: Log level name.
        json: Emit JSON lines instead of human-readable console output.
    """
    numeric = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, stream=sys.stderr, format="%(message)s")
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        # resolve sys.stderr at call time so redirected/closed streams are never cached
        logger_factory=lambda *_: structlog.PrintLogger(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def bind_run(run_id: str) -> None:
    """Bind a run id to the logging context of the current task/thread."""
    structlog.contextvars.bind_contextvars(run_id=run_id)


def clear_context() -> None:
    """Clear context variables bound for logging."""
    structlog.contextvars.clear_contextvars()
