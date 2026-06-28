"""Structured logging utilities for the application.

Provides a small JSON formatter and helpers to configure and obtain
loggers consistently across the codebase. Uses `%s`-style logging
semantics via the standard library and emits JSON to stdout.

Follow the project's logging rules: structured output, no prints,
and use `logger.exception()` to attach stack traces when catching.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit logs as compact JSON objects.

    The formatter includes a UTC ISO-8601 timestamp, level, logger
    name and the formatted message. Any non-standard LogRecord
    attributes passed via ``extra=...`` are included when serialisable.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # include any extra fields provided via `extra=` (skip standard attrs)
        standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }

        for key, value in record.__dict__.items():
            if key in standard_attrs:
                continue
            try:
                # ensure value is JSON serialisable, fall back to repr()
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = repr(value)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger with a single stream handler.

    The handler is added only once (idempotent — avoids duplicate log
    lines), but the level is always (re)applied so callers can raise or
    lower verbosity after the first call.
    """

    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``.

    Ensures logging is configured, but never overrides a level already
    set by an explicit `configure_logging(level)` call.
    """

    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)


__all__ = ["get_logger", "configure_logging", "JsonFormatter"]
