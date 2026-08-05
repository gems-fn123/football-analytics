"""Logging setup. Rich if available, stdlib otherwise."""

from __future__ import annotations

import logging
import os


def get_logger(name: str = "footy") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = os.getenv("FOOTY_LOG_LEVEL", "INFO").upper()
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
