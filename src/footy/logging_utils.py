"""Logging setup. Rich if available, stdlib otherwise."""

from __future__ import annotations

import logging
import os

ROOT = "footy"


def get_logger(name: str = ROOT) -> logging.Logger:
    """Return a logger with exactly one handler in the tree.

    The handler is attached only to the `footy` root. Children such as
    `footy.pipeline` propagate up to it. Attaching to each child as well would
    print every line once per level of the hierarchy.
    """
    root = logging.getLogger(ROOT)
    if not root.handlers:
        level = os.getenv("FOOTY_LOG_LEVEL", "INFO").upper()
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
            fmt = "%(message)s"
        except ImportError:
            handler = logging.StreamHandler()
            fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
        root.setLevel(level)

    return root if name == ROOT else logging.getLogger(name)
