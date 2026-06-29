"""Idempotent logger factory.

The old ``AppLogger`` added file+console handlers in ``__init__`` every time it
was constructed, so two instances meant duplicated log lines. Here handlers are
attached once per named logger.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "vtracker", *, log_file: str | None = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Idempotent: don't stack handlers on repeated calls.
    if logger.handlers:
        return logger
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.propagate = False
    return logger
