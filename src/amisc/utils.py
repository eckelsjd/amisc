"""Provides some basic utilities for the package.

Includes:

- `relative_error`: compute the relative L2 error between two vectors
- `get_logger`: logging utility with nice formatting
"""
import logging
import sys
from pathlib import Path

import numpy as np

LOG_FORMATTER = logging.Formatter(u"%(asctime)s — [%(levelname)s] — %(name)-25s — %(message)s")


def relative_error(pred, targ, axis=None):
    return np.sqrt(np.sum((pred - targ)**2, axis=axis) / np.sum(targ**2, axis=axis))


def get_logger(name: str, stdout=True, log_file: str | Path = None) -> logging.Logger:
    """Return a file/stdout logger with the given name.

    :param name: the name of the logger to return
    :param stdout: whether to add a stdout handler to the logger
    :param log_file: add file logging to this file (optional)
    :returns: the logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    if stdout:
        std_handler = logging.StreamHandler(sys.stdout)
        std_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(std_handler)
    if log_file is not None:
        f_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(f_handler)

    return logger
