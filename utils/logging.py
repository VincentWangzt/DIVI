import logging
import os
import sys
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def get_logger(name: str = "sivi") -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    _LOGGER = logger
    return _LOGGER


def set_file_handler(log_dir: str, filename: str = "run.log") -> None:
    """Optionally attach a file handler lazily when a path is provided."""
    global _LOGGER
    logger = get_logger()
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, filename)
    # Avoid duplicate file handlers
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(
                h, 'baseFilename', None) == file_path:
            return
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            fmt=
            "%(asctime)s | %(levelname)s | %(name)s:%(filename)s:%(lineno)03d| %(funcName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    logger.addHandler(fh)


get_logger()  # Initialize the default logger at module load time
