# Copyright (c) Sony AI Inc.
# All rights reserved.

import logging
import os
import time
from datetime import timedelta


class LogFormatter(logging.Formatter):
    """Custom logging formatter that formats log records in a specific way."""

    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def format(self, record):
        """Formats the specified log record and returns the resulting string.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log record as a string.
        """
        elapsed_seconds = round(record.created - self.start_time)
        prefix = (
            f"{record.levelname} - {time.strftime('%x %X')} - "
            f"{timedelta(seconds=elapsed_seconds)}"
        )
        message = record.getMessage()
        message = message.replace("\n", f"\n{' ' * (len(prefix) + 3)}")
        return f"{prefix} - {message}"

    def reset_time(self):
        """Reset the start time for elapsed time calculation."""
        self.start_time = time.time()


def create_logger(filepath: str) -> logging.Logger:
    """Create a logger for monitoring training.

    Args:
        filepath (str): Filepath to write the log file to.

    Returns:
        logging.Logger: A logger instance with the specified settings.
    """
    log_formatter = LogFormatter()

    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

    # Create file handler and set level to debug.
    file_handler = logging.FileHandler(filepath, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # Create console handler and set level to info.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # Create logger and set level to debug.
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    log_formatter.reset_time()
    return logger
