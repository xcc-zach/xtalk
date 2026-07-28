"""Internal logging initialization for the Xtalk package."""

import logging
import os

_DEFAULT_LOG_LEVEL = logging.INFO
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
_PACKAGE_LOGGER_NAME = "xtalk"


def _initialize_package_logging() -> None:
    """Initialize the package logger from ``XTALK_LOG_LEVEL``.

    Notes
    -----
    The package logger only defines the threshold for ``xtalk.*`` records.
    Applications remain responsible for configuring output handlers and
    formatters.
    """
    level_name = os.getenv("XTALK_LOG_LEVEL", "INFO").strip().upper()
    package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    package_logger.setLevel(_LOG_LEVELS.get(level_name, _DEFAULT_LOG_LEVEL))

    if not any(
        isinstance(handler, logging.NullHandler)
        for handler in package_logger.handlers
    ):
        package_logger.addHandler(logging.NullHandler())
