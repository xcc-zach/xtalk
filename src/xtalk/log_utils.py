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


def _parse_log_level_config(config: str) -> tuple[int, dict[str, int]]:
    """Parse the package default and module-specific logging levels.

    Parameters
    ----------
    config
        Comma-separated logging directives. A directive without ``=`` sets
        the package default. A ``logger=level`` directive overrides the level
        for that logger and its descendants.

    Returns
    -------
    tuple[int, dict[str, int]]
        The package default logging level and valid module-level overrides.
    """
    default_level = _DEFAULT_LOG_LEVEL
    module_levels: dict[str, int] = {}

    for directive in config.split(","):
        directive = directive.strip()
        if not directive:
            continue

        logger_name, separator, level_name = directive.partition("=")
        if not separator:
            default_level = _LOG_LEVELS.get(
                logger_name.strip().upper(),
                default_level,
            )
            continue

        logger_name = logger_name.strip()
        level = _LOG_LEVELS.get(level_name.strip().upper())
        is_package_logger = (
            logger_name == _PACKAGE_LOGGER_NAME
            or logger_name.startswith(f"{_PACKAGE_LOGGER_NAME}.")
        )
        if is_package_logger and level is not None:
            module_levels[logger_name] = level

    return default_level, module_levels


def _initialize_package_logging() -> None:
    """Initialize the package logger from ``XTALK_LOG_LEVEL``.

    Notes
    -----
    A single level such as ``DEBUG`` sets the threshold for all ``xtalk.*``
    records. A comma-separated value can combine a package default with
    module-specific overrides, for example
    ``INFO,xtalk.serving.event_bus=DEBUG``. Applications remain responsible
    for configuring output handlers and formatters.
    """
    config = os.getenv("XTALK_LOG_LEVEL", "INFO")
    default_level, module_levels = _parse_log_level_config(config)

    package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    package_logger.setLevel(default_level)

    for logger_name, level in module_levels.items():
        logging.getLogger(logger_name).setLevel(level)

    if not any(
        isinstance(handler, logging.NullHandler)
        for handler in package_logger.handlers
    ):
        package_logger.addHandler(logging.NullHandler())
