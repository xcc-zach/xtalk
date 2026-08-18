"""Tests for Xtalk package logging configuration."""

from __future__ import annotations

import logging
import os
import unittest
from unittest.mock import patch

from xtalk.log_utils import _initialize_package_logging


class PackageLoggingTests(unittest.TestCase):
    """Verify package and module-specific logging levels."""

    _LOGGER_NAMES = (
        "xtalk",
        "xtalk.serving",
        "xtalk.serving.event_bus",
        "xtalk.serving.event_bus.publisher",
        "xtalk.models",
        "external.library",
    )

    def setUp(self) -> None:
        """Save logger state and clear explicit levels before each test."""
        self._original_levels = {
            name: logging.getLogger(name).level for name in self._LOGGER_NAMES
        }
        for name in self._LOGGER_NAMES:
            logging.getLogger(name).setLevel(logging.NOTSET)

    def tearDown(self) -> None:
        """Restore logger levels changed by a test."""
        for name, level in self._original_levels.items():
            logging.getLogger(name).setLevel(level)

    def _initialize(self, config: str) -> None:
        """Initialize package logging with one temporary environment value."""
        with patch.dict(os.environ, {"XTALK_LOG_LEVEL": config}):
            _initialize_package_logging()

    def test_single_level_remains_supported(self) -> None:
        """Apply the legacy single-level syntax to the full package."""
        self._initialize("DEBUG")

        self.assertEqual(logging.getLogger("xtalk").level, logging.DEBUG)
        self.assertEqual(
            logging.getLogger("xtalk.serving.event_bus").getEffectiveLevel(),
            logging.DEBUG,
        )

    def test_module_override_sets_only_the_selected_subtree(self) -> None:
        """Apply an override to its logger and descendants only."""
        self._initialize("INFO,xtalk.serving.event_bus=DEBUG")

        self.assertEqual(logging.getLogger("xtalk").level, logging.INFO)
        self.assertEqual(
            logging.getLogger("xtalk.serving").getEffectiveLevel(),
            logging.INFO,
        )
        self.assertEqual(
            logging.getLogger("xtalk.serving.event_bus").level,
            logging.DEBUG,
        )
        self.assertEqual(
            logging.getLogger(
                "xtalk.serving.event_bus.publisher"
            ).getEffectiveLevel(),
            logging.DEBUG,
        )

    def test_multiple_overrides_and_whitespace_are_supported(self) -> None:
        """Parse multiple directives independently of surrounding whitespace."""
        self._initialize(
            " WARNING , xtalk.serving.event_bus = debug , xtalk.models = ERROR "
        )

        self.assertEqual(logging.getLogger("xtalk").level, logging.WARNING)
        self.assertEqual(
            logging.getLogger("xtalk.serving.event_bus").level,
            logging.DEBUG,
        )
        self.assertEqual(logging.getLogger("xtalk.models").level, logging.ERROR)

    def test_module_only_config_uses_info_as_the_package_default(self) -> None:
        """Keep the established INFO default when no default is specified."""
        self._initialize("xtalk.serving.event_bus=DEBUG")

        self.assertEqual(logging.getLogger("xtalk").level, logging.INFO)
        self.assertEqual(
            logging.getLogger("xtalk.serving.event_bus").level,
            logging.DEBUG,
        )

    def test_invalid_and_external_directives_are_ignored(self) -> None:
        """Ignore malformed levels and logger names outside the package."""
        self._initialize(
            "INVALID,xtalk.serving=VERBOSE,external.library=DEBUG,=ERROR"
        )

        self.assertEqual(logging.getLogger("xtalk").level, logging.INFO)
        self.assertEqual(logging.getLogger("xtalk.serving").level, logging.NOTSET)
        self.assertEqual(
            logging.getLogger("external.library").level,
            logging.NOTSET,
        )

    def test_last_valid_directive_for_a_logger_wins(self) -> None:
        """Use the last valid override when a logger appears repeatedly."""
        self._initialize(
            "INFO,xtalk.serving.event_bus=WARNING,"
            "xtalk.serving.event_bus=DEBUG"
        )

        self.assertEqual(
            logging.getLogger("xtalk.serving.event_bus").level,
            logging.DEBUG,
        )
