"""Unit tests for the centralized logger (classifier.logging_setup).

No real LLM / network: exercises handler installation, idempotency, UTF-8
file output, and level wiring on a temporary log directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from classifier.config import settings
from classifier.logging_setup import _handlers, setup_logging


@pytest.fixture
def _clean_logging() -> None:
    """Reset root logger + idempotency state before and after each test.

    ``setup_logging`` mutates the process-global root logger. In the full
    suite, integration tests (``test_cli``) run first and install real
    handlers via ``cli.main``; this fixture must therefore tear them down
    BEFORE each test too, or idempotency would make ``setup_logging(tmp)`` a
    no-op. ``_handlers`` is the module's single source of truth, so clearing
    it (after removing the installed handlers) is enough.
    """
    _reset_installed_handlers()
    yield None
    _reset_installed_handlers()


def _reset_installed_handlers() -> None:
    """Remove every handler ``setup_logging`` installed and reset state."""
    root = logging.getLogger()
    for h in list(_handlers):
        if h in root.handlers:
            root.removeHandler(h)
        try:
            h.close()
        except (OSError, ValueError) as exc:  # stale/closed handler: ignore
            logging.getLogger(__name__).debug("Failed to close log handler: %s", exc)
    _handlers.clear()


def test_writes_info_to_utf8_log_file(tmp_path: Path, _clean_logging: None) -> None:
    setup_logging(log_dir=tmp_path)
    logging.getLogger("classifier.tests").info("診斷訊息-%d", 42)

    log_file = tmp_path / "pipeline.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "診斷訊息-42" in content
    assert "INFO" in content


def test_is_idempotent(tmp_path: Path, _clean_logging: None) -> None:
    setup_logging(log_dir=tmp_path)
    first_count = len(_handlers)
    setup_logging(log_dir=tmp_path)  # second call must be a no-op
    assert len(_handlers) == first_count == 2  # rotating-file + console


def test_force_reinstalls(tmp_path: Path, _clean_logging: None) -> None:
    setup_logging(log_dir=tmp_path)
    setup_logging(log_dir=tmp_path, force=True)
    assert len(_handlers) == 2


def test_uses_settings_log_level(tmp_path: Path, _clean_logging: None) -> None:
    setup_logging(log_dir=tmp_path)
    root = logging.getLogger()
    assert root.level == logging.getLevelName(settings.log_level.upper())
