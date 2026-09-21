"""Centralized logging configuration for the classify-twse-query pipeline.

Before this module the project had no durable logs: a hung supply-chain query
(wrong stock codes from the LLM fallback triggering minutes-long annual-report
fetches) left no trace. ``setup_logging()`` installs a rotating file handler
(``logs/pipeline.log``, UTF-8) plus a console handler, honoring
``settings.log_level``, and is idempotent so every entry point (CLI, API,
one-click launcher) can call it safely.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from classifier.config import settings

logger = logging.getLogger(__name__)

_LOG_DIR_NAME = "logs"
_LOG_FILE_NAME = "pipeline.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB per file, then rotate.
_BACKUP_COUNT = 3
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Handlers installed by setup_logging; the idempotency guard, the "force" path,
# and the tests all rely on this single source of truth.
_handlers: list[logging.Handler] = []


def setup_logging(log_dir: Path | None = None, *, force: bool = False) -> logging.Logger:
    """Install the project's handlers on the root logger (idempotent).

    The first call adds a ``RotatingFileHandler`` (UTF-8, ``logs/pipeline.log``
    under the project root unless ``log_dir`` overrides it) and a
    ``StreamHandler`` (stderr, so CLI JSON on stdout stays clean), both at
    ``settings.log_level``. Later calls are no-ops unless ``force=True``,
    which first removes and closes the previously installed handlers.

    Args:
        log_dir: Directory for the log file (defaults to ``<project>/logs``).
        force: Reinstall handlers even if already configured.

    Returns:
        The root logger, for convenience.
    """
    # Only mutate the module-level ``_handlers`` list (extend/clear), never
    # rebind it, so no ``global`` declaration is needed.
    root = logging.getLogger()

    if _handlers and not force:
        return root

    for h in _handlers:
        if h in root.handlers:
            root.removeHandler(h)
        try:
            h.close()
        except (OSError, ValueError) as exc:
            # A stale handler failing to close must never abort setup.
            logger.debug("Failed to close stale log handler: %s", exc)
    _handlers.clear()

    base_dir = log_dir or (Path(__file__).resolve().parent.parent / _LOG_DIR_NAME)
    base_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = RotatingFileHandler(
        base_dir / _LOG_FILE_NAME,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()  # defaults to sys.stderr
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    _handlers.extend([file_handler, console_handler])
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    return root
