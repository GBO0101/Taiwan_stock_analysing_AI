"""Regression tests for the one-click launcher (``start.py``).

These cover the pipe-deadlock bug: ``_popen`` used ``subprocess.PIPE`` for the
child's stdout/stderr but nothing ever drained that pipe. Once the OS pipe
buffer filled (~64 KiB on Windows), the child's ``stderr.write`` blocked
forever — and since ``StreamHandler.emit`` holds the global logging lock, the
whole API pipeline froze mid-query. The children must be redirected to real
files instead, so their writes can never block a caller.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

import start as start_mod


def test_popen_redirects_child_output_to_file_not_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_popen`` must pass a writable file as ``stdout``, never ``PIPE``.

    A pipe that nobody reads deadlocks the child once the buffer fills; the
    regression froze a live annual-report extraction for 25+ minutes. The
    child's stderr must be merged into the same file.
    """
    captured: dict[str, object] = {}

    class _FakePopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    log_file = tmp_path / "api-server.log"
    start_mod._popen([sys.executable, "-m", "classifier.api"], log_file=log_file)

    kwargs = captured["kwargs"]
    assert kwargs["stdout"] is not subprocess.PIPE, (
        "undrained PIPE blocks the child forever when the buffer fills"
    )
    stdout = kwargs["stdout"]
    assert stdout is not None and hasattr(stdout, "write"), (
        "stdout must be a real, writable file handle"
    )
    assert kwargs["stderr"] == subprocess.STDOUT, (
        "child stderr must merge into the same redirect file"
    )


def test_popen_child_output_lands_in_log_file(tmp_path: Path) -> None:
    """End-to-end: a spawned child's stdout actually reaches the log file.

    Guards against a future change silently reverting to a blocking sink
    (pipe/blocking socket) that the mock-based test above would not catch.
    """
    log_file = tmp_path / "probe.log"
    proc = start_mod._popen(
        [sys.executable, "-c", "print('launcher-write-probe')"],
        log_file=log_file,
    )
    try:
        assert proc.wait(timeout=30) == 0
    except subprocess.TimeoutExpired:  # pragma: no cover - cleanup guard
        proc.kill()
        pytest.fail("probe child did not exit in time")
    for _ in range(50):  # file flush window
        if log_file.exists() and log_file.read_text(encoding="utf-8").strip():
            break
        time.sleep(0.1)
    assert log_file.read_text(encoding="utf-8").strip() == "launcher-write-probe", (
        "child output must be captured in the redirect file"
    )
