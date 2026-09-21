"""One-click launcher for classify-twse-query.

Starts the FastAPI backend (port 8000) and the frontend static server
(port 8080) together, then opens the browser. Ctrl-C stops both.
"""

from __future__ import annotations

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import TextIO

from classifier.logging_setup import setup_logging

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
LOG_DIR = ROOT / "logs"

API_HOST = "127.0.0.1"
API_PORT = 8000
FRONTEND_PORT = 8080

# File handles backing children's redirected stdout/stderr. Kept at module
# scope so they survive for each child's lifetime (the ``Popen`` objects do
# not own them); closed in ``main``'s ``finally`` on shutdown.
_LOG_FILE_HANDLES: list[TextIO] = []


def _popen(args: list[str], log_file: Path | None = None) -> subprocess.Popen:
    """Spawn a child whose stdout/stderr go to ``log_file`` (never a pipe).

    Piping child output with nothing draining the pipe is a latent deadlock:
    once the OS pipe buffer fills (64 KiB on Windows), the child's
    ``stderr.write`` blocks forever, and because ``StreamHandler.emit`` holds
    the global logging lock, the whole pipeline freezes mid-query. Redirect
    to a durable log file instead; ``log_file=None`` means ``DEVNULL``.
    """
    output: TextIO | int = subprocess.DEVNULL
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        output = log_file.open("a", encoding="utf-8")
        _LOG_FILE_HANDLES.append(output)
    return subprocess.Popen(
        args,
        cwd=str(ROOT),
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> int:
    setup_logging()
    # Keep a reference to the handles so they are not garbage-collected
    # (closing the fd would sever the child's stdout/stderr mid-run).
    api_log = LOG_DIR / "api-server.log"
    frontend_log = LOG_DIR / "frontend-server.log"
    api = _popen([sys.executable, "-m", "classifier.api"], api_log)
    static = _popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(FRONTEND_PORT),
            "--bind",
            API_HOST,
            "--directory",
            str(FRONTEND_DIR),
        ],
        frontend_log,
    )

    print(f"API      -> http://{API_HOST}:{API_PORT}")
    print(f"Frontend -> http://{API_HOST}:{FRONTEND_PORT}")
    print(f"Child output -> {api_log.name} / {frontend_log.name} under {LOG_DIR}")
    print("正在啟動服務，稍候開啟瀏覽器...（按 Ctrl-C 結束）")

    time.sleep(3)
    try:
        webbrowser.open(f"http://{API_HOST}:{FRONTEND_PORT}")
    except Exception:  # noqa: BLE001 - headless box: never crash on browser open
        # Headless / no default browser: just print the URL instead of crashing.
        print(f"請手動開啟：http://{API_HOST}:{FRONTEND_PORT}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在關閉服務...")
    finally:
        for proc in (api, static):
            proc.terminate()
        for proc in (api, static):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        for handle in _LOG_FILE_HANDLES:
            handle.close()
        _LOG_FILE_HANDLES.clear()

    return 0


if __name__ == "__main__":
    sys.exit(main())
