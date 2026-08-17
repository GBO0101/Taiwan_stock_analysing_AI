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

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"

API_HOST = "127.0.0.1"
API_PORT = 8000
FRONTEND_PORT = 8080


def _popen(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        args,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> int:
    api = _popen([sys.executable, "-m", "classifier.api"])
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
        ]
    )

    print(f"API      -> http://{API_HOST}:{API_PORT}")
    print(f"Frontend -> http://{API_HOST}:{FRONTEND_PORT}")
    print("正在啟動服務，稍候開啟瀏覽器...（按 Ctrl-C 結束）")

    time.sleep(3)
    try:
        webbrowser.open(f"http://{API_HOST}:{FRONTEND_PORT}")
    except Exception:
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
