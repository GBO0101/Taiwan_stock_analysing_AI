"""FastAPI HTTP API for the classify-twse-query pipeline.

Endpoints:
  POST /pipeline  - run the strict sequential pipeline, return JSON trace
  GET  /pipeline/progress/{client_id} - poll live progress of a running pipeline
  POST /chart     - render a chart, return raw image/png bytes
  GET  /health    - service health (no external calls)

Run with:  python -m classifier.api
"""

from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from classifier.chart_renderer import ChartRenderer, ChartRenderError
from classifier.models import ChartRequest
from classifier.pipeline import Pipeline, PipelineError

app = FastAPI(title="classify-twse-query", version="0.1.0")

# Allow the static frontend (e.g. http://127.0.0.1:8080) to call this API
# cross-origin. In production, restrict allow_origins to your known frontend host.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Live progress registry, keyed by an opaque client-generated id. Entries are
# pruned on write (TTL 5 min, capped at 50) so the dict never grows unbounded.
_PROGRESS_TTL_SECONDS = 300.0
_PROGRESS_MAX_ENTRIES = 50
_progress: dict[str, dict[str, Any]] = {}
_progress_lock = threading.Lock()


def _prune_progress(now: float) -> None:
    """Drop expired entries and cap the registry size (call under the lock)."""
    for cid in [
        cid for cid, e in _progress.items() if now - e["updated_at"] > _PROGRESS_TTL_SECONDS
    ]:
        _progress.pop(cid, None)
    while len(_progress) > _PROGRESS_MAX_ENTRIES:
        _progress.pop(next(iter(_progress)), None)


def _set_progress(client_id: str, status: str, step: str, message: str) -> None:
    """Register or update a progress entry; the first call seeds timestamps."""
    with _progress_lock:
        now = time.monotonic()
        entry = _progress.setdefault(client_id, {})
        entry["status"] = status
        entry["step"] = step
        entry["message"] = message
        entry.setdefault("started_at", now)
        entry["updated_at"] = now
        _prune_progress(now)


def _error_response(status_code: int, message: str) -> JSONResponse:
    """Return a structured error body {"error": message} at the top level."""
    return JSONResponse(status_code=status_code, content={"error": message})


@app.post("/pipeline")
def pipeline_endpoint(payload: dict) -> dict:
    """Run the full pipeline on a question and return the step trace.

    Request body: {"question": "台積電未來展望", "client_id": "<optional>"}
    When a client_id is supplied, live progress is tracked and can be polled
    via GET /pipeline/progress/{client_id} until completion + TTL.
    """
    question = payload.get("question")
    if not question or not isinstance(question, str):
        return _error_response(422, "field 'question' (str) is required")

    raw_client_id = payload.get("client_id")
    client_id = raw_client_id if isinstance(raw_client_id, str) and raw_client_id else None
    if client_id is not None:
        _set_progress(client_id, "running", "boundary", "Step 1/4：解析問題範圍")

    def _report(step: str, message: str) -> None:
        if client_id is not None:
            _set_progress(client_id, "running", step, message)

    try:
        result = Pipeline().run(question=question, progress=_report if client_id else None)
    except PipelineError as e:
        if client_id is not None:
            _set_progress(client_id, "failed", "failed", str(e))
        return _error_response(500, str(e))

    if client_id is not None:
        _set_progress(client_id, "done", "done", "完成")
    return result.model_dump()


@app.get("/pipeline/progress/{client_id}")
def progress_endpoint(client_id: str) -> dict:
    """Return the live progress snapshot for a pipeline run.

    - unknown id: {"status": "unknown"}
    - running: {"status", "step", "message", "elapsed_sec"}
    - done/failed: same shape (elapsed_sec included)
    """
    with _progress_lock:
        entry = _progress.get(client_id)
        if entry is None:
            return {"status": "unknown"}
        elapsed = round(entry["updated_at"] - entry["started_at"], 1)
        snapshot: dict[str, Any] = {
            "status": entry["status"],
            "step": entry.get("step", ""),
            "message": entry.get("message", ""),
            "elapsed_sec": elapsed,
        }
        return snapshot


@app.post("/chart")
def chart_endpoint(request: ChartRequest) -> Response:
    """Render a chart and return raw PNG bytes.

    Request body is a ChartRequest JSON. No session_id required.
    """
    try:
        renderer = ChartRenderer()
        path = renderer.render(request)
    except ChartRenderError as e:
        return _error_response(500, str(e))

    with open(path, "rb") as f:
        png_bytes = f.read()

    return Response(content=png_bytes, media_type="image/png")


@app.get("/health")
def health_endpoint() -> dict:
    """Return service-health information without invoking OpenAI or FinMind."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    from classifier.logging_setup import setup_logging

    setup_logging()
    uvicorn.run(app, host="127.0.0.1", port=8000)
