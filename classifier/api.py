"""FastAPI HTTP API for the classify-twse-query pipeline.

Endpoints:
  POST /pipeline  - run the strict sequential pipeline, return JSON trace
  POST /chart     - render a chart, return raw image/png bytes
  GET  /health    - service health (no external calls)

Run with:  python -m classifier.api
"""

from __future__ import annotations

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


def _error_response(status_code: int, message: str) -> JSONResponse:
    """Return a structured error body {"error": message} at the top level."""
    return JSONResponse(status_code=status_code, content={"error": message})


@app.post("/pipeline")
def pipeline_endpoint(payload: dict) -> dict:
    """Run the full pipeline on a question and return the step trace.

    Request body: {"question": "台積電未來展望"}
    """
    question = payload.get("question")
    if not question or not isinstance(question, str):
        return _error_response(422, "field 'question' (str) is required")

    try:
        result = Pipeline().run(question=question)
    except PipelineError as e:
        return _error_response(500, str(e))

    return result.model_dump()


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

    uvicorn.run(app, host="127.0.0.1", port=8000)
