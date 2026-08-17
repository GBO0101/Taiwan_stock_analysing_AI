"""HTTP API integration tests using FastAPI TestClient (no real network calls)."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from classifier.llm_client import LLMError
from classifier.models import (
    BoundaryResult,
    ChartRequest,
    ChartDataRequirement,
    ChartType,
    ClassificationResult,
    ClassificationType,
    DecompositionResult,
    SubQuery,
)
from classifier.pipeline import Pipeline, PipelineError
from classifier import api as api_module


def _fake_llm(needs_viz=False, chart_req=None, chart_type=None, ctype=ClassificationType.ANALYTICAL):
    fake = MagicMock()
    boundary = BoundaryResult(
        stock_codes=["2330"], company_names=["台積電"], confidence=0.9
    )
    classification = ClassificationResult(
        type=ctype,
        confidence=0.9,
        needs_visualization=needs_viz,
        chart_data_requirements=chart_req,
        chart_type=chart_type,
    )
    decomposition = DecompositionResult(
        sub_queries=[SubQuery(id="q0", operation="finmind_query", datasets=["TaiwanStockPrice"])]
    )

    def extract(prompt, response_model):
        return {
            "BoundaryResult": boundary,
            "ClassificationResult": classification,
            "DecompositionResult": decomposition,
        }[response_model.__name__]

    fake.extract_structured.side_effect = extract
    return fake


@pytest.fixture
def client():
    return TestClient(api_module.app)


def test_health_no_external_calls(client):
    with patch("classifier.api.Pipeline") as p:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        p.assert_not_called()


def test_pipeline_endpoint_returns_trace(client):
    fake = _fake_llm()
    with patch("classifier.api.Pipeline", return_value=MagicMock(run=lambda question: PipelineResult_for_test(fake))):
        # Build a real PipelineResult via the actual Pipeline with the fake client.
        from classifier.pipeline import Pipeline

        real = Pipeline(llm_client=fake)
        with patch("classifier.api.Pipeline", return_value=real):
            resp = client.post("/pipeline", json={"question": "台積電未來展望"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["question"] == "台積電未來展望"
        steps = [s["step"] for s in data["steps"]]
        assert steps == ["boundary", "classification", "decomposition"]


def test_pipeline_endpoint_missing_question(client):
    resp = client.post("/pipeline", json={})
    assert resp.status_code == 422


def test_pipeline_endpoint_pipeline_error(client):
    from classifier.pipeline import PipelineError

    fake = _fake_llm()
    fake.extract_structured.side_effect = LLMError("boom")
    real = Pipeline(llm_client=fake)
    with patch("classifier.api.Pipeline", return_value=real):
        resp = client.post("/pipeline", json={"question": "台積電"})
    assert resp.status_code == 500
    assert "error" in resp.json()


def test_chart_endpoint_returns_png(client):
    fake_finmind = MagicMock()
    fake_finmind.get_stock_price.return_value = [
        {"date": "2026-01-01", "open": 10, "close": 11, "high": 12, "low": 9},
        {"date": "2026-01-02", "open": 11, "close": 12, "high": 13, "low": 10},
    ]
    request = ChartRequest(
        stock_codes=["2330"],
        chart_data_requirements=ChartDataRequirement.PRICE_TREND,
        chart_type=ChartType.LINE,
    )
    with patch("classifier.chart_renderer.FreeDataFetcher", return_value=fake_finmind):
        resp = client.post("/chart", json=request.model_dump())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")
    assert len(resp.content) > 0


def test_chart_endpoint_invalid_body(client):
    resp = client.post("/chart", json={"stock_codes": ["2330"]})  # missing required fields
    assert resp.status_code == 422


def PipelineResult_for_test(fake):  # pragma: no cover - helper placeholder
    from classifier.pipeline import Pipeline

    return Pipeline(llm_client=fake).run("台積電未來展望")
