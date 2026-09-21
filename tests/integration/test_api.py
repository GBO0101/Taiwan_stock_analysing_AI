"""HTTP API integration tests using FastAPI TestClient (no real network calls)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from classifier import api as api_module
from classifier.annual_report import AnnualReportError
from classifier.llm_client import LLMError
from classifier.models import (
    BoundaryResult,
    ChartDataRequirement,
    ChartRequest,
    ChartType,
    ClassificationResult,
    ClassificationType,
    DecompositionResult,
    SubQuery,
)
from classifier.pipeline import Pipeline
from classifier.stock_enumeration import _RangeValidationModel, _SupplyChainModel


def _fake_llm(
    needs_viz=False, chart_req=None, chart_type=None, ctype=ClassificationType.ANALYTICAL
):
    fake = MagicMock()
    boundary = BoundaryResult(stock_codes=["2330"], company_names=["台積電"], confidence=0.9)
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
    supply_chain = _SupplyChainModel(upstream=[], downstream=[], confidence=0.3)
    range_validation = _RangeValidationModel(confidence=0.6)

    def extract(prompt, response_model):
        return {
            "BoundaryResult": boundary,
            "ClassificationResult": classification,
            "DecompositionResult": decomposition,
            "_SupplyChainModel": supply_chain,
            "_RangeValidationModel": range_validation,
        }[response_model.__name__]

    fake.extract_structured.side_effect = extract
    return fake


@pytest.fixture(autouse=True)
def _no_annual_report_network():
    """Keep Step 4 offline: the annual-report path always falls back to LLM.

    Step 4 now tries the target's annual report (doc.twse.com.tw) first, which
    would hit the real network in tests. Force AnnualReportError so the fake
    LLM's _SupplyChainModel is used instead.
    """
    with patch(
        "classifier.stock_enumeration.extract_annual_report",
        side_effect=AnnualReportError("no network in tests"),
    ):
        yield


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
    with patch(
        "classifier.api.Pipeline",
        return_value=MagicMock(run=lambda question: PipelineResult_for_test(fake)),
    ):
        # Build a real PipelineResult via the actual Pipeline with the fake client.
        from classifier.pipeline import Pipeline

        real = Pipeline(llm_client=fake)
        with patch("classifier.api.Pipeline", return_value=real):
            resp = client.post("/pipeline", json={"question": "台積電未來展望"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["question"] == "台積電未來展望"
        steps = [s["step"] for s in data["steps"]]
        assert steps == ["boundary", "classification", "decomposition", "stock_enumeration"]


def test_pipeline_endpoint_missing_question(client):
    resp = client.post("/pipeline", json={})
    assert resp.status_code == 422


def test_pipeline_endpoint_pipeline_error(client):

    fake = _fake_llm()
    fake.extract_structured.side_effect = LLMError("boom")
    real = Pipeline(llm_client=fake)
    with patch("classifier.api.Pipeline", return_value=real):
        resp = client.post("/pipeline", json={"question": "台積電"})
    assert resp.status_code == 500
    assert "error" in resp.json()


@pytest.fixture(autouse=True)
def _clear_progress_registry():
    """Reset the module-level progress registry between tests."""
    yield
    with api_module._progress_lock:
        api_module._progress.clear()


def test_pipeline_progress_done_after_post(client):
    fake = _fake_llm()
    real = Pipeline(llm_client=fake)
    with patch("classifier.api.Pipeline", return_value=real):
        resp = client.post(
            "/pipeline",
            json={"question": "台積電未來展望", "client_id": "cid-done"},
        )
        assert resp.status_code == 200

    prog = client.get("/pipeline/progress/cid-done")
    assert prog.status_code == 200
    data = prog.json()
    assert data["status"] == "done"
    assert data["step"] == "done"
    assert "elapsed_sec" in data


def test_pipeline_progress_running_mid_flight(client):
    fake = _fake_llm()
    real = Pipeline(llm_client=fake)
    entered = threading.Event()
    release = threading.Event()

    def slow_run(question, progress=None):
        assert progress is not None
        progress("boundary", "Step 1/4：解析問題範圍")
        entered.set()
        release.wait(timeout=5)
        return real.run(question, progress=progress)

    pipeline_mock = MagicMock()
    pipeline_mock.run.side_effect = slow_run
    with patch("classifier.api.Pipeline", return_value=pipeline_mock):
        thread = threading.Thread(
            target=lambda: client.post(
                "/pipeline",
                json={"question": "台積電", "client_id": "cid-running"},
            )
        )
        thread.start()
        try:
            assert entered.wait(timeout=5), "pipeline run never started"

            prog = client.get("/pipeline/progress/cid-running")
            assert prog.status_code == 200
            data = prog.json()
            assert data["status"] == "running"
            assert data["step"] == "boundary"
            assert "elapsed_sec" in data
        finally:
            release.set()
            thread.join(timeout=5)


def test_pipeline_progress_unknown_client(client):
    resp = client.get("/pipeline/progress/never-registered")
    assert resp.status_code == 200
    assert resp.json() == {"status": "unknown"}


def test_pipeline_progress_failed_on_error(client):
    fake = _fake_llm()
    fake.extract_structured.side_effect = LLMError("boom")
    real = Pipeline(llm_client=fake)
    with patch("classifier.api.Pipeline", return_value=real):
        resp = client.post(
            "/pipeline",
            json={"question": "台積電", "client_id": "cid-fail"},
        )
    assert resp.status_code == 500

    prog = client.get("/pipeline/progress/cid-fail")
    assert prog.status_code == 200
    data = prog.json()
    assert data["status"] == "failed"
    assert "message" in data


def test_pipeline_without_client_id_no_progress_tracked(client):
    fake = _fake_llm()
    real = Pipeline(llm_client=fake)
    with patch("classifier.api.Pipeline", return_value=real):
        resp = client.post("/pipeline", json={"question": "台積電未來展望"})
    assert resp.status_code == 200

    # No client_id in the request → nothing should be registered.
    assert api_module._progress == {}


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
