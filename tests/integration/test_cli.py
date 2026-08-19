"""CLI integration tests (no real OpenAI/FinMind calls)."""

from __future__ import annotations

import json
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from unittest.mock import patch, MagicMock

from classifier.models import (
    BoundaryResult,
    ChartDataRequirement,
    ChartType,
    ClassificationResult,
    ClassificationType,
    DecompositionResult,
    PipelineResult,
    PipelineStepResult,
    StepStatus,
    SubQuery,
)
from classifier.pipeline import Pipeline, PipelineError
from classifier import cli as cli_module


def _fake_pipeline(result=None, raise_error=False):
    fake = MagicMock()
    if raise_error:
        fake.run.side_effect = PipelineError("forced pipeline failure")
    else:
        fake.run.return_value = result
    return fake


def _sample_result(question="台積電未來展望", ctype=ClassificationType.ANALYTICAL):
    return PipelineResult(
        question=question,
        steps=[
            PipelineStepResult(step="boundary", status=StepStatus.COMPLETED, output=BoundaryResult(stock_codes=["2330"], confidence=0.9).model_dump()),
            PipelineStepResult(step="classification", status=StepStatus.COMPLETED, output=ClassificationResult(type=ctype, confidence=0.9).model_dump()),
            PipelineStepResult(step="decomposition", status=StepStatus.COMPLETED, output=DecompositionResult(sub_queries=[SubQuery(id="q0", operation="finmind_query", datasets=["TaiwanStockPrice"])]).model_dump()),
        ],
    )


def test_cli_oneshot_success():
    result = _sample_result()
    fake = _fake_pipeline(result=result)
    with patch("classifier.cli.Pipeline", return_value=fake), redirect_stdout(StringIO()) as buf:
        code = cli_module.main([result.question])
    assert code == 0
    data = json.loads(buf.getvalue())
    assert data["question"] == result.question
    assert [s["step"] for s in data["steps"]] == ["boundary", "classification", "decomposition"]


def test_cli_oneshot_pipeline_error_returns_nonzero():
    fake = _fake_pipeline(raise_error=True)
    with patch("classifier.cli.Pipeline", return_value=fake), redirect_stdout(StringIO()):
        code = cli_module.main(["台積電"])
    assert code == 1


def test_cli_no_args_returns_help_code():
    code = cli_module.main([])
    assert code == 1


def test_cli_chat_mode_runs_turn():
    result = _sample_result()
    fake = _fake_pipeline(result=result)
    inputs = iter(["台積電未來展望", "exit"])
    with patch("classifier.cli.Pipeline", return_value=fake), patch("builtins.input", side_effect=lambda *a: next(inputs)), redirect_stdout(StringIO()) as buf:
        code = cli_module.main(["--chat"])
    assert code == 0
    data = json.loads(buf.getvalue().strip())
    assert data["question"] == "台積電未來展望"


def test_cli_oneshot_viz_requested_but_no_stock_emits_note():
    """When boundary wants a chart but no stock resolved, surface a clear note.

    Chart intent now lives on the boundary (single source of truth), so the
    boundary must carry ``chart_data_requirements`` for the note to fire.
    """
    result = PipelineResult(
        question="啞妮 2024年1月到7月趨勢圖",
        steps=[
            PipelineStepResult(
                step="boundary",
                status=StepStatus.COMPLETED,
                output=BoundaryResult(
                    stock_codes=[],
                    confidence=0.9,
                    chart_data_requirements=ChartDataRequirement.PRICE_TREND,
                ).model_dump(),
            ),
            PipelineStepResult(
                step="classification",
                status=StepStatus.COMPLETED,
                output=ClassificationResult(
                    type=ClassificationType.FACTUAL,
                    confidence=0.9,
                    needs_visualization=True,
                ).model_dump(),
            ),
        ],
    )
    fake = _fake_pipeline(result=result)
    with patch("classifier.cli.Pipeline", return_value=fake), redirect_stdout(StringIO()) as buf, redirect_stderr(StringIO()) as err:
        code = cli_module.main([result.question])
    assert code == 0
    data = json.loads(buf.getvalue())
    assert "chart_note" in data
    assert "stock_codes" in data["chart_note"]
    assert "未產生圖表" in err.getvalue()


def test_cli_build_chart_request_uses_boundary_source():
    """_build_chart_request reads chart intent from boundary, not classification."""
    result = PipelineResult(
        question="台積電股價走勢圖",
        steps=[
            PipelineStepResult(
                step="boundary",
                status=StepStatus.COMPLETED,
                output=BoundaryResult(
                    stock_codes=["2330"],
                    confidence=0.9,
                    chart_data_requirements=ChartDataRequirement.PRICE_TREND,
                    chart_type=ChartType.LINE,
                ).model_dump(),
            ),
            # Classification deliberately disagrees (stale/wrong) — must be ignored.
            PipelineStepResult(
                step="classification",
                status=StepStatus.COMPLETED,
                output=ClassificationResult(
                    type=ClassificationType.FACTUAL,
                    confidence=0.9,
                    needs_visualization=False,
                    chart_data_requirements=None,
                ).model_dump(),
            ),
        ],
    )
    req = cli_module._build_chart_request(result)
    assert req is not None
    assert req.chart_data_requirements == ChartDataRequirement.PRICE_TREND
    assert req.chart_type == ChartType.LINE
    assert req.stock_codes == ["2330"]


def test_cli_no_chart_reason_none_when_boundary_has_no_viz():
    """No boundary chart requirement -> _no_chart_reason returns None."""
    result = PipelineResult(
        question="台積電現在股價多少",
        steps=[
            PipelineStepResult(
                step="boundary",
                status=StepStatus.COMPLETED,
                output=BoundaryResult(stock_codes=["2330"], confidence=0.9).model_dump(),
            ),
            PipelineStepResult(
                step="classification",
                status=StepStatus.COMPLETED,
                output=ClassificationResult(
                    type=ClassificationType.FACTUAL,
                    confidence=0.9,
                    needs_visualization=True,
                    chart_data_requirements=ChartDataRequirement.PRICE_TREND,
                ).model_dump(),
            ),
        ],
    )
    assert cli_module._no_chart_reason(result) is None
