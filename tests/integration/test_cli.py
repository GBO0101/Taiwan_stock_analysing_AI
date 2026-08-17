"""CLI integration tests (no real OpenAI/FinMind calls)."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch, MagicMock

from classifier.models import (
    BoundaryResult,
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
