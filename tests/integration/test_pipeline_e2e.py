"""End-to-end pipeline integration tests (no real OpenAI/FinMind calls).

Verifies the STRICT sequential execution contract from plan #32.2:
  - Step 1 succeeds -> Step 2 may execute
  - Step 1 fails     -> Step 2 must NOT execute
  - Step 2 live/factual/non_financial -> Step 3 must NOT execute
  - Step 2 analytical -> Step 3 must execute
  - Step 2 fails     -> Step 3 must NOT execute
"""

from __future__ import annotations

from classifier.llm_client import LLMError
from classifier.models import (
    BoundaryResult,
    ClassificationResult,
    ClassificationType,
    DecompositionResult,
    PipelineResult,
    SubQuery,
)
from classifier.pipeline import Pipeline, PipelineError


class FakeLLMClient:
    """Returns canned models based on the requested response_model class name."""

    def __init__(self, boundary=None, classification=None, decomposition=None, fail_on=None):
        self.boundary = boundary or BoundaryResult(
            stock_codes=["2330"], company_names=["台積電"], confidence=0.9
        )
        self.classification = classification or ClassificationResult(
            type=ClassificationType.ANALYTICAL, confidence=0.9
        )
        self.decomposition = decomposition or DecompositionResult(
            sub_queries=[SubQuery(id="q0", operation="finmind_query", datasets=["TaiwanStockPrice"])]
        )
        # Which step should raise: "boundary", "classification", or "decomposition"
        self.fail_on = fail_on or set()
        self.calls = []

    def extract_structured(self, prompt, response_model):
        name = response_model.__name__
        self.calls.append(name)
        if name == "BoundaryResult":
            if "boundary" in self.fail_on:
                raise LLMError("forced boundary failure")
            return self.boundary
        if name == "ClassificationResult":
            if "classification" in self.fail_on:
                raise LLMError("forced classification failure")
            return self.classification
        if name == "DecompositionResult":
            if "decomposition" in self.fail_on:
                raise LLMError("forced decomposition failure")
            return self.decomposition
        raise RuntimeError(f"unexpected model {name}")


def _pipeline_with(fake: FakeLLMClient) -> Pipeline:
    return Pipeline(llm_client=fake)


def test_step1_success_allows_step2():
    fake = FakeLLMClient()
    result = _pipeline_with(fake).run("台積電未來展望")
    assert "BoundaryResult" in fake.calls
    assert "ClassificationResult" in fake.calls
    assert result.steps[0].status == "completed"
    assert result.steps[1].status == "completed"


def test_step1_failure_blocks_step2():
    fake = FakeLLMClient(fail_on={"boundary"})
    try:
        _pipeline_with(fake).run("台積電")
        assert False, "expected PipelineError"
    except PipelineError:
        pass
    assert "BoundaryResult" in fake.calls
    assert "ClassificationResult" not in fake.calls


def test_non_analytical_skips_step3():
    fake = FakeLLMClient(
        classification=ClassificationResult(type=ClassificationType.LIVE, confidence=0.9)
    )
    result = _pipeline_with(fake).run("台積電現在股價多少")
    assert "DecompositionResult" not in fake.calls
    types = [s.step for s in result.steps]
    assert types == ["boundary", "classification", "decomposition"]
    assert result.steps[2].status == "skipped"


def test_factual_skips_step3():
    fake = FakeLLMClient(
        classification=ClassificationResult(type=ClassificationType.FACTUAL, confidence=0.9)
    )
    result = _pipeline_with(fake).run("台積電2024年營收")
    assert "DecompositionResult" not in fake.calls
    assert result.steps[2].status == "skipped"


def test_non_financial_skips_step3():
    fake = FakeLLMClient(
        classification=ClassificationResult(type=ClassificationType.NON_FINANCIAL, confidence=0.9)
    )
    result = _pipeline_with(fake).run("香蕉好吃嗎")
    assert "DecompositionResult" not in fake.calls
    assert result.steps[2].status == "skipped"


def test_analytical_runs_step3():
    fake = FakeLLMClient(
        classification=ClassificationResult(type=ClassificationType.ANALYTICAL, confidence=0.9)
    )
    result = _pipeline_with(fake).run("台積電未來展望如何")
    assert "DecompositionResult" in fake.calls
    assert result.steps[2].status == "completed"


def test_step2_failure_blocks_step3():
    fake = FakeLLMClient(fail_on={"classification"})
    try:
        _pipeline_with(fake).run("台積電")
        assert False, "expected PipelineError"
    except PipelineError:
        pass
    assert "ClassificationResult" in fake.calls
    assert "DecompositionResult" not in fake.calls
