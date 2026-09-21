"""Unit tests for Pipeline Runner."""

from unittest.mock import Mock, patch

import pytest

from classifier.annual_report import AnnualReportError
from classifier.boundary import BoundaryExtractionError
from classifier.classification import ClassificationError
from classifier.decomposition import DecompositionError
from classifier.models import (
    BoundaryResult,
    ClassificationResult,
    ClassificationType,
    DecompositionResult,
    OperationType,
    StepStatus,
    SubQuery,
)
from classifier.pipeline import Pipeline, PipelineError
from classifier.stock_enumeration import _SupplyChainModel


class TestPipeline:
    """Test pipeline runner functionality."""

    @patch(
        "classifier.stock_enumeration.extract_annual_report",
        side_effect=AnnualReportError("no network in tests"),
    )
    @patch("classifier.pipeline.LLMClient")
    @patch("classifier.pipeline.IndicatorMapper")
    def test_pipeline_analytical_query(self, mock_mapper_class, mock_llm_class, mock_annual_report):
        """Test full pipeline with analytical query."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper
        mock_mapper.to_prompt_format.return_value = []

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        classification = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.92,
        )
        decomposition = DecompositionResult(
            sub_queries=[
                SubQuery(
                    id="q0",
                    operation=OperationType.FINMIND_QUERY,
                    datasets=["TaiwanStockPrice"],
                    params={"stock_id": "2330"},
                    depends_on=[],
                ),
            ]
        )

        mock_llm.extract_structured.side_effect = [
            boundary,
            classification,
            decomposition,
            _SupplyChainModel(upstream=[], downstream=[], confidence=0.3),
        ]

        pipeline = Pipeline(llm_client=mock_llm, indicator_mapper=mock_mapper)
        result = pipeline.run("台積電近五年獲利能力如何")

        assert result.question == "台積電近五年獲利能力如何"
        assert len(result.steps) == 4
        assert result.steps[0].step == "boundary"
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[1].step == "classification"
        assert result.steps[1].status == StepStatus.COMPLETED
        assert result.steps[2].step == "decomposition"
        assert result.steps[2].status == StepStatus.COMPLETED
        assert result.steps[3].step == "stock_enumeration"
        assert result.steps[3].status == StepStatus.COMPLETED
        # Completed Step 4 → unified summary present.
        assert result.summary is not None
        assert result.summary.topic_code == "2330"

    @patch(
        "classifier.stock_enumeration.extract_annual_report",
        side_effect=AnnualReportError("no network in tests"),
    )
    @patch("classifier.pipeline.LLMClient")
    @patch("classifier.pipeline.IndicatorMapper")
    def test_pipeline_live_query_skips_decomposition(
        self, mock_mapper_class, mock_llm_class, mock_annual_report
    ):
        """Test pipeline skips decomposition for live queries."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        classification = ClassificationResult(
            type=ClassificationType.LIVE,
            confidence=0.9,
        )

        mock_llm.extract_structured.side_effect = [
            boundary,
            classification,
            _SupplyChainModel(upstream=[], downstream=[], confidence=0.3),
        ]

        pipeline = Pipeline(llm_client=mock_llm, indicator_mapper=mock_mapper)
        result = pipeline.run("台積電現在股價多少")

        assert len(result.steps) == 4
        assert result.steps[2].step == "decomposition"
        assert result.steps[2].status == StepStatus.SKIPPED
        assert result.steps[3].step == "stock_enumeration"

    @patch(
        "classifier.stock_enumeration.extract_annual_report",
        side_effect=AnnualReportError("no network in tests"),
    )
    @patch("classifier.pipeline.LLMClient")
    @patch("classifier.pipeline.IndicatorMapper")
    def test_pipeline_factual_query_skips_decomposition(
        self, mock_mapper_class, mock_llm_class, mock_annual_report
    ):
        """Test pipeline skips decomposition for factual queries."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        classification = ClassificationResult(
            type=ClassificationType.FACTUAL,
            confidence=0.85,
        )

        mock_llm.extract_structured.side_effect = [
            boundary,
            classification,
            _SupplyChainModel(upstream=[], downstream=[], confidence=0.3),
        ]

        pipeline = Pipeline(llm_client=mock_llm, indicator_mapper=mock_mapper)
        result = pipeline.run("台積電2024年EPS是多少")

        assert len(result.steps) == 4
        assert result.steps[2].status == StepStatus.SKIPPED
        assert result.steps[3].step == "stock_enumeration"
        assert result.steps[3].status == StepStatus.COMPLETED

    @patch("classifier.pipeline.LLMClient")
    @patch("classifier.pipeline.IndicatorMapper")
    def test_pipeline_non_financial_query(self, mock_mapper_class, mock_llm_class):
        """Test pipeline with non-financial query."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper

        boundary = BoundaryResult(
            stock_codes=[],
            company_names=[],
            confidence=0.5,
        )
        classification = ClassificationResult(
            type=ClassificationType.NON_FINANCIAL,
            confidence=0.9,
        )

        mock_llm.extract_structured.side_effect = [boundary, classification]

        pipeline = Pipeline(llm_client=mock_llm, indicator_mapper=mock_mapper)
        result = pipeline.run("香蕉好吃嗎")

        assert len(result.steps) == 4
        assert result.steps[2].status == StepStatus.SKIPPED
        assert result.steps[3].step == "stock_enumeration"
        assert result.steps[3].status == StepStatus.SKIPPED

    @patch("classifier.pipeline.LLMClient")
    @patch("classifier.pipeline.IndicatorMapper")
    def test_pipeline_boundary_failure(self, mock_mapper_class, mock_llm_class):
        """Test pipeline handles boundary extraction failure."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper

        mock_llm.extract_structured.side_effect = BoundaryExtractionError("Extraction failed")

        pipeline = Pipeline(llm_client=mock_llm, indicator_mapper=mock_mapper)

        with pytest.raises(PipelineError, match="Step 1"):
            pipeline.run("台積電股價")

    @patch("classifier.pipeline.LLMClient")
    @patch("classifier.pipeline.IndicatorMapper")
    def test_pipeline_classification_failure(self, mock_mapper_class, mock_llm_class):
        """Test pipeline handles classification failure."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper

        boundary = BoundaryResult(stock_codes=["2330"], confidence=0.95)
        mock_llm.extract_structured.side_effect = [
            boundary,
            ClassificationError("Classification failed"),
        ]

        pipeline = Pipeline(llm_client=mock_llm, indicator_mapper=mock_mapper)

        with pytest.raises(PipelineError, match="Step 2"):
            pipeline.run("台積電股價")

    @patch("classifier.pipeline.LLMClient")
    @patch("classifier.pipeline.IndicatorMapper")
    def test_pipeline_decomposition_failure(self, mock_mapper_class, mock_llm_class):
        """Test pipeline handles decomposition failure."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper
        mock_mapper.to_prompt_format.return_value = []

        boundary = BoundaryResult(stock_codes=["2330"], confidence=0.95)
        classification = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.92,
        )
        mock_llm.extract_structured.side_effect = [
            boundary,
            classification,
            DecompositionError("Decomposition failed"),
        ]

        pipeline = Pipeline(llm_client=mock_llm, indicator_mapper=mock_mapper)

        with pytest.raises(PipelineError, match="Step 3"):
            pipeline.run("台積電未來展望")

    @patch(
        "classifier.stock_enumeration.extract_annual_report",
        side_effect=AnnualReportError("no network in tests"),
    )
    @patch("classifier.pipeline.LLMClient")
    @patch("classifier.pipeline.IndicatorMapper")
    def test_pipeline_with_context(self, mock_mapper_class, mock_llm_class, mock_annual_report):
        """Test pipeline with chat context."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        classification = ClassificationResult(
            type=ClassificationType.LIVE,
            confidence=0.9,
        )

        mock_llm.extract_structured.side_effect = [
            boundary,
            classification,
            _SupplyChainModel(upstream=[], downstream=[], confidence=0.3),
        ]

        context = {
            "last_question": "台積電",
            "last_boundary": {"stock_codes": ["2330"], "company_names": ["台積電"]},
        }

        pipeline = Pipeline(llm_client=mock_llm, indicator_mapper=mock_mapper)
        result = pipeline.run("它最近股價如何？", context=context)

        assert result.question == "它最近股價如何？"
        assert result.steps[0].status == StepStatus.COMPLETED
        # Single-stock live query runs Step 4 (supply chain).
        assert result.steps[3].step == "stock_enumeration"
        assert result.steps[3].status == StepStatus.COMPLETED

    @patch(
        "classifier.stock_enumeration.extract_annual_report",
        side_effect=AnnualReportError("no network in tests"),
    )
    @patch("classifier.pipeline.LLMClient")
    @patch("classifier.pipeline.IndicatorMapper")
    def test_pipeline_progress_callback_order(
        self, mock_mapper_class, mock_llm_class, mock_annual_report
    ):
        """Analytical query: progress fires boundary → classification → decomposition
        → stock_enumeration → done, each with a non-empty message."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper
        mock_mapper.to_prompt_format.return_value = []

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        classification = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.92,
        )
        decomposition = DecompositionResult(
            sub_queries=[
                SubQuery(
                    id="q0",
                    operation=OperationType.FINMIND_QUERY,
                    datasets=["TaiwanStockPrice"],
                ),
            ]
        )

        mock_llm.extract_structured.side_effect = [
            boundary,
            classification,
            decomposition,
            _SupplyChainModel(upstream=[], downstream=[], confidence=0.3),
        ]

        events: list[tuple[str, str]] = []
        pipeline = Pipeline(llm_client=mock_llm, indicator_mapper=mock_mapper)
        pipeline.run(
            "台積電近五年獲利能力如何",
            progress=lambda step, msg: events.append((step, msg)),
        )

        steps_order = [step for step, _ in events]
        # Top-level steps, in strict order, each with a message; Step 4 also
        # fires sub-step events (annual-report extraction) between the top-level
        # stock_enumeration marker and "done".
        assert "boundary" in steps_order
        assert steps_order.index("boundary") < steps_order.index("classification")
        assert steps_order.index("classification") < steps_order.index("decomposition")
        assert steps_order.index("decomposition") < steps_order.index("stock_enumeration")
        assert steps_order.index("stock_enumeration") < steps_order.index("done")
        assert steps_order[-1] == "done"
        assert all(msg for _, msg in events)
        # The extraction sub-step message mentions the target stock.
        assert any("2330" in msg and "年報" in msg for _, msg in events)

    @patch("classifier.pipeline.LLMClient")
    @patch("classifier.pipeline.IndicatorMapper")
    def test_pipeline_progress_non_analytical_skips_decomposition(
        self, mock_mapper_class, mock_llm_class
    ):
        """Non-analytical query: progress has no decomposition step."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper
        mock_mapper.to_prompt_format.return_value = []

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        classification = ClassificationResult(
            type=ClassificationType.LIVE,
            confidence=0.9,
        )

        mock_llm.extract_structured.side_effect = [
            boundary,
            classification,
            _SupplyChainModel(upstream=[], downstream=[], confidence=0.3),
        ]

        events: list[tuple[str, str]] = []
        pipeline = Pipeline(llm_client=mock_llm, indicator_mapper=mock_mapper)
        pipeline.run(
            "台積電現在股價多少",
            progress=lambda step, msg: events.append((step, msg)),
        )

        steps_order = [step for step, _ in events]
        # LIVE queries skip decomposition — no step-3 marker at all.
        assert "decomposition" not in steps_order
        assert "boundary" in steps_order
        assert steps_order.index("boundary") < steps_order.index("classification")
        assert steps_order.index("classification") < steps_order.index("stock_enumeration")
        assert steps_order[-1] == "done"

    @patch("classifier.pipeline.LLMClient")
    @patch("classifier.pipeline.IndicatorMapper")
    def test_pipeline_progress_not_called_when_omitted(self, mock_mapper_class, mock_llm_class):
        """Backward compatible: progress=None is the default and never breaks."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper
        mock_mapper.to_prompt_format.return_value = []

        boundary = BoundaryResult(stock_codes=["2330"], confidence=0.95)
        classification = ClassificationResult(type=ClassificationType.LIVE, confidence=0.9)

        mock_llm.extract_structured.side_effect = [
            boundary,
            classification,
            _SupplyChainModel(upstream=[], downstream=[], confidence=0.3),
        ]

        pipeline = Pipeline(llm_client=mock_llm, indicator_mapper=mock_mapper)
        result = pipeline.run("台積電現在股價多少")

        assert len(result.steps) == 4
        assert result.steps[0].status == StepStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
