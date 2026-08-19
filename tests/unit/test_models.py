"""Unit tests for pipeline contract models."""

import pytest
from datetime import date, timedelta
from classifier.models import (
    ClassificationType,
    ChartType,
    ChartDataRequirement,
    OperationType,
    StepStatus,
    TimeScope,
    StockScope,
    DataDimension,
    Market,
    DateRange,
    BoundaryResult,
    ClassificationResult,
    SubQuery,
    DecompositionResult,
    PipelineStepResult,
    PipelineResult,
    ChartRequest,
)


class TestEnums:
    """Test enum values."""

    def test_classification_type_values(self):
        assert ClassificationType.LIVE == "live"
        assert ClassificationType.FACTUAL == "factual"
        assert ClassificationType.ANALYTICAL == "analytical"
        assert ClassificationType.NON_FINANCIAL == "non_financial"

    def test_chart_type_values(self):
        assert ChartType.LINE == "line"
        assert ChartType.KLINE == "kline"
        assert ChartType.BAR == "bar"

    def test_chart_data_requirement_values(self):
        assert ChartDataRequirement.PRICE_TREND == "price_trend"
        assert ChartDataRequirement.PRICE_OHLC == "price_ohlc"
        assert ChartDataRequirement.REVENUE_TREND == "revenue_trend"
        assert ChartDataRequirement.REVENUE_COMPARISON == "revenue_comparison"
        assert ChartDataRequirement.INDICATOR_TREND == "indicator_trend"
        assert ChartDataRequirement.SECTOR_ANALYSIS == "sector_analysis"

    def test_operation_type_values(self):
        assert OperationType.FINMIND_QUERY == "finmind_query"
        assert OperationType.COMPUTE == "compute"
        assert OperationType.SENTIMENT_ANALYSIS == "sentiment_analysis"

    def test_step_status_values(self):
        assert StepStatus.COMPLETED == "completed"
        assert StepStatus.FAILED == "failed"
        assert StepStatus.SKIPPED == "skipped"


class TestDateRange:
    """Test DateRange model."""

    def test_valid_relative_range(self):
        dr = DateRange(type=TimeScope.RELATIVE, value="30d")
        assert dr.type == TimeScope.RELATIVE
        assert dr.value == "30d"

    def test_valid_absolute_range(self):
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024-01-01/2024-12-31")
        assert dr.type == TimeScope.ABSOLUTE
        assert dr.value == "2024-01-01/2024-12-31"

    def test_resolve_absolute_range(self):
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024-01-01/2024-05-31")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 5, 31))

    def test_resolve_absolute_month_precision(self):
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024-01/2024-05")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 5, 1))

    def test_resolve_absolute_swapped_bounds(self):
        # LLM may emit end/start reversed; result must be ordered.
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024-05-31/2024-01-01")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 5, 31))

    def test_resolve_relative_range(self):
        dr = DateRange(type=TimeScope.RELATIVE, value="30d")
        start, end = dr.resolve_dates()
        assert end == date.today()
        assert (end - start).days == 30

    def test_resolve_relative_units(self):
        assert DateRange(type=TimeScope.RELATIVE, value="2w").resolve_dates()[1] == date.today()
        assert DateRange(type=TimeScope.RELATIVE, value="3m").resolve_dates()[1] == date.today()
        assert DateRange(type=TimeScope.RELATIVE, value="1y").resolve_dates()[1] == date.today()

    def test_resolve_invalid_returns_none(self):
        assert DateRange(type=TimeScope.ABSOLUTE, value="not-a-date").resolve_dates() is None
        assert DateRange(type=TimeScope.RELATIVE, value="abc").resolve_dates() is None

    def test_resolve_absolute_single_full_date(self):
        # A single explicit day resolves to that day (no silent fallback to a
        # now-based window, which was the original bug).
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024-01-01")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 1, 1))

    def test_resolve_absolute_slash_separators(self):
        # LLM may emit slashes instead of dashes for both date and range.
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024/01/2024/07")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 7, 1))

    def test_resolve_absolute_word_separator(self):
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024-01-01 to 2024-07-31")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 7, 31))

    def test_resolve_absolute_tilde_separator(self):
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024-01-01~2024-07-31")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 7, 31))

    def test_resolve_absolute_chinese_range(self):
        # The exact phrasing from the bug report.
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024年1月到7月")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 7, 31))

    def test_resolve_absolute_chinese_full_day(self):
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024年1月1日到7月31日")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 7, 31))

    def test_resolve_absolute_chinese_tilde_range(self):
        # "1~6月" uses a tilde separator between months; must span Jan-Jun.
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024年 1~6月")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 6, 30))

    def test_resolve_absolute_chinese_dash_range(self):
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024年1月-7月")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 7, 31))

    def test_resolve_absolute_single_month_expands(self):
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024-01")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 1, 31))

    def test_resolve_absolute_single_year_expands(self):
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 12, 31))

    def test_resolve_absolute_chinese_year_only(self):
        dr = DateRange(type=TimeScope.ABSOLUTE, value="2024年")
        assert dr.resolve_dates() == (date(2024, 1, 1), date(2024, 12, 31))


class TestBoundaryResult:
    """Test BoundaryResult model."""

    def test_minimal_valid(self):
        br = BoundaryResult(confidence=0.9)
        assert br.confidence == 0.9
        assert br.stock_codes == []
        assert br.market == Market.TWSE

    def test_full_valid(self):
        br = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            sectors=["半導體"],
            date_range=DateRange(type=TimeScope.RELATIVE, value="30d"),
            time_scope=TimeScope.RELATIVE,
            stock_scope=StockScope.SINGLE_STOCK,
            data_dimension=DataDimension.PRICE,
            market=Market.TWSE,
            metrics=["price"],
            chart_type=ChartType.KLINE,
            confidence=0.97,
        )
        assert br.stock_codes == ["2330"]
        assert br.chart_type == ChartType.KLINE

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            BoundaryResult(confidence=1.5)
        with pytest.raises(ValueError):
            BoundaryResult(confidence=-0.1)

    def test_serialization(self):
        br = BoundaryResult(
            stock_codes=["2330"],
            confidence=0.9,
        )
        data = br.model_dump()
        assert data["stock_codes"] == ["2330"]
        assert data["confidence"] == 0.9


class TestClassificationResult:
    """Test ClassificationResult model."""

    def test_live_classification(self):
        cr = ClassificationResult(
            type=ClassificationType.LIVE,
            confidence=0.95,
        )
        assert cr.type == ClassificationType.LIVE
        assert cr.needs_clarification is False
        assert cr.needs_visualization is False

    def test_analytical_with_visualization(self):
        cr = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.9,
            needs_visualization=True,
            chart_type=ChartType.LINE,
            chart_data_requirements=ChartDataRequirement.PRICE_TREND,
            target_datasets=["TaiwanStockPrice"],
        )
        assert cr.type == ClassificationType.ANALYTICAL
        assert cr.needs_visualization is True
        assert cr.chart_type == ChartType.LINE
        assert cr.chart_data_requirements == ChartDataRequirement.PRICE_TREND

    def test_non_financial(self):
        cr = ClassificationResult(
            type=ClassificationType.NON_FINANCIAL,
            confidence=0.99,
            needs_clarification=True,
            clarification_question="這個問題超出台灣股票領域範圍。",
        )
        assert cr.type == ClassificationType.NON_FINANCIAL
        assert cr.needs_clarification is True


class TestSubQuery:
    """Test SubQuery model."""

    def test_finmind_query_valid(self):
        sq = SubQuery(
            id="q0",
            operation=OperationType.FINMIND_QUERY,
            datasets=["TaiwanStockPrice"],
            params={"stock_id": "2330"},
        )
        assert sq.id == "q0"
        assert sq.operation == OperationType.FINMIND_QUERY
        assert sq.datasets == ["TaiwanStockPrice"]

    def test_finmind_query_requires_dataset(self):
        with pytest.raises(ValueError, match="finmind_query operation requires at least one dataset"):
            SubQuery(
                id="q0",
                operation=OperationType.FINMIND_QUERY,
                datasets=[],
            )

    def test_compute_operation(self):
        sq = SubQuery(
            id="q1",
            operation=OperationType.COMPUTE,
            computed_field="gross_margin",
            formula="q0.value / q1.value",
            depends_on=["q0", "q1"],
        )
        assert sq.operation == OperationType.COMPUTE
        assert sq.computed_field == "gross_margin"
        assert sq.formula == "q0.value / q1.value"

    def test_sentiment_analysis_operation(self):
        sq = SubQuery(
            id="q2",
            operation=OperationType.SENTIMENT_ANALYSIS,
            datasets=["TaiwanStockNews"],
            depends_on=["q1"],
        )
        assert sq.operation == OperationType.SENTIMENT_ANALYSIS


class TestDecompositionResult:
    """Test DecompositionResult model."""

    def test_valid_dag(self):
        queries = [
            SubQuery(id="q0", operation=OperationType.FINMIND_QUERY, datasets=["TaiwanStockInfo"]),
            SubQuery(id="q1", operation=OperationType.FINMIND_QUERY, datasets=["TaiwanStockNews"], depends_on=["q0"]),
            SubQuery(id="q2", operation=OperationType.COMPUTE, computed_field="score", formula="q0 + q1", depends_on=["q0", "q1"]),
        ]
        dr = DecompositionResult(sub_queries=queries)
        assert len(dr.sub_queries) == 3

    def test_duplicate_ids_rejected(self):
        queries = [
            SubQuery(id="q0", operation=OperationType.FINMIND_QUERY, datasets=["TaiwanStockInfo"]),
            SubQuery(id="q0", operation=OperationType.FINMIND_QUERY, datasets=["TaiwanStockNews"]),
        ]
        with pytest.raises(ValueError, match="Sub-query IDs must be unique"):
            DecompositionResult(sub_queries=queries)

    def test_missing_dependency_rejected(self):
        queries = [
            SubQuery(id="q1", operation=OperationType.FINMIND_QUERY, datasets=["TaiwanStockNews"], depends_on=["q0"]),
        ]
        with pytest.raises(ValueError, match="Dependency 'q0' not found"):
            DecompositionResult(sub_queries=queries)

    def test_cycle_rejected(self):
        queries = [
            SubQuery(id="q0", operation=OperationType.FINMIND_QUERY, datasets=["TaiwanStockInfo"], depends_on=["q1"]),
            SubQuery(id="q1", operation=OperationType.FINMIND_QUERY, datasets=["TaiwanStockNews"], depends_on=["q0"]),
        ]
        with pytest.raises(ValueError, match="Sub-query DAG contains cycles"):
            DecompositionResult(sub_queries=queries)

    def test_wrong_topological_order_rejected(self):
        queries = [
            SubQuery(id="q1", operation=OperationType.FINMIND_QUERY, datasets=["TaiwanStockNews"], depends_on=["q0"]),
            SubQuery(id="q0", operation=OperationType.FINMIND_QUERY, datasets=["TaiwanStockInfo"]),
        ]
        with pytest.raises(ValueError, match="Prerequisite 'q0' must appear before dependent"):
            DecompositionResult(sub_queries=queries)

    def test_formula_references_valid_nodes(self):
        queries = [
            SubQuery(id="q0", operation=OperationType.FINMIND_QUERY, datasets=["TaiwanStockInfo"]),
            SubQuery(id="q1", operation=OperationType.COMPUTE, formula="q0.value * 2", depends_on=["q0"]),
        ]
        dr = DecompositionResult(sub_queries=queries)
        assert len(dr.sub_queries) == 2

    def test_formula_invalid_reference_rejected(self):
        queries = [
            SubQuery(id="q0", operation=OperationType.FINMIND_QUERY, datasets=["TaiwanStockInfo"]),
            SubQuery(id="q1", operation=OperationType.COMPUTE, formula="q99.value * 2", depends_on=["q0"]),
        ]
        with pytest.raises(ValueError, match="Formula references unknown node 'q99'"):
            DecompositionResult(sub_queries=queries)


class TestPipelineStepResult:
    """Test PipelineStepResult model."""

    def test_completed_step(self):
        psr = PipelineStepResult(
            step="boundary",
            status=StepStatus.COMPLETED,
            output={"stock_codes": ["2330"]},
        )
        assert psr.step == "boundary"
        assert psr.status == StepStatus.COMPLETED
        assert psr.output == {"stock_codes": ["2330"]}

    def test_failed_step(self):
        psr = PipelineStepResult(
            step="classification",
            status=StepStatus.FAILED,
            output={},
        )
        assert psr.status == StepStatus.FAILED


class TestPipelineResult:
    """Test PipelineResult model."""

    def test_full_pipeline_result(self):
        pr = PipelineResult(
            question="台積電未來展望",
            steps=[
                PipelineStepResult(step="boundary", status=StepStatus.COMPLETED, output={}),
                PipelineStepResult(step="classification", status=StepStatus.COMPLETED, output={}),
                PipelineStepResult(step="decomposition", status=StepStatus.COMPLETED, output={}),
            ],
        )
        assert pr.question == "台積電未來展望"
        assert len(pr.steps) == 3

    def test_non_analytical_no_step3(self):
        pr = PipelineResult(
            question="台積電現在股價多少",
            steps=[
                PipelineStepResult(step="boundary", status=StepStatus.COMPLETED, output={}),
                PipelineStepResult(step="classification", status=StepStatus.COMPLETED, output={}),
            ],
        )
        assert len(pr.steps) == 2
        assert all(s.step != "decomposition" for s in pr.steps)


class TestChartRequest:
    """Test ChartRequest model."""

    def test_valid_chart_request(self):
        cr = ChartRequest(
            stock_codes=["2330"],
            chart_data_requirements=ChartDataRequirement.PRICE_TREND,
            chart_type=ChartType.LINE,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
        )
        assert cr.stock_codes == ["2330"]
        assert cr.chart_data_requirements == ChartDataRequirement.PRICE_TREND
        assert cr.chart_type == ChartType.LINE

    def test_optional_dates(self):
        cr = ChartRequest(
            stock_codes=["2330"],
            chart_data_requirements=ChartDataRequirement.PRICE_OHLC,
            chart_type=ChartType.KLINE,
        )
        assert cr.start_date is None
        assert cr.end_date is None

    def test_date_range_resolves_to_start_end(self):
        # A ChartRequest built from a pipeline date_range must populate the
        # concrete start/end dates the renderer consumes (bug: chart ignored
        # the requested range and showed the current period instead).
        cr = ChartRequest(
            stock_codes=["2330"],
            chart_data_requirements=ChartDataRequirement.PRICE_TREND,
            chart_type=ChartType.LINE,
            date_range=DateRange(type=TimeScope.ABSOLUTE, value="2024-01-01/2024-05-31"),
        )
        assert cr.start_date == date(2024, 1, 1)
        assert cr.end_date == date(2024, 5, 31)

    def test_explicit_dates_take_precedence_over_range(self):
        cr = ChartRequest(
            stock_codes=["2330"],
            chart_data_requirements=ChartDataRequirement.PRICE_TREND,
            chart_type=ChartType.LINE,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            date_range=DateRange(type=TimeScope.ABSOLUTE, value="2024-01-01/2024-05-31"),
        )
        assert cr.start_date == date(2023, 1, 1)
        assert cr.end_date == date(2023, 12, 31)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])