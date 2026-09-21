"""Unit tests for Step 1 Boundary Extraction."""

from unittest.mock import Mock, patch

import pytest

from classifier.boundary import BoundaryExtractionError, extract_boundary
from classifier.llm_client import LLMError
from classifier.models import (
    BoundaryResult,
    ChartDataRequirement,
    ChartType,
    DateRange,
    TimeScope,
)


class TestBoundaryExtraction:
    """Test boundary extraction functionality."""

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_success(self, mock_llm_client_class):
        """Test successful boundary extraction."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        expected_result = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            sectors=["半導體"],
            date_range={"type": "relative", "value": "30d"},
            time_scope="relative",
            stock_scope="single_stock",
            data_dimension="price",
            market="TWSE",
            metrics=["price"],
            chart_type="line",
            confidence=0.95,
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("台積電近30天走勢圖")

        assert result.stock_codes == ["2330"]
        assert result.company_names == ["台積電"]
        assert result.time_scope == "relative"
        assert result.chart_type == "line"
        assert result.confidence == 0.95

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_kline(self, mock_llm_client_class):
        """Test boundary extraction with K-line chart type."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        expected_result = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            chart_type="kline",
            confidence=0.97,
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("台積電K線圖")

        assert result.chart_type == "kline"
        assert result.confidence == 0.97

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_with_context(self, mock_llm_client_class):
        """Test boundary extraction with chat context."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        expected_result = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.9,
        )
        mock_client.extract_structured.return_value = expected_result

        context = {
            "last_question": "台積電",
            "last_boundary": {"stock_codes": ["2330"], "company_names": ["台積電"]},
        }
        result = extract_boundary("它最近股價如何？", context=context)

        assert result.stock_codes == ["2330"]
        mock_client.extract_structured.assert_called_once()

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_llm_error(self, mock_llm_client_class):
        """Test boundary extraction handles LLM errors."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        mock_client.extract_structured.side_effect = LLMError("API timeout")

        with pytest.raises(BoundaryExtractionError, match="Boundary extraction failed"):
            extract_boundary("台積電股價")

    def test_extract_boundary_custom_client(self):
        """Test boundary extraction with custom LLM client."""
        mock_client = Mock()
        expected_result = BoundaryResult(stock_codes=["2330"], confidence=0.9)
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("台積電", llm_client=mock_client)

        assert result.stock_codes == ["2330"]
        mock_client.extract_structured.assert_called_once()

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_resolves_name_only(self, mock_llm_client_class):
        """Name-only query: resolver fills stock_codes deterministically (gap #1)."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        # LLM follows Rule 1: puts name in company_names, does NOT guess the code.
        expected_result = BoundaryResult(
            stock_codes=[],
            company_names=["台積電"],
            confidence=0.9,
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("台積電股價走勢圖")

        assert result.stock_codes == ["2330"]
        assert result.company_names == ["台積電"]

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_drops_unknown_llm_code(self, mock_llm_client_class):
        """LLM-guessed code not in map is dropped; name still resolves (gap #1,#6)."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        expected_result = BoundaryResult(
            stock_codes=["9999"],  # LLM guessed wrong
            company_names=["台積電"],
            confidence=0.9,
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("台積電股價走勢圖")

        assert result.stock_codes == ["2330"]

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_code_query_shows_chinese_name(self, mock_llm_client_class):
        """Query by code: company_names must carry the Chinese name, not the code.

        Reproduces the report where ``company_names`` stayed ``["3008"]`` instead
        of ``["大立光"]`` when the user asked by stock code.
        """
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        expected_result = BoundaryResult(
            stock_codes=["3008"],  # LLM echoed the code as both code and name
            company_names=["3008"],
            confidence=0.9,
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("3008股價走勢圖")

        assert result.stock_codes == ["3008"]
        assert result.company_names == ["大立光"]

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_reconciles_flaky_relative_to_absolute(self, mock_llm_client_class):
        """A year-anchored range the LLM wrongly tagged 'relative' is corrected.

        Regression for the report where '和益 2024年 1~6月趨勢圖' was extracted as
        relative/6M (last 6 months) instead of absolute 2024 H1, producing a chart
        for the wrong historical window.
        """
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        expected_result = BoundaryResult(
            stock_codes=[],
            company_names=["和益"],
            date_range=DateRange(type=TimeScope.RELATIVE, value="6M"),
            time_scope=TimeScope.RELATIVE,
            confidence=0.9,
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("和益 2024年 1~6月趨勢圖")

        assert result.stock_codes == ["1709"]
        assert result.date_range is not None
        assert result.date_range.type == TimeScope.ABSOLUTE
        assert result.date_range.value == "2024-01-01/2024-06-30"
        assert result.time_scope == TimeScope.ABSOLUTE

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_reconciles_null_date_range_from_year(self, mock_llm_client_class):
        """A year-anchored question whose date_range the LLM left null is fixed.

        The /v1 JSON-mode path of the local model emits ``date_range: null``
        even when the question names a specific year ("南亞2024年上下游名單").
        The deterministic reconciliation must derive the absolute window from
        the question text (gap: data_range never detected).
        """
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        expected_result = BoundaryResult(
            stock_codes=[],
            company_names=["南亞"],
            date_range=None,  # LLM returned null
            time_scope=None,
            confidence=0.9,
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("南亞2024年上下游名單")

        assert result.stock_codes == ["1303"]
        assert result.date_range is not None
        assert result.date_range.type == TimeScope.ABSOLUTE
        assert result.date_range.value == "2024-01-01/2024-12-31"
        assert result.time_scope == TimeScope.ABSOLUTE

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_reconciles_null_date_range_from_bare_year_context(
        self, mock_llm_client_class
    ):
        """A bare 4-digit year + supply-chain context (no 年 suffix) anchors.

        "南亞2024上下游名單" has no "年" character, so the year-anchored regex
        ``\\d{{4}}\\s*年`` alone misses it. Without the deterministic fallback
        the pipeline falls back to the newest annual report (FY 114) and only
        the top supplier survives; with it, FY 113 (南亞 2024) is used and
        every listed supplier (incl. 金益鼎 8390 from 主要原料之供應狀況) shows up.
        """
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        expected_result = BoundaryResult(
            stock_codes=[],
            company_names=["南亞"],
            date_range=None,  # LLM returned null
            time_scope=None,
            confidence=0.9,
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("南亞2024上下游名單")

        assert result.stock_codes == ["1303"]
        assert result.date_range is not None
        assert result.date_range.type == TimeScope.ABSOLUTE
        assert result.date_range.value == "2024-01-01/2024-12-31"
        assert result.time_scope == TimeScope.ABSOLUTE

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_does_not_anchor_bare_stock_code(self, mock_llm_client_class):
        """A 4-digit stock code beside a supply-chain keyword is not a year.

        "台塑1303供應商" could read as a bare-year + context pattern; the
        year-range guard (1990-2040) must reject 1303 so the question keeps
        its (null) range instead of inventing year 1303.
        """
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        expected_result = BoundaryResult(
            stock_codes=[],
            company_names=["台塑"],
            date_range=None,
            time_scope=None,
            confidence=0.9,
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("台塑1303供應商")

        assert result.stock_codes == ["1301"]
        assert result.date_range is None
        assert result.time_scope is None

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_reconciles_null_date_range_from_minguo_year(
        self, mock_llm_client_class
    ):
        """民國年 (ROC-era) forms are reconciled to Gregorian absolute ranges.

        "民國113年" maps to Gregorian 2024. The previous regex required a month
        after a 4-digit year, so both "民國113年度年報" and bare "2024年" missed
        the anchor entirely.
        """
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        expected_result = BoundaryResult(
            stock_codes=[],
            company_names=["台塑"],
            date_range=None,
            time_scope=None,
            confidence=0.9,
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("台塑民國113年度年報揭露的供應商與客戶有哪些")

        assert result.stock_codes == ["1301"]
        assert result.date_range is not None
        assert result.date_range.type == TimeScope.ABSOLUTE
        assert result.date_range.value == "2024-01-01/2024-12-31"
        assert result.time_scope == TimeScope.ABSOLUTE

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_null_date_range_without_year_stays_null(self, mock_llm_client_class):
        """No year anchor in the question -> reconciliation leaves null alone.

        Guards against inventing a window for questions like "台塑現在股價多少".
        """
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        expected_result = BoundaryResult(
            stock_codes=[],
            company_names=["台塑"],
            date_range=None,
            time_scope=None,
            confidence=0.9,
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("台塑現在股價多少")

        assert result.stock_codes == ["1301"]
        assert result.date_range is None
        assert result.time_scope is None

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_sets_chart_data_requirements(self, mock_llm_client_class):
        """Boundary output carries validated chart_data_requirements (修 F).

        The chart decision is derived at Step 1 from question keywords via the
        same ChartValidator logic used on classification, so STEP BOUNDRY shows
        the final chart requirement rather than the LLM's raw guess.
        """
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        # LLM emitted no chart hint at all; reconciliation must derive it.
        expected_result = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.9,
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("台積電股價走勢圖")

        assert result.chart_data_requirements == ChartDataRequirement.PRICE_TREND
        assert result.chart_type == ChartType.LINE

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_corrects_chart_type_to_kline(self, mock_llm_client_class):
        """A wrong LLM chart_type hint is forced to match the requirement (修 F)."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        expected_result = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.9,
            chart_type=ChartType.LINE,  # LLM guessed wrong
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("台積電K線圖")

        assert result.chart_data_requirements == ChartDataRequirement.PRICE_OHLC
        assert result.chart_type == ChartType.KLINE

    @patch("classifier.boundary.LLMClient")
    def test_extract_boundary_clears_chart_fields_without_visual_wording(
        self, mock_llm_client_class
    ):
        """No chart wording -> chart_data_requirements/chart_type cleared (修 F)."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        # LLM wrongly left a chart hint on a non-chart question.
        expected_result = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.9,
            chart_type=ChartType.LINE,
        )
        mock_client.extract_structured.return_value = expected_result

        result = extract_boundary("台積電現在股價多少")

        assert result.chart_data_requirements is None
        assert result.chart_type is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
