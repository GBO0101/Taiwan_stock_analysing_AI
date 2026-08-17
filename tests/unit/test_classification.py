"""Unit tests for Step 2 Query Classification."""

import pytest
from unittest.mock import Mock, patch
from classifier.classification import classify_query, ClassificationError
from classifier.boundary import extract_boundary
from classifier.models import BoundaryResult, ClassificationResult, ClassificationType
from classifier.llm_client import LLMError


class TestClassification:
    """Test query classification functionality."""

    @patch("classifier.classification.LLMClient")
    def test_classify_live_query(self, mock_llm_client_class):
        """Test classification of a live query."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        expected_result = ClassificationResult(
            type=ClassificationType.LIVE,
            confidence=0.9,
            needs_visualization=False,
            target_datasets=["TaiwanStockPrice"],
        )
        mock_client.extract_structured.return_value = expected_result

        result = classify_query("台積電現在股價多少", boundary)

        assert result.type == ClassificationType.LIVE
        assert result.confidence == 0.9
        assert result.needs_visualization is False
        mock_client.extract_structured.assert_called_once()

    @patch("classifier.classification.LLMClient")
    def test_classify_factual_query(self, mock_llm_client_class):
        """Test classification of a factual query."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        expected_result = ClassificationResult(
            type=ClassificationType.FACTUAL,
            confidence=0.85,
            needs_visualization=False,
            target_datasets=["TaiwanStockFinancialStatements"],
        )
        mock_client.extract_structured.return_value = expected_result

        result = classify_query("台積電2024年EPS是多少", boundary)

        assert result.type == ClassificationType.FACTUAL
        assert result.confidence == 0.85

    @patch("classifier.classification.LLMClient")
    def test_classify_analytical_query(self, mock_llm_client_class):
        """Test classification of an analytical query."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        expected_result = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.92,
            needs_visualization=True,
            chart_type="line",
            chart_data_requirements="indicator_trend",
            target_datasets=["TaiwanStockFinancialStatements", "TaiwanStockPER"],
        )
        mock_client.extract_structured.return_value = expected_result

        result = classify_query("台積電近五年獲利能力走勢圖", boundary)

        assert result.type == ClassificationType.ANALYTICAL
        assert result.needs_visualization is True
        assert result.chart_type == "line"
        assert result.chart_data_requirements == "indicator_trend"

    @patch("classifier.classification.LLMClient")
    def test_classify_non_financial_query(self, mock_llm_client_class):
        """Test classification of a non-financial query."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        boundary = BoundaryResult(
            stock_codes=[],
            company_names=[],
            confidence=0.5,
        )
        expected_result = ClassificationResult(
            type=ClassificationType.NON_FINANCIAL,
            confidence=0.9,
            needs_visualization=False,
            target_datasets=[],
        )
        mock_client.extract_structured.return_value = expected_result

        result = classify_query("香蕉好吃嗎", boundary)

        assert result.type == ClassificationType.NON_FINANCIAL
        assert result.target_datasets == []

    @patch("classifier.classification.LLMClient")
    def test_classify_with_clarification(self, mock_llm_client_class):
        """Test classification requesting clarification."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        expected_result = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.6,
            needs_clarification=True,
            clarification_question="請問您想分析哪個時間範圍？",
            needs_visualization=False,
            target_datasets=[],
        )
        mock_client.extract_structured.return_value = expected_result

        result = classify_query("台積電表現如何", boundary)

        assert result.needs_clarification is True
        assert result.clarification_question is not None

    @patch("classifier.classification.LLMClient")
    def test_classify_llm_error(self, mock_llm_client_class):
        """Test classification handles LLM errors."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        mock_client.extract_structured.side_effect = LLMError("API timeout")

        boundary = BoundaryResult(stock_codes=["2330"], confidence=0.9)

        with pytest.raises(ClassificationError, match="Classification failed"):
            classify_query("台積電股價", boundary)

    def test_classify_custom_client(self):
        """Test classification with custom LLM client."""
        mock_client = Mock()
        boundary = BoundaryResult(stock_codes=["2330"], confidence=0.9)
        expected_result = ClassificationResult(
            type=ClassificationType.LIVE,
            confidence=0.9,
        )
        mock_client.extract_structured.return_value = expected_result

        result = classify_query("台積電股價", boundary, llm_client=mock_client)

        assert result.type == ClassificationType.LIVE
        mock_client.extract_structured.assert_called_once()

    @patch("classifier.classification.LLMClient")
    def test_classify_passes_boundary_to_prompt(self, mock_llm_client_class):
        """Test that boundary data is passed to the prompt."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            sectors=["半導體"],
            confidence=0.95,
        )
        expected_result = ClassificationResult(
            type=ClassificationType.LIVE,
            confidence=0.9,
        )
        mock_client.extract_structured.return_value = expected_result

        classify_query("台積電股價", boundary)

        # Verify the prompt was rendered with boundary data
        call_args = mock_client.extract_structured.call_args
        prompt = call_args[0][0]
        assert "2330" in prompt
        assert "台積電" in prompt

    @patch("classifier.classification.LLMClient")
    def test_classify_validator_corrects_price_trend(self, mock_llm_client_class):
        """Validator fixes misclassification: 股價走勢圖 wrongly tagged indicator_trend (gap #2)."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        # LLM wrongly tags a price-trend question as indicator_trend (pre-fix bug)
        expected_result = ClassificationResult(
            type=ClassificationType.LIVE,
            confidence=0.9,
            needs_visualization=True,
            chart_type="line",
            chart_data_requirements="indicator_trend",
        )
        mock_client.extract_structured.return_value = expected_result

        result = classify_query("台積電股價走勢圖", boundary)

        assert result.chart_data_requirements == "price_trend"
        assert result.chart_type == "line"

    @patch("classifier.classification.LLMClient")
    def test_classify_validator_derives_visualization(self, mock_llm_client_class):
        """Validator derives needs_visualization from keywords when LLM misses it (gap #2,#9)."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        boundary = BoundaryResult(stock_codes=["2330"], company_names=["台積電"], confidence=0.95)
        expected_result = ClassificationResult(
            type=ClassificationType.LIVE,
            confidence=0.9,
            needs_visualization=False,  # LLM missed the chart intent
            chart_data_requirements=None,
            chart_type=None,
        )
        mock_client.extract_structured.return_value = expected_result

        result = classify_query("台積電股價走勢圖", boundary)

        assert result.needs_visualization is True
        assert result.chart_data_requirements == "price_trend"

    @patch("classifier.classification.LLMClient")
    def test_classify_multi_stock_comparison(self, mock_llm_client_class):
        """Multi-stock comparison: REVENUE_COMPARISON + bar from a comparison question (gap #15)."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        boundary = BoundaryResult(
            stock_codes=[],
            company_names=["台積電", "鴻海"],  # name-only; resolver fills codes
            confidence=0.95,
        )
        expected_result = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.9,
            needs_visualization=True,
            chart_type="bar",
            chart_data_requirements="revenue_comparison",
        )
        mock_client.extract_structured.return_value = expected_result

        result = classify_query("台積電和鴻海營收比較圖", boundary)

        assert result.needs_visualization is True
        assert result.chart_data_requirements == "revenue_comparison"
        assert result.chart_type == "bar"

    @patch("classifier.classification.LLMClient")
    @patch("classifier.boundary.LLMClient")
    def test_full_multi_stock_comparison(self, mock_boundary_llm, mock_class_llm):
        """End-to-end: name-only multi-stock -> codes resolved -> comparison chart (gap #1, #15)."""
        b_client = Mock()
        mock_boundary_llm.return_value = b_client
        b_client.extract_structured.return_value = BoundaryResult(
            stock_codes=[],
            company_names=["台積電", "鴻海"],
            confidence=0.95,
        )
        c_client = Mock()
        mock_class_llm.return_value = c_client
        c_client.extract_structured.return_value = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.9,
            needs_visualization=True,
            chart_type="bar",
            chart_data_requirements="revenue_comparison",
        )

        boundary = extract_boundary("台積電和鴻海營收比較圖")
        assert boundary.stock_codes == ["2330", "2317"]

        result = classify_query("台積電和鴻海營收比較圖", boundary)
        assert result.chart_data_requirements == "revenue_comparison"
        assert result.chart_type == "bar"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
