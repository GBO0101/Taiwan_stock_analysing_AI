"""Unit tests for Step 1 Boundary Extraction."""

import pytest
from unittest.mock import Mock, patch
from classifier.boundary import extract_boundary, BoundaryExtractionError
from classifier.models import BoundaryResult
from classifier.llm_client import LLMError


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
            "last_boundary": {"stock_codes": ["2330"], "company_names": ["台積電"]}
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])