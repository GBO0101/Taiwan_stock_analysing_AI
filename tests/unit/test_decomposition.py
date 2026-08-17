"""Unit tests for Step 3 Query Decomposition."""

import pytest
from unittest.mock import Mock, patch
from classifier.decomposition import decompose_query, DecompositionError
from classifier.models import (
    BoundaryResult,
    ClassificationResult,
    DecompositionResult,
    ClassificationType,
    SubQuery,
    OperationType,
)
from classifier.llm_client import LLMError


class TestDecomposition:
    """Test query decomposition functionality."""

    @patch("classifier.decomposition.LLMClient")
    def test_decompose_future_outlook(self, mock_llm_client_class):
        """Test decomposition of a future outlook query."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        classification = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.92,
        )
        expected_result = DecompositionResult(
            sub_queries=[
                SubQuery(
                    id="q0",
                    operation=OperationType.FINMIND_QUERY,
                    datasets=["TaiwanStockInfo"],
                    params={"stock_id": "2330"},
                    depends_on=[],
                ),
                SubQuery(
                    id="q1",
                    operation=OperationType.FINMIND_QUERY,
                    datasets=["TaiwanStockNews"],
                    params={"stock_id": "2330", "start_date": "2024-01-01"},
                    depends_on=[],
                ),
                SubQuery(
                    id="q2",
                    operation=OperationType.SENTIMENT_ANALYSIS,
                    datasets=["TaiwanStockNews"],
                    params={},
                    depends_on=["q1"],
                ),
            ]
        )
        mock_client.extract_structured.return_value = expected_result

        result = decompose_query("台積電未來展望", boundary, classification)

        assert len(result.sub_queries) == 3
        assert result.sub_queries[0].id == "q0"
        assert result.sub_queries[0].operation == OperationType.FINMIND_QUERY
        assert result.sub_queries[2].depends_on == ["q1"]

    @patch("classifier.decomposition.LLMClient")
    def test_decompose_gross_margin(self, mock_llm_client_class):
        """Test decomposition of a gross margin calculation query."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        classification = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.92,
        )
        expected_result = DecompositionResult(
            sub_queries=[
                SubQuery(
                    id="q0",
                    operation=OperationType.FINMIND_QUERY,
                    datasets=["TaiwanStockFinancialStatements"],
                    params={"stock_id": "2330", "start_date": "2022-01-01"},
                    depends_on=[],
                ),
                SubQuery(
                    id="q1",
                    operation=OperationType.COMPUTE,
                    computed_field="gross_margin",
                    formula="q0.gross_profit / q0.revenue * 100",
                    depends_on=["q0"],
                ),
            ]
        )
        mock_client.extract_structured.return_value = expected_result

        result = decompose_query("台積電毛利率趨勢", boundary, classification)

        assert len(result.sub_queries) == 2
        assert result.sub_queries[1].operation == OperationType.COMPUTE
        assert result.sub_queries[1].formula == "q0.gross_profit / q0.revenue * 100"

    @patch("classifier.decomposition.LLMClient")
    def test_decompose_llm_error(self, mock_llm_client_class):
        """Test decomposition handles LLM errors."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client
        mock_client.extract_structured.side_effect = LLMError("API timeout")

        boundary = BoundaryResult(stock_codes=["2330"], confidence=0.9)
        classification = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.9,
        )

        with pytest.raises(DecompositionError, match="Decomposition failed"):
            decompose_query("台積電未來展望", boundary, classification)

    def test_decompose_custom_client(self):
        """Test decomposition with custom LLM client."""
        mock_client = Mock()
        boundary = BoundaryResult(stock_codes=["2330"], confidence=0.9)
        classification = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.9,
        )
        expected_result = DecompositionResult(
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
        mock_client.extract_structured.return_value = expected_result

        result = decompose_query(
            "台積電股價趨勢",
            boundary,
            classification,
            llm_client=mock_client,
        )

        assert len(result.sub_queries) == 1
        mock_client.extract_structured.assert_called_once()

    @patch("classifier.decomposition.LLMClient")
    def test_decompose_passes_context_to_prompt(self, mock_llm_client_class):
        """Test that boundary and classification are passed to the prompt."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        classification = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.92,
        )
        expected_result = DecompositionResult(
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
        mock_client.extract_structured.return_value = expected_result

        decompose_query("台積電股價趨勢", boundary, classification)

        call_args = mock_client.extract_structured.call_args
        prompt = call_args[0][0]
        assert "2330" in prompt
        assert "analytical" in prompt

    @patch("classifier.decomposition.LLMClient")
    def test_decompose_with_indicator_mappings(self, mock_llm_client_class):
        """Test decomposition passes indicator mappings to prompt."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        boundary = BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.95,
        )
        classification = ClassificationResult(
            type=ClassificationType.ANALYTICAL,
            confidence=0.92,
        )
        expected_result = DecompositionResult(
            sub_queries=[
                SubQuery(
                    id="q0",
                    operation=OperationType.FINMIND_QUERY,
                    datasets=["TaiwanStockFinancialStatements"],
                    params={"stock_id": "2330"},
                    depends_on=[],
                ),
            ]
        )
        mock_client.extract_structured.return_value = expected_result

        decompose_query("台積電EPS趨勢", boundary, classification)

        call_args = mock_client.extract_structured.call_args
        prompt = call_args[0][0]
        assert "revenue_growth" in prompt or "eps" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
