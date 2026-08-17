"""Unit tests for FinMindClient."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from classifier.finmind_client import (
    FinMindClient,
    FinMindError,
    FinMindAuthError,
    FinMindAPIError,
)


class TestFinMindClient:
    """Test FinMindClient functionality."""

    def test_init_with_defaults(self):
        """Test initialization with default settings."""
        client = FinMindClient()
        assert client.api_token == "test_token_for_testing"
        assert client.base_url == "https://api.finmindtrade.com/api/v4"
        assert "Authorization" in client._session.headers
        assert client._session.headers["Authorization"] == "Bearer test_token_for_testing"

    def test_init_with_overrides(self):
        """Test initialization with custom parameters."""
        client = FinMindClient(api_token="custom_token", base_url="https://custom.api")
        assert client.api_token == "custom_token"
        assert client.base_url == "https://custom.api"
        assert client._session.headers["Authorization"] == "Bearer custom_token"

    @patch("classifier.finmind_client.requests.Session")
    def test_request_success(self, mock_session_class):
        """Test successful API request."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "200",
            "msg": "success",
            "data": [{"stock_id": "2330", "stock_name": "台積電"}],
        }
        mock_session.get.return_value = mock_response

        client = FinMindClient()
        result = client._request("TaiwanStockInfo", {"stock_id": "2330"})

        assert result == [{"stock_id": "2330", "stock_name": "台積電"}]
        mock_session.get.assert_called_once()

    @patch("classifier.finmind_client.requests.Session")
    def test_request_auth_error_401(self, mock_session_class):
        """Test 401 authentication error."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 401
        mock_session.get.return_value = mock_response

        client = FinMindClient()
        with pytest.raises(FinMindAuthError, match="Invalid or expired API token"):
            client._request("TaiwanStockInfo", {})

    @patch("classifier.finmind_client.requests.Session")
    def test_request_auth_error_403(self, mock_session_class):
        """Test 403 forbidden error."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 403
        mock_session.get.return_value = mock_response

        client = FinMindClient()
        with pytest.raises(FinMindAuthError, match="Access forbidden"):
            client._request("TaiwanStockInfo", {})

    @patch("classifier.finmind_client.requests.Session")
    def test_request_api_error(self, mock_session_class):
        """Test API error response."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"msg": "Invalid parameter"}
        mock_session.get.return_value = mock_response

        client = FinMindClient()
        with pytest.raises(FinMindAPIError, match="Invalid parameter"):
            client._request("TaiwanStockInfo", {})

    @patch("classifier.finmind_client.requests.Session")
    def test_request_status_not_200(self, mock_session_class):
        """Test API response with status != 200."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "400",
            "msg": "Dataset not found",
            "data": [],
        }
        mock_session.get.return_value = mock_response

        client = FinMindClient()
        with pytest.raises(FinMindAPIError, match="Dataset not found"):
            client._request("InvalidDataset", {})

    @patch("classifier.finmind_client.requests.Session")
    def test_request_timeout(self, mock_session_class):
        """Test request timeout."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get.side_effect = requests.exceptions.Timeout("Request timed out")

        client = FinMindClient()
        with pytest.raises(FinMindAPIError, match="Request timeout"):
            client._request("TaiwanStockInfo", {})

    @patch("classifier.finmind_client.requests.Session")
    def test_request_connection_error(self, mock_session_class):
        """Test connection error."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = FinMindClient()
        with pytest.raises(FinMindAPIError, match="Connection error"):
            client._request("TaiwanStockInfo", {})

    @patch("classifier.finmind_client.requests.Session")
    def test_get_stock_info(self, mock_session_class):
        """Test get_stock_info method."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "200",
            "data": [{"stock_id": "2330", "stock_name": "台積電", "industry_category": "半導體業"}],
        }
        mock_session.get.return_value = mock_response

        client = FinMindClient()
        result = client.get_stock_info("2330")

        assert len(result) == 1
        assert result[0]["stock_id"] == "2330"
        assert result[0]["stock_name"] == "台積電"

    @patch("classifier.finmind_client.requests.Session")
    def test_get_stock_price(self, mock_session_class):
        """Test get_stock_price method."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "200",
            "data": [
                {"stock_id": "2330", "date": "2024-01-01", "open": 500, "close": 510, "high": 520, "low": 495, "volume": 1000000},
            ],
        }
        mock_session.get.return_value = mock_response

        client = FinMindClient()
        result = client.get_stock_price("2330", "2024-01-01", "2024-01-31")

        assert len(result) == 1
        assert result[0]["close"] == 510

    @patch("classifier.finmind_client.requests.Session")
    def test_get_stock_news(self, mock_session_class):
        """Test get_stock_news method."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "200",
            "data": [{"stock_id": "2330", "date": "2024-01-01", "title": "台積電新聞"}],
        }
        mock_session.get.return_value = mock_response

        client = FinMindClient()
        result = client.get_stock_news("2330", "2024-01-01")

        assert len(result) == 1
        assert result[0]["title"] == "台積電新聞"

    @patch("classifier.finmind_client.requests.Session")
    def test_get_stock_per(self, mock_session_class):
        """Test get_stock_per method."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "200",
            "data": [{"stock_id": "2330", "date": "2024-01-01", "pe_ratio": 15.5}],
        }
        mock_session.get.return_value = mock_response

        client = FinMindClient()
        result = client.get_stock_per("2330", "2024-01-01")

        assert len(result) == 1
        assert result[0]["pe_ratio"] == 15.5

    @patch("classifier.finmind_client.requests.Session")
    def test_get_month_revenue(self, mock_session_class):
        """Test get_month_revenue method."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "200",
            "data": [{"stock_id": "2330", "revenue": 1000000000, "date": "2024-01"}],
        }
        mock_session.get.return_value = mock_response

        client = FinMindClient()
        result = client.get_month_revenue("2330", "2024-01-01")

        assert len(result) == 1
        assert result[0]["revenue"] == 1000000000

    @patch("classifier.finmind_client.requests.Session")
    def test_get_institutional_investors(self, mock_session_class):
        """Test get_institutional_investors method."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "200",
            "data": [{"stock_id": "2330", "date": "2024-01-01", "buy": 10000, "sell": 5000}],
        }
        mock_session.get.return_value = mock_response

        client = FinMindClient()
        result = client.get_institutional_investors("2330", "2024-01-01")

        assert len(result) == 1
        assert result[0]["buy"] == 10000

    @patch("classifier.finmind_client.requests.Session")
    def test_get_financial_statements(self, mock_session_class):
        """Test get_financial_statements method."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "200",
            "data": [{"stock_id": "2330", "date": "2024-03-31", "eps": 10.5}],
        }
        mock_session.get.return_value = mock_response

        client = FinMindClient()
        result = client.get_financial_statements("2330", "2024-01-01")

        assert len(result) == 1
        assert result[0]["eps"] == 10.5

    @patch("classifier.finmind_client.requests.Session")
    def test_get_cash_flow_statement(self, mock_session_class):
        """Test get_cash_flow_statement method."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "200",
            "data": [{"stock_id": "2330", "date": "2024-03-31", "free_cash_flow": 500000000}],
        }
        mock_session.get.return_value = mock_response

        client = FinMindClient()
        result = client.get_cash_flow_statement("2330", "2024-01-01")

        assert len(result) == 1
        assert result[0]["free_cash_flow"] == 500000000

    def test_exception_hierarchy(self):
        """Test exception class hierarchy."""
        assert issubclass(FinMindAuthError, FinMindError)
        assert issubclass(FinMindAPIError, FinMindError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])