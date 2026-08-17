"""FinMind API client wrapper."""

from typing import Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from classifier.config import settings


class FinMindError(Exception):
    """Base exception for FinMind API errors."""

    pass


class FinMindAuthError(FinMindError):
    """Authentication errors."""

    pass


class FinMindAPIError(FinMindError):
    """API response errors."""

    def __init__(self, message: str, status_code: int | None = None, response_data: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class FinMindClient:
    """FinMind API client for market data retrieval.

    Used only for chart rendering in this MVP.
    Step 1, Step 2, and Step 3 must never call this client.
    """

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize FinMind client.

        Args:
            api_token: FinMind API token (defaults to settings)
            base_url: FinMind API base URL (defaults to settings)
        """
        self.api_token = api_token or settings.finmind_api_token
        self.base_url = base_url or settings.finmind_base_url

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        })

    def _request(self, dataset: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Make a request to FinMind API.

        Args:
            dataset: Dataset name
            params: Query parameters

        Returns:
            List of data records as dictionaries

        Raises:
            FinMindAuthError: Authentication failure
            FinMindAPIError: API error response
        """
        url = f"{self.base_url}/data"
        request_params = {"dataset": dataset, **params}

        try:
            response = self._session.get(url, params=request_params, timeout=30)
        except requests.exceptions.Timeout as e:
            raise FinMindAPIError(f"Request timeout: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise FinMindAPIError(f"Connection error: {e}") from e
        except requests.exceptions.RequestException as e:
            raise FinMindAPIError(f"Request failed: {e}") from e

        if response.status_code == 401:
            raise FinMindAuthError("Invalid or expired API token")
        elif response.status_code == 403:
            raise FinMindAuthError("Access forbidden - check token permissions")
        elif response.status_code >= 400:
            try:
                error_data = response.json()
                msg = error_data.get("msg", f"HTTP {response.status_code}")
            except Exception:
                msg = f"HTTP {response.status_code}"
            raise FinMindAPIError(msg, status_code=response.status_code, response_data=error_data if 'error_data' in locals() else None)

        try:
            data = response.json()
        except Exception as e:
            raise FinMindAPIError(f"Invalid JSON response: {e}") from e

        if data.get("status") != "200":
            msg = data.get("msg", "Unknown API error")
            raise FinMindAPIError(msg, status_code=response.status_code, response_data=data)

        return data.get("data", [])

    def get_stock_info(self, stock_id: str | None = None) -> list[dict[str, Any]]:
        """Get Taiwan stock basic information.

        Args:
            stock_id: Optional stock ID filter

        Returns:
            List of stock info records
        """
        params = {}
        if stock_id:
            params["stock_id"] = stock_id
        return self._request("TaiwanStockInfo", params)

    def get_stock_price(
        self,
        stock_id: str,
        start_date: str,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get Taiwan stock price data.

        Args:
            stock_id: Stock ID (e.g., "2330")
            start_date: Start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format

        Returns:
            List of price records
        """
        params = {
            "stock_id": stock_id,
            "start_date": start_date,
        }
        if end_date:
            params["end_date"] = end_date
        return self._request("TaiwanStockPrice", params)

    def get_stock_news(
        self,
        stock_id: str,
        start_date: str,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get Taiwan stock news.

        Args:
            stock_id: Stock ID
            start_date: Start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format

        Returns:
            List of news records
        """
        params = {
            "stock_id": stock_id,
            "start_date": start_date,
        }
        if end_date:
            params["end_date"] = end_date
        return self._request("TaiwanStockNews", params)

    def get_stock_per(
        self,
        stock_id: str,
        start_date: str,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get Taiwan stock PER (Price-Earnings Ratio) data.

        Args:
            stock_id: Stock ID
            start_date: Start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format

        Returns:
            List of PER records
        """
        params = {
            "stock_id": stock_id,
            "start_date": start_date,
        }
        if end_date:
            params["end_date"] = end_date
        return self._request("TaiwanStockPER", params)

    def get_month_revenue(
        self,
        stock_id: str,
        start_date: str,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get Taiwan stock monthly revenue.

        Args:
            stock_id: Stock ID
            start_date: Start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format

        Returns:
            List of monthly revenue records
        """
        params = {
            "stock_id": stock_id,
            "start_date": start_date,
        }
        if end_date:
            params["end_date"] = end_date
        return self._request("TaiwanStockMonthRevenue", params)

    def get_institutional_investors(
        self,
        stock_id: str,
        start_date: str,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get Taiwan stock institutional investors buy/sell data.

        Args:
            stock_id: Stock ID
            start_date: Start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format

        Returns:
            List of institutional investor records
        """
        params = {
            "stock_id": stock_id,
            "start_date": start_date,
        }
        if end_date:
            params["end_date"] = end_date
        return self._request("TaiwanStockInstitutionalInvestorsBuySell", params)

    def get_financial_statements(
        self,
        stock_id: str,
        start_date: str,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get Taiwan stock financial statements.

        Args:
            stock_id: Stock ID
            start_date: Start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format

        Returns:
            List of financial statement records
        """
        params = {
            "stock_id": stock_id,
            "start_date": start_date,
        }
        if end_date:
            params["end_date"] = end_date
        return self._request("TaiwanStockFinancialStatements", params)

    def get_cash_flow_statement(
        self,
        stock_id: str,
        start_date: str,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get Taiwan stock cash flow statement.

        Args:
            stock_id: Stock ID
            start_date: Start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format

        Returns:
            List of cash flow statement records
        """
        params = {
            "stock_id": stock_id,
            "start_date": start_date,
        }
        if end_date:
            params["end_date"] = end_date
        return self._request("TaiwanStockCashFlowsStatement", params)