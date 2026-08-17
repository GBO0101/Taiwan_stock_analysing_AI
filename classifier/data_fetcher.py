"""Free, key-less Taiwan market data fetcher using TWSE open APIs.

Replaces FinMind for chart rendering so the pipeline works without a paid
FinMind account. Covers price (STOCK_DAY) and valuation/PE (BWIBBU). Monthly
revenue is NOT available from the free TWSE/MOPS endpoints and raises a clear
error instead.
"""

from datetime import date, datetime
import re
from typing import Any

import requests


class FreeDataError(Exception):
    """Raised when free data cannot be retrieved."""

    pass


def _parse_input_date(value: str | date) -> date:
    """Accept 'YYYY-MM-DD' strings or date objects."""
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_roc_slash(value: str) -> date | None:
    """Parse '113/01/02' (ROC year) into a date."""
    m = re.match(r"(\d+)/(\d+)/(\d+)", str(value))
    if not m:
        return None
    return date(int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3)))


def _parse_roc_chinese(value: str) -> date | None:
    """Parse '113年01月02日' (ROC year) into a date."""
    m = re.search(r"(\d+)年(\d+)月(\d+)日", str(value))
    if not m:
        return None
    return date(int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3)))


def _to_float(value: Any) -> float | None:
    """Parse a numeric string, tolerating commas/whitespace/dashes."""
    if value is None:
        return None
    s = str(value).replace(",", "").strip()
    if s in ("", "-", "N/A", "NaN"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


class FreeDataFetcher:
    """Fetches Taiwan stock data from free TWSE endpoints (no API key)."""

    STOCK_DAY_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
    BWIBBU_URL = "https://www.twse.com.tw/exchangeReport/BWIBBU"

    def __init__(self, timeout: int = 30):
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0"})

    def _month_range(self, start: date, end: date):
        """Yield (year, month) tuples from start to end inclusive."""
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            yield y, m
            m += 1
            if m > 12:
                m = 1
                y += 1

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._session.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise FreeDataError(f"Request timeout: {e}") from e
        except requests.exceptions.RequestException as e:
            raise FreeDataError(f"Request failed: {e}") from e
        try:
            return resp.json()
        except Exception as e:
            raise FreeDataError(f"Invalid JSON response: {e}") from e

    def get_stock_price(
        self,
        stock_id: str,
        start_date: str,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get Taiwan stock OHLCV data from TWSE STOCK_DAY.

        Returns a list of dicts with keys: date, open, high, low, close, volume.
        """
        start = _parse_input_date(start_date)
        end = _parse_input_date(end_date) if end_date else date.today()
        result: list[dict[str, Any]] = []
        seen: set[date] = set()
        for y, m in self._month_range(start, end):
            params = {
                "response": "json",
                "date": f"{y}{m:02d}01",
                "stockNo": stock_id,
            }
            j = self._get_json(self.STOCK_DAY_URL, params)
            if j.get("stat") != "OK":
                continue
            for row in j.get("data", []):
                if len(row) < 7:
                    continue
                d = _parse_roc_slash(row[0])
                if d is None or not (start <= d <= end) or d in seen:
                    continue
                seen.add(d)
                result.append(
                    {
                        "date": d.isoformat(),
                        "open": _to_float(row[3]),
                        "high": _to_float(row[4]),
                        "low": _to_float(row[5]),
                        "close": _to_float(row[6]),
                        "volume": int(_to_float(row[1]) or 0),
                    }
                )
        return result

    def get_stock_per(
        self,
        stock_id: str,
        start_date: str,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get Taiwan stock PE ratio history from TWSE BWIBBU.

        Returns a list of dicts with keys: date, pe.
        """
        start = _parse_input_date(start_date)
        end = _parse_input_date(end_date) if end_date else date.today()
        result: list[dict[str, Any]] = []
        seen: set[date] = set()
        for y, m in self._month_range(start, end):
            params = {
                "response": "json",
                "date": f"{y}{m:02d}01",
                "stockNo": stock_id,
            }
            j = self._get_json(self.BWIBBU_URL, params)
            if j.get("stat") != "OK":
                continue
            for row in j.get("data", []):
                if len(row) < 4:
                    continue
                d = _parse_roc_chinese(row[0])
                if d is None or not (start <= d <= end) or d in seen:
                    continue
                seen.add(d)
                pe = _to_float(row[3])
                if pe is None:
                    continue
                result.append({"date": d.isoformat(), "pe": pe})
        return result

    def get_month_revenue(
        self,
        stock_id: str,
        start_date: str,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Monthly revenue is NOT available from the free TWSE/MOPS endpoints."""
        raise FreeDataError(
            "月營收資料需 FinMind 付費帳號（MOPS 免費源被擋），免費 TWSE 源無法取得"
        )
