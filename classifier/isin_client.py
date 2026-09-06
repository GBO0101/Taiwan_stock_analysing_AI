"""TWSE ISIN publication-list client for Step 4 stock enumeration.

Fetches and parses the TWSE 有價證券品項列表 (isin.twse.com.tw/isin/C_public.jsp),
which lists every listed security with its 產業別 (industry) label. We filter to
4-digit common-stock codes (len(code) == 4) and expose lookups by sector and
all-stocks enumeration. Results are cached locally with a TTL.
"""

from __future__ import annotations

import logging
from typing import Any

import requests
from bs4 import BeautifulSoup

from classifier.models import StockItem
from classifier.stock_cache import get_cache, set_cache

logger = logging.getLogger(__name__)

# strMode: 2 = 上市 (TWSE), 4 = 上櫃 (TPEx/GreTai), 5 = 興櫃 (Emerging)
_MODE_URL = {
    "TWSE": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
    "TPEx": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",
    "EMERGING": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=5",
}

_TIMEOUT = 15
_CACHE_TTL = 7 * 24 * 3600  # 7 days


class IsinClientError(Exception):
    """ISIN listing fetch/parse errors."""


class IsinClient:
    """Client for the TWSE ISIN publication list."""

    def __init__(self, cache_ttl: int = _CACHE_TTL) -> None:
        self.cache_ttl = cache_ttl

    def fetch_listing(self, market: str = "TWSE") -> list[dict[str, Any]]:
        """Return parsed listing rows for a market, using the local cache.

        Args:
            market: "TWSE", "TPEx", or "EMERGING".

        Returns:
            A list of {code, name, market, industry} dicts for 4-digit stocks.

        Raises:
            IsinClientError: If fetch or parse fails and no cache is available.
        """
        cache_key = f"isin_{market}"
        cached = get_cache(cache_key, ttl_seconds=self.cache_ttl)
        if cached is not None:
            logger.debug("ISIN listing for %s served from cache", market)
            return cached  # type: ignore[no-any-return]

        url = _MODE_URL.get(market)
        if url is None:
            raise IsinClientError(f"Unknown market: {market}")

        try:
            resp = requests.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            rows = self._parse(resp.content)
        except (requests.RequestException, ValueError) as e:
            raise IsinClientError(f"Failed to fetch ISIN listing ({market}): {e}") from e

        if not rows:
            raise IsinClientError(f"ISIN listing returned no rows for market {market}")

        set_cache(cache_key, rows)
        return rows

    @staticmethod
    def _parse(raw: bytes) -> list[dict[str, Any]]:
        """Parse the ISIN HTML page into structured rows.

        The page is served in Big5 and contains a table (class="h4") with columns:
        code+name, ISIN, 上市日, 市場別, 產業別, CFICode, 備註. Only rows whose
        first cell splits into a 4-digit code and a name are kept.
        """
        try:
            html = raw.decode("big5", errors="replace")
        except (LookupError, UnicodeDecodeError):
            html = raw.decode("utf-8", errors="replace")

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", class_="h4")
        if table is None:
            raise ValueError("ISIN table not found (page structure changed)")

        rows: list[dict[str, Any]] = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 5:
                continue
            first = cells[0].get_text("\u3000", strip=True)
            if "\u3000" not in first:
                continue
            code_part, name_part = first.split("\u3000", 1)
            code = code_part.strip()
            name = name_part.strip()
            if len(code) != 4 or not code.isdigit():
                continue
            market = cells[3].get_text(strip=True)
            industry = cells[4].get_text(strip=True)
            rows.append({"code": code, "name": name, "market": market, "industry": industry})

        return rows

    def list_all(self, market: str = "TWSE") -> list[StockItem]:
        """Return all 4-digit common stocks as StockItem (code + name)."""
        rows = self.fetch_listing(market)
        return [StockItem(code=r["code"], name=r["name"]) for r in rows]

    def by_sector(self, sector: str, market: str = "TWSE") -> list[StockItem]:
        """Return stocks whose industry label matches ``sector`` (substring)."""
        rows = self.fetch_listing(market)
        matched = [r for r in rows if sector in r["industry"]]
        return [StockItem(code=r["code"], name=r["name"]) for r in matched]


# Global instance
isin_client = IsinClient()
