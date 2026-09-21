"""TWSE electronic document query client (doc.twse.com.tw) for Step 4.

Downloads annual-report PDFs containing major customer/supplier data.
The T57SB01 API uses form-encoded POSTs with Minguo-year dating and Big5
encoded HTML responses. SSL certificates on doc.twse are often stale, so
verification is disabled for that host only.
"""

from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

from classifier.stock_cache import get_cache, set_cache

logger = logging.getLogger(__name__)

_BASE_URL = "https://doc.twse.com.tw/server-java/t57sb01"
_CACHE_TTL = 7 * 24 * 3600  # 7 days
_TIMEOUT = 30

# PDF filename pattern in step-1 results and step-9 redirect HTML
_PDF_RE = re.compile(r"/pdf/([^\"'<>\s]+\.pdf)", re.IGNORECASE)


class DocTwseClientError(Exception):
    """doc.twse.com.tw fetch/parse errors."""


class DocTwseClient:
    """Client for the TWSE Electronic Document Query System (T57SB01).

    Supports:
    - Searching for annual reports (mtype=F, dtype=F04/F18) and financial
      reports (mtype=A, dtype=AI1).
    - Downloading the matched PDF files.
    """

    def __init__(self, timeout: int = _TIMEOUT) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        # Suppress only the single InsecureRequestWarning from urllib3
        requests.packages.urllib3.disable_warnings(  # type: ignore[attr-defined]
            InsecureRequestWarning
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _post_form(self, **fields: str | int) -> str:
        """POST to T57SB01 and return the decoded HTML body (Big5)."""
        payload = {k: str(v) for k, v in fields.items()}
        try:
            resp = self._session.post(
                _BASE_URL,
                data=payload,
                timeout=self._timeout,
                verify=False,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise DocTwseClientError(f"doc.twse timeout: {e}") from e
        except requests.exceptions.RequestException as e:
            raise DocTwseClientError(f"doc.twse request failed: {e}") from e

        return resp.content.decode("big5", errors="replace")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def search_reports(
        self,
        co_id: str,
        year: int,
        mtype: str = "F",
        dtype: str = "F04",
    ) -> list[dict[str, str]]:
        """Search for available reports for a given company/year/type.

        Args:
            co_id:   4-digit stock code (e.g. "2330").
            year:    Minguo year (e.g. 113 for 2024).
            mtype:   Report category ("F" = 股東會相關, "A" = 財務報告書).
            dtype:   Document type within category
                     ("F04" = 股東會年報, "F18" = 含合併財報,
                      "AI1" = IFRSs 合併報表).

        Returns:
            List of {"filename": str, "name": str} dicts.
            Empty list if no reports found.
        """
        cache_key = f"doc_twse_{co_id}_{year}_{mtype}_{dtype}"
        cached = get_cache(cache_key, ttl_seconds=_CACHE_TTL)
        if cached is not None:
            logger.debug("doc.twse search for %s served from cache", cache_key)
            return cached  # type: ignore[no-any-return]

        html = self._post_form(
            step=1,
            co_id=co_id,
            year=year,
            mtype=mtype,
            dtype=dtype,
            id="",
            key="",
            seamon="",
        )

        entries = self._parse_search_results(html)

        if entries:
            set_cache(cache_key, entries)
        return entries

    def has_report(
        self,
        co_id: str,
        year: int,
        mtype: str = "F",
        dtype: str = "F04",
    ) -> bool:
        """Return True if at least one matching report exists."""
        return bool(self.search_reports(co_id, year, mtype, dtype))

    def get_pdf_url(
        self,
        co_id: str,
        year: int,
        mtype: str,
        filename: str,
    ) -> str:
        """Resolve the timestamped PDF download URL via step=9.

        Args:
            co_id:    4-digit stock code.
            year:     Minguo year.
            mtype:    Report category ("F" or "A").
            filename: PDF filename returned by :meth:`search_reports`.

        Returns:
            Full HTTPS URL to the PDF file.

        Raises:
            DocTwseClientError: If the URL cannot be resolved.
        """
        html = self._post_form(
            step=9,
            kind=mtype,
            co_id=co_id,
            filename=filename,
            id="",
            key="",
            seamon="",
        )

        match = _PDF_RE.search(html)
        if not match:
            raise DocTwseClientError(
                f"Could not extract PDF URL from step-9 response for {filename}"
            )

        path = match.group(1)
        url = f"https://doc.twse.com.tw/pdf/{path}"
        logger.debug("Resolved PDF URL: %s", url)
        return url

    def download_pdf(self, pdf_url: str) -> bytes:
        """Download a PDF and return its raw bytes.

        Raises:
            DocTwseClientError: On network error or if content is not a PDF.
        """
        try:
            resp = self._session.get(pdf_url, timeout=self._timeout, verify=False)
            resp.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise DocTwseClientError(f"PDF download timeout: {e}") from e
        except requests.exceptions.RequestException as e:
            raise DocTwseClientError(f"PDF download failed: {e}") from e

        data = resp.content
        if not data or not data.lstrip().startswith(b"%PDF"):
            raise DocTwseClientError(
                f"Downloaded content is not a valid PDF ({len(data)} bytes)"
            )
        return data

    def fetch_pdf(
        self,
        co_id: str,
        year: int,
        mtype: str = "F",
        dtype: str = "F04",
    ) -> tuple[bytes, str]:
        """Convenience: search + resolve URL + download in one call.

        Returns:
            (pdf_bytes, pdf_url) tuple.

        Raises:
            DocTwseClientError: If no report found or download fails.
        """
        entries = self.search_reports(co_id, year, mtype, dtype)
        if not entries:
            raise DocTwseClientError(
                f"No reports found for {co_id} year={year} mtype={mtype} dtype={dtype}"
            )
        filename = entries[0]["filename"]
        url = self.get_pdf_url(co_id, year, mtype, filename)
        return self.download_pdf(url), url

    # ------------------------------------------------------------------
    # HTML parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_search_results(html: str) -> list[dict[str, str]]:
        """Parse step-1 HTML response and extract PDF filenames.

        Looks for any ``*.pdf`` text in the page body.  Falls back to
        extracting from ``<a href="...">`` links if no plain-text match.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Check for "no results" indicator (varies by page)
        body_text = soup.get_text()
        if "查無所需資料" in body_text or "無符合條件之資料" in body_text:
            return []

        # Strategy 1: find all text nodes matching a PDF filename pattern
        # Filenames look like: 2023_2330_20240604F04.pdf
        pdf_pattern = re.compile(
            r"\d{4,}[_\-]\d{4,}[_\-]\d{6,}.*?\.pdf", re.IGNORECASE
        )
        found: list[str] = []
        for text_node in soup.find_all(string=True):
            for m in pdf_pattern.finditer(str(text_node)):
                candidate = m.group(0).strip()
                if candidate not in found:
                    found.append(candidate)

        # Strategy 2: extract from anchor hrefs
        if not found:
            for a in soup.find_all("a", href=True):
                href = str(a["href"])
                if href.lower().endswith(".pdf") and any(
                    c.isdigit() for c in href
                ):
                    # Strip any leading path segments
                    name = href.rsplit("/", 1)[-1]
                    if name not in found:
                        found.append(name)

        # Build structured entries
        entries: list[dict[str, str]] = []
        for fname in found:
            entries.append({"filename": fname, "name": fname})
        return entries


# Global instance
doc_twse_client = DocTwseClient()
