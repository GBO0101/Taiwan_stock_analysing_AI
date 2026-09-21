"""Annual-report PDF extraction for Step 4 supply-chain evidence.

Downloads a company's annual report (股東會年報) from doc.twse.com.tw,
extracts text with PyMuPDF, sends it to the LLM for structured extraction
of major customers/suppliers, and resolves raw names to listed stock codes.

This module does NOT call FinMind — it is a pure doc.twse + LLM path.
"""

from __future__ import annotations

import io
import logging
import re

from classifier.doc_twse_client import DocTwseClient, DocTwseClientError
from classifier.doc_twse_client import doc_twse_client as default_doc_client
from classifier.llm_client import LLMClient, LLMError
from classifier.models import (
    AnnualReportDisclosure,
    CounterpartyDisclosure,
)
from classifier.prompts import prompt_manager
from classifier.stock_cache import CacheError, get_cache, set_cache
from classifier.stock_resolver import StockResolver
from classifier.stock_resolver import stock_resolver as default_resolver

logger = logging.getLogger(__name__)

# Attempt up to 2 years back when the target year has no report yet.
_MAX_FALLBACK_YEARS = 2

# Annual reports change once a year; a 30-day cache is plenty.
_DISCLOSURE_CACHE_TTL = 30 * 24 * 3600


class AnnualReportError(Exception):
    """Annual-report extraction errors."""


# ---------------------------------------------------------------------------
# Internal Pydantic model for LLM output validation
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field, ValidationError


class _LLMCounterpartyEntry(BaseModel):
    """Single customer/supplier entry extracted by the LLM."""

    raw_name: str = Field(..., description="Original name from annual report")
    ratio: float | None = Field(default=None, description="Revenue/purchase %")
    is_related_party: bool = Field(default=False)
    is_anonymous: bool = Field(default=False)
    note: str | None = Field(default=None)


class _AnnualReportExtract(BaseModel):
    """Full LLM extraction result from annual-report text."""

    customers: list[_LLMCounterpartyEntry] = Field(default_factory=list)
    suppliers: list[_LLMCounterpartyEntry] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    fiscal_year: int = Field(..., description="ROC fiscal year")
    notes: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# Text extraction (PyMuPDF)
# ---------------------------------------------------------------------------


_MAX_PDF_PAGES = 400  # 股東會年報常見 150-400 頁；客戶/供應商章節多在後半部


def _extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = _MAX_PDF_PAGES) -> str:
    """Extract text from a PDF using PyMuPDF.

    Scans up to ``max_pages`` pages and concatenates all text. The default
    covers the whole annual report: customer/supplier disclosure tables sit
    in the second half (around page 140-160), so capping too low (e.g. 80)
    silently starves the LLM. Returns an empty string on failure (never
    raises — caller handles gracefully).
    """
    try:
        import fitz  # type: ignore[import-untyped]  # PyMuPDF

        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        pages = min(len(doc), max_pages)
        parts: list[str] = []
        for i in range(pages):
            page = doc[i]
            parts.append(page.get_text())
        doc.close()
        return "\n".join(parts)
    except ImportError:
        logger.warning("PyMuPDF (fitz) not installed — cannot extract PDF text")
        return ""
    except (ValueError, OSError, TypeError, RuntimeError) as e:
        # RuntimeError covers pymupdf.EmptyFileError / FileDataError
        # (both subclass RuntimeError) raised for empty/corrupt streams.
        logger.warning("PDF text extraction failed: %s", e)
        return ""


# Regulatory disclosure headers that unambiguously mark the customer/supplier
# tables in a TWSE annual report (年報應行記載事項準則). The exact legal title
# varies by company, e.g. 「最近二年度任一年度曾佔進(銷)貨總額百分之十以上之
# 供應商/客戶資訊」 or 「前十大客戶之名稱及銷貨金額、比例」.
# 主要原料之供應狀況/主要供應廠商: per-raw-material supplier lists (much
# wider than the >10%-of-purchases disclosure table — includes non-listed and
# offshore counterparties too).
_SECTION_HEADER_RE = re.compile(
    r"(供應商/客戶資訊|客戶/供應商資訊|進\(?銷\)?貨總額百分之十以上|"
    r"前十(大|名)客戶|前五大客戶|主要客戶|"
    r"前十(大|名)供應商|前五大供應商|主要供應商|主要供應廠商|"
    r"關聯企業及主要交易往來|與關係人進銷貨|"
    r"佔?全年度(進|銷)貨淨額|進貨淨額|銷貨淨額|"
    r"主要原料之供應狀況|原料之使用狀況)",
    re.IGNORECASE,
)

# Window padding around each matched header (200 chars before, 4000 after).
_SECTION_WINDOW_BEFORE = 200
_SECTION_WINDOW_AFTER = 4000

# Sticky headers = the actual disclosure tables. Windows carrying them get
# evicted last when the merged output would overflow the token budget.
_TABLE_HEADER_RE = re.compile(
    r"(供應商/客戶資訊|客戶/供應商資訊|前十(大|名)客戶|前五大客戶|前十大客戶|"
    r"前十(大|名)供應商|前五大供應商|主要客戶|主要供應商|主要供應廠商|"
    r"主要原料之供應狀況|與關係人進銷貨)"
)

# Roughly 15k tokens — safe for local LLMs on Windows.
_MAX_EXTRACT_CHARS = 50000

# Per-segment LLM extraction retries. Local LLMs intermittently fail JSON-mode
# validation (e.g. ``is_related_party: null``) or emit an empty extraction;
# both are retryable, and splitting the report into segments makes each call
# short enough to be read whole.
_MAX_SEGMENT_RETRIES = 3

# Independent extraction passes per segment, unioned afterwards. Even after
# splitting, a single LLM call still stochastically drops table rows (南亞 113
# measured: the same segment yielded 13 vs 3 suppliers across runs). Unioning
# a few passes converges: each pass loses a *different* subset, so the union
# of pass1+pass2 already covered the full 13-row table. Passes stop early once
# a pass adds no new (name, note) rows.
_MAX_SEGMENT_PASSES = 3

# Group A: regulatory customer/supplier disclosure tables (供應商/客戶資訊,
# 前十大客戶/供應商). Group B: the per-raw-material supply table
# (主要原料之供應狀況). Grouping the located spans by *table kind* keeps each
# prompt short enough for local LLMs to read the whole table — merging both
# kinds into one prompt made local LLMs routinely drop the raw-material rows
# (南亞 113 案例：銅線行 金益鼎企業、鏶鑫企業 在合併視窗中被忽略).
_DISCLOSURE_SEGMENT_RE = re.compile(
    r"(供應商/客戶資訊|客戶/供應商資訊|前十(大|名)客戶|前五大客戶|主要客戶|"
    r"前十(大|名)供應商|前五大供應商|主要供應商|關聯企業及主要交易往來|"
    r"與關係人進銷貨|佔?全年度(進|銷)貨淨額|進貨淨額|銷貨淨額)"
)
_RAW_MATERIAL_SEGMENT_RE = re.compile(r"(主要原料之供應狀況|主要供應廠商|原料之使用狀況)")

# Rows of either table carry at least one of these; a segment without any of
# them (pure prose, e.g. 「本公司並無占個體銷貨總額百分之十以上之客戶」)
# legitimately extracts to an empty list — don't retry those. (全形逗號刻意
# 排除：散文太常見，誤判會讓空結果白燒 retry.)
_TABLE_ROW_MARK_RE = re.compile(r"(公噸|公斤|千元|、|%)")

# Enumeration separators inside a supplier cell: local LLMs routinely merge
# 「台橡、奇美、ASAHI、SIBUR(LRD)、中石化」 into one raw_name despite the
# prompt telling them to split on 、/，; this split is the deterministic
# fallback (applied in ``_resolve_entries``).
_ENUMERATION_SPLIT_RE = re.compile(r"[、，,，,;；]")

# Names are CJK tokens separated by whitespace/punct in the (flattened)
# raw-material table text; used by the deterministic backstop scan.
_NAME_TOKEN_RE = re.compile(r"[、，,，,;；（）()\[\]「」『』、\s]+")
_HAS_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# A raw-material supply-table data row always carries a quantity/amount unit
# (公噸/公斤 for QTY, 千元 for AMOUNT). Prose paragraphs in the same segment
# (market outlook, 說明…) do not — e.g. 「中國大陸」 in the business-plan prose
# must not be resolved into a listed stock (台化 113: 大陸→2526 false pull).
# The backstop scan is therefore restricted to unit-bearing lines only.
_DATA_ROW_UNIT_RE = re.compile(r"(公噸|公斤|千元)")


def _locate_customer_supplier_sections(full_text: str) -> str:
    """Best-effort extraction of just the customer/supplier tables.

    Searches for the regulatory disclosure headers (e.g. 「供應商/客戶資訊」,
    「前十大客戶」, 「與關係人進銷貨」) and returns merged windows of text around
    them. Overlapping/adjacent windows are coalesced so the output stays
    compact. Falls back to the full text if no header is found.
    """
    matches = list(_SECTION_HEADER_RE.finditer(full_text))
    if not matches:
        return full_text  # no header found, return full text

    # Take a generous window around each match, then merge overlaps so a
    # cluster of headers (e.g. 「前十大客戶」+「前十大供應商」) yields one span.
    spans: list[tuple[int, int]] = []
    for m in matches:
        start = max(0, m.start() - _SECTION_WINDOW_BEFORE)
        end = min(len(full_text), m.end() + _SECTION_WINDOW_AFTER)
        spans.append((start, end))

    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Windows containing the sticky disclosure-table headers rank first, so
    # overflow never evicts the actual tables in favour of generic 進貨/銷貨
    # mentions from the financial statements.
    ordered = sorted(
        merged,
        key=lambda span: (
            0 if _TABLE_HEADER_RE.search(full_text[span[0] : span[1]]) else 1,
            span[0],
        ),
    )

    parts: list[str] = []
    used = 0
    for start, end in ordered:
        if used >= _MAX_EXTRACT_CHARS:
            break
        chunk = full_text[start:end]
        room = _MAX_EXTRACT_CHARS - used
        if len(chunk) > room:
            chunk = chunk[:room]
        parts.append(chunk)
        used += len(chunk)

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def _flatten_table_lines(text: str, join_len: int = 28) -> str:
    """Re-flow table cells that PyMuPDF split onto separate lines.

    `page.get_text()` often emits each PDF table cell on its own line (e.g.
    ``"1 "``, ``"台塑石化"``, ``"53,578,492"``, ``"33.44"``, ``"註3"`` on five
    consecutive lines), which local LLMs fail to read as a row. This helper
    joins each short, non-sentence-ending line onto the previous line so the
    row becomes ``"1 台塑石化 53,578,492 33.44 註3"``. Long prose lines
    (> ``join_len`` chars) and lines ending in sentence punctuation start
    new lines as usual.
    """
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            continue
        ends_sentence = line[-1] in "。！？；：.,:;"
        if out and out[-1] and len(line) <= join_len and not ends_sentence:
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return "\n".join(out)


def _split_report_segments(focused: str) -> list[str]:
    """Partition the located section text into independent prompt segments.

    The merged ``focused`` text interleaves two table kinds: the regulatory
    disclosure tables (group A) and the per-raw-material supplier table
    (group B, 主要原料之供應狀況). Feeding both to one LLM call lets local
    models truncate/lose the group-B rows (measured: identical input produced
    3/3/15 suppliers across runs — the raw-material rows were dropped). Each
    group is prompted separately and the results merged, so group B always
    gets its own short, focused prompt.

    Falls back to a single segment when the text cannot be classified.
    """
    disclosure: list[str] = []
    raw_material: list[str] = []

    for block in re.split(r"\n*---\n*", focused):
        block = block.strip()
        if not block:
            continue
        if _RAW_MATERIAL_SEGMENT_RE.search(block):
            raw_material.append(block)
            # Blocks can carry both table kinds (e.g. supplier table followed
            # directly by the raw-material table); keep those in group A too
            # so nothing is silently dropped.
            if _DISCLOSURE_SEGMENT_RE.search(block):
                disclosure.append(block)
        else:
            disclosure.append(block)

    parts: list[str] = []
    if disclosure:
        parts.append("\n\n---\n\n".join(disclosure))
    if raw_material:
        parts.append("\n\n---\n\n".join(raw_material))
    return parts or [focused]


def _merge_extracts(
    extracts: list[_AnnualReportExtract],
    fiscal_year: int,
) -> _AnnualReportExtract:
    """Union per-segment extractions into one disclosure, deduped.

    ``(raw_name, note)`` identifies a row; the same named company may appear
    in both the ratio-bearing disclosure table and the ratio-less raw-material
    table, and both rows are kept (matching the pre-split behaviour where one
    LLM call saw the whole report).
    """
    customers: list[_LLMCounterpartyEntry] = []
    suppliers: list[_LLMCounterpartyEntry] = []
    seen_customers: set[tuple[str, str | None]] = set()
    seen_suppliers: set[tuple[str, str | None]] = set()

    for extract in extracts:
        for c in extract.customers:
            key = (c.raw_name, c.note)
            if key not in seen_customers:
                seen_customers.add(key)
                customers.append(c)
        for s in extract.suppliers:
            key = (s.raw_name, s.note)
            if key not in seen_suppliers:
                seen_suppliers.add(key)
                suppliers.append(s)

    confidence = max((e.confidence for e in extracts), default=0.5)
    notes = " | ".join(n for e in extracts if e.notes for n in [e.notes]) or None
    return _AnnualReportExtract(
        customers=customers,
        suppliers=suppliers,
        confidence=confidence,
        fiscal_year=fiscal_year,
        notes=notes,
    )


def _extract_segment(
    segment: str,
    stock_code: str,
    stock_name: str,
    fiscal_year: int,
    llm_client: LLMClient,
) -> _AnnualReportExtract:
    """Run one LLM extraction for a single segment with bounded retries.

    Retries when the LLM call fails (validation/parse errors) or returns a
    completely empty extraction on a segment that clearly contains table rows
    (local LLMs intermittently drop everything instead of just some rows).
    """
    if len(segment) > _MAX_EXTRACT_CHARS:
        segment = segment[:_MAX_EXTRACT_CHARS]
    segment = _flatten_table_lines(segment)
    prompt = prompt_manager.render(
        "annual_report.j2",
        stock_code=stock_code,
        stock_name=stock_name,
        fiscal_year=fiscal_year,
        report_text=segment,
    )
    looks_like_table = bool(_TABLE_ROW_MARK_RE.search(segment))

    attempts = _MAX_SEGMENT_RETRIES if looks_like_table else 1
    for attempt in range(1, attempts + 1):
        try:
            extract = llm_client.extract_structured(prompt, _AnnualReportExtract, num_ctx=16384)
        except LLMError as e:
            if attempt < attempts:
                logger.warning(
                    "Annual-report segment extraction attempt %d/%d failed: %s",
                    attempt,
                    attempts,
                    e,
                )
                continue
            raise
        if (
            looks_like_table
            and not extract.customers
            and not extract.suppliers
            and attempt < attempts
        ):
            logger.warning(
                "Annual-report segment extraction attempt %d/%d returned empty "
                "despite table rows present — retrying",
                attempt,
                attempts,
            )
            continue
        return extract
    return _AnnualReportExtract(customers=[], suppliers=[], fiscal_year=fiscal_year)  # unreachable


def _scan_raw_material_names(
    segment: str,
    resolver: StockResolver,
    target_code: str,
    existing: list[_LLMCounterpartyEntry],
) -> list[_LLMCounterpartyEntry]:
    """Deterministic backstop: recover supplier rows the LLM dropped entirely.

    The raw-material supply table (主要原料之供應狀況/主要供應廠商) lists its
    suppliers as 、-separated company names; local LLMs occasionally omit whole
    rows across every pass (台化 113: the 丙烯腈 → 台塑公司 row never appeared).
    Tokenizing the flattened segment text and resolving every CJK token through
    the deterministic map recovers those rows.

    Only tokens that resolve to a listed stock are kept: 長春/奇美/日本出光/
    ASAHI etc. resolve to ``None`` and are dropped, so the backstop can never
    introduce a non-listed company. The target company itself (whose own report
    this is) is excluded, and names already present (by resolved code) are
    skipped to avoid duplicating LLM rows.

    Scanning is restricted to lines carrying a quantity/amount unit (公噸/公斤/
    千元): the raw-material table's data rows always bear one, while prose in
    the same segment does not. Without this guard the market-outlook prose
    「中國大陸…」 would resolve 大陸 → 2526 as a spurious supplier (台化 113
    measured false pull).
    """
    flat = _flatten_table_lines(segment)
    # existing 的 raw_name 可能是 LLM 併格（如「台塑石化公司、南亞公司」）——
    # 用同一套拆格邏輯展開，避免 backstop 對已捕捉的公司重複補行。
    existing_codes: set[str] = set()
    for e in existing:
        for part in _ENUMERATION_SPLIT_RE.split(e.raw_name):
            code = resolver.resolve_name(part.strip())
            if code is not None:
                existing_codes.add(code)
    found: list[_LLMCounterpartyEntry] = []
    seen_codes: set[str] = set()

    for line in flat.splitlines():
        if not _DATA_ROW_UNIT_RE.search(line):
            continue  # prose (market outlook 等) 沒有單位標記，整列跳過
        for token in _NAME_TOKEN_RE.split(line):
            token = token.strip()
            if not _HAS_CJK_RE.search(token):
                continue  # foreign names (SIBUR(LRD), ASAHI…) / bare numbers
            code = resolver.resolve_name(token)
            if code is None or code == target_code or code in existing_codes or code in seen_codes:
                continue
            found.append(_LLMCounterpartyEntry(raw_name=token, note=None))
            seen_codes.add(code)

    return found


def _extract(
    text: str,
    stock_code: str,
    stock_name: str,
    fiscal_year: int,
    llm_client: LLMClient,
    resolver: StockResolver | None = None,
) -> _AnnualReportExtract:
    """Send extracted text to the LLM for structured customer/supplier extraction.

    ``resolver`` enables the deterministic raw-material backstop scan; it is
    always supplied by ``extract_annual_report`` but left optional so pure-LLM
    unit tests stay independent of the stock map.
    """
    focused = _locate_customer_supplier_sections(text)

    # Safety net for the no-header fallback (which returns the full text).
    if len(focused) > _MAX_EXTRACT_CHARS:
        focused = focused[:_MAX_EXTRACT_CHARS]

    # Prompt each table kind separately, then union the results: local LLMs
    # reliably read short focused prompts but routinely drop the tail
    # (raw-material) rows of one giant combined prompt (see
    # ``_split_report_segments``).
    segments = _split_report_segments(focused)

    extracts: list[_AnnualReportExtract] = []
    for segment in segments:
        # Even a short segment is still read stochastically: measured same-input
        # runs returned 13 vs 3 suppliers. Union a few independent passes; each
        # pass loses a *different* subset, so the union converges on the full
        # table. Stop once a pass adds no new (name, note) rows.
        pass_extracts: list[_AnnualReportExtract] = []
        seen: set[tuple[str, str | None]] = set()
        for _ in range(_MAX_SEGMENT_PASSES):
            extract = _extract_segment(segment, stock_code, stock_name, fiscal_year, llm_client)
            pass_extracts.append(extract)
            new_rows = {(e.raw_name, e.note) for e in extract.customers + extract.suppliers}
            grew = new_rows - seen
            seen |= new_rows
            if not grew:
                break  # converged: last pass added no rows not already seen
        merged_segment = _merge_extracts(pass_extracts, fiscal_year)
        # Deterministic backstop for raw-material segments: the LLM can drop a
        # whole row (台化 113: 丙烯腈→台塑公司 was absent from every pass).
        # Resolve the segment's tokens through the stock map and add any listed
        # company the LLM missed (note=None, ratio=None as for other picks).
        if resolver is not None and _RAW_MATERIAL_SEGMENT_RE.search(segment):
            merged_segment.suppliers.extend(
                _scan_raw_material_names(segment, resolver, stock_code, merged_segment.suppliers)
            )
        extracts.append(merged_segment)

    return _merge_extracts(extracts, fiscal_year)


def _resolve_entries(
    entries: list[_LLMCounterpartyEntry],
    resolver: StockResolver,
    relation_label: str,
) -> tuple[list[CounterpartyDisclosure], list[str]]:
    """Resolve raw company names to listed stock codes.

    Local LLMs routinely merge a 、/，-separated supplier cell into one
    ``raw_name`` (台化 113: 「南亞公司、長春、日本出光」 as a single name →
    南亞/1303 lost). Splitting on the enumeration separators deterministically
    recovers the individual companies before resolving.

    Returns ``(disclosures, warnings)``.
    """
    disclosures: list[CounterpartyDisclosure] = []
    warnings: list[str] = []
    seen: set[tuple[str, str | None]] = set()

    for entry in entries:
        # Split merged cells (e.g. raw-material table rows); a split part must
        # be non-empty and keep the entry's note.
        raw_names = [
            part.strip() for part in _ENUMERATION_SPLIT_RE.split(entry.raw_name) if part.strip()
        ]
        if not raw_names:
            raw_names = [entry.raw_name]

        for raw_name in raw_names:
            key = (raw_name, entry.note)
            if key in seen:
                continue
            seen.add(key)
            code = resolver.resolve_name(raw_name)
            resolved_name: str | None = None
            if code:
                info = resolver.verify_code(code)
                resolved_name = info.get("name") or None

            disclosures.append(
                CounterpartyDisclosure(
                    raw_name=raw_name,
                    resolved_code=code,
                    resolved_name=resolved_name,
                    ratio=entry.ratio,
                    is_related_party=entry.is_related_party,
                    note=entry.note,
                    is_anonymous=entry.is_anonymous,
                )
            )
            if not code and not entry.is_anonymous:
                warnings.append(
                    f"{relation_label} '{raw_name}' could not be resolved to a listed stock"
                )

    return disclosures, warnings


def extract_annual_report(
    stock_code: str,
    stock_name: str,
    fiscal_year: int,
    *,
    llm_client: LLMClient | None = None,
    doc_client: DocTwseClient | None = None,
    resolver: StockResolver | None = None,
) -> AnnualReportDisclosure:
    """Extract customer/supplier data from the company's annual report.

    Tries ``fiscal_year`` first, then falls back up to ``_MAX_FALLBACK_YEARS``
    years back if no PDF is available for the target year.

    Args:
        stock_code:  4-digit stock code (e.g. "2330").
        stock_name:  Company name (e.g. "台積電").
        fiscal_year: Target ROC fiscal year (e.g. 113 for 2024).

    Returns:
        An :class:`AnnualReportDisclosure` with resolved customers/suppliers.

    Raises:
        AnnualReportError: If extraction fails entirely.
    """
    _llm = llm_client or LLMClient()
    _doc = doc_client or default_doc_client
    _resolver = resolver or default_resolver

    # Cache-first: avoid re-downloading the same PDF + re-running the LLM.
    cache_key = f"annual_report_{stock_code}_{fiscal_year}"
    try:
        cached = get_cache(cache_key, _DISCLOSURE_CACHE_TTL)
    except CacheError as e:
        cached = None
        logger.warning("Annual-report cache read failed: %s", e)
    if cached is not None:
        try:
            return AnnualReportDisclosure.model_validate(cached)
        except ValidationError as e:
            logger.warning("Cached annual-report disclosure invalid, re-extracting: %s", e)

    last_error: str = ""

    for year_offset in range(_MAX_FALLBACK_YEARS + 1):
        try_year = fiscal_year - year_offset
        if year_offset > 0:
            logger.info(
                "Falling back to fiscal year %d (offset %d) for %s",
                try_year,
                year_offset,
                stock_code,
            )

        # Try F04 (股東會年報, the only dtype doc.twse actually serves) first,
        # then F18 as a fallback for companies that may use it.
        # NOTE: doc.twse F-class search uses the *shareholder-meeting year*
        # (= fiscal year + 1). We search doc_year but label results with the
        # correct fiscal year.
        for mtype, dtype, label in [
            ("F", "F04", "F04"),
            ("F", "F18", "F18"),
        ]:
            doc_year = try_year + 1
            try:
                pdf_bytes, pdf_url = _doc.fetch_pdf(stock_code, doc_year, mtype=mtype, dtype=dtype)
            except DocTwseClientError as e:
                last_error = str(e)
                logger.debug(
                    "doc.twse %s failed for %s year=%d: %s", label, stock_code, try_year, e
                )
                continue

            # Extract text
            raw_text = _extract_text_from_pdf(pdf_bytes)
            if not raw_text.strip():
                last_error = f"PDF text extraction returned empty for {label}"
                logger.warning(last_error)
                continue

            # LLM extraction (deterministic raw-material backstop enabled)
            try:
                extract = _extract(raw_text, stock_code, stock_name, try_year, _llm, _resolver)
            except LLMError as e:
                last_error = f"LLM extraction failed: {e}"
                logger.warning(last_error)
                continue

            # Resolve names
            customers, cust_warnings = _resolve_entries(extract.customers, _resolver, "customer")
            suppliers, supp_warnings = _resolve_entries(extract.suppliers, _resolver, "supplier")

            # Determine the actual PDF filename from the URL
            pdf_name: str | None = None
            if pdf_url:
                pdf_name = pdf_url.rsplit("/", 1)[-1] if "/" in pdf_url else pdf_url

            result = AnnualReportDisclosure(
                target_code=stock_code,
                target_name=stock_name,
                fiscal_year=try_year,
                report_type=label,
                pdf_name=pdf_name,
                pdf_url=pdf_url,
                customers=customers,
                suppliers=suppliers,
                source_text=raw_text[:2000],  # store snippet, not full text
                confidence=extract.confidence,
            )

            if year_offset > 0:
                # Note the fallback in the source_text
                result.source_text = (
                    f"[Fallback: target year {fiscal_year} unavailable, "
                    f"using {try_year}] " + (result.source_text or "")
                )

            notes = cust_warnings + supp_warnings
            if notes:
                logger.info("Resolution warnings for %s: %s", stock_code, notes)

            try:
                set_cache(cache_key, result.model_dump(mode="json"))
            except CacheError as e:
                logger.warning("Annual-report cache write failed: %s", e)

            return result

    raise AnnualReportError(
        f"Could not extract annual report for {stock_code} ({stock_name}) "
        f"year={fiscal_year} (tried {fiscal_year}..{fiscal_year - _MAX_FALLBACK_YEARS}): "
        f"{last_error}"
    )
