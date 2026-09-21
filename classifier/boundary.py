"""Step 1: Boundary Extraction."""

import logging
import re
from datetime import date
from typing import Any

from classifier.chart_validator import (
    CHART_TYPE_BY_REQUIREMENT,
    derive_requirement,
    derive_visualization,
)
from classifier.llm_client import LLMClient, LLMError
from classifier.models import (
    BoundaryResult,
    ChartDataRequirement,
    DateRange,
    TimeScope,
    _resolve_absolute_range,
)
from classifier.prompts import prompt_manager
from classifier.stock_resolver import StockResolver
from classifier.stock_resolver import stock_resolver as default_resolver

logger = logging.getLogger(__name__)

# A question that anchors a range to a specific calendar year (e.g. "2024年",
# "2024年 1~6月") is unambiguously ABSOLUTE. The LLM sometimes still emits a
# relative range or null for these; we re-derive the absolute window from the
# raw question text in that case.
_YEAR_ANCHORED_RE = re.compile(r"\d{4}\s*年(?:[^月]*月)?")

# 民國 N 年 (e.g. 民國113年) maps to Gregorian N+1911 (民國113 = 2024). The
# LLM frequently leaves ``date_range`` null for these; see
# ``_resolve_question_year_range``.
_MINGUO_YEAR_RE = re.compile(r"民國\s*(\d{2,3})\s*年")

# A bare 4-digit calendar year immediately followed by supply-chain /
# annual-report context (e.g. "南亞2024上下游名單", "台塑2024供應商名單")
# is also an unambiguous ABSOLUTE anchor, even without the "年" suffix.
# The year is validated in ``_looks_year_anchored`` so a stock code sitting
# next to such a keyword (e.g. "2330供應商") is not mistaken for a year.
_BARE_YEAR_CONTEXT_RE = re.compile(r"(\d{4})\s*(?:上下游|供應商|客戶|名單|年報|年度|財報)")

# Calendar years outside this window are never treated as anchors; stock
# codes (2330, 1303, ...) fall well outside it.
_BARE_YEAR_MIN = 1990
_BARE_YEAR_MAX = 2040


def _looks_year_anchored(text: str | None) -> bool:
    """True when the text anchors a range to a specific calendar year.

    Matches Gregorian (``2024年``, ``2024年 1~6月``), bare year + context
    (``2024上下游名單``), and ROC-era (``民國113年``) forms.
    """
    text = text or ""
    if _YEAR_ANCHORED_RE.search(text):
        return True
    m = _BARE_YEAR_CONTEXT_RE.search(text)
    if m:
        year = int(m.group(1))
        if _BARE_YEAR_MIN <= year <= _BARE_YEAR_MAX:
            return True
    return bool(_MINGUO_YEAR_RE.search(text))


def _resolve_question_year_range(question: str) -> tuple[date, date] | None:
    """Resolve a year anchored in the question itself (LLM-agnostic).

    The LLM sometimes leaves ``date_range`` null or relative even when the
    question names a specific historical year (e.g. "南亞2024年上下游名單",
    "台塑民國113年度年報揭露的供應商與客戶有哪些"). Derive the absolute window
    deterministically: ``民國 N 年`` -> Gregorian year N+1911 (full year);
    otherwise delegate to the general absolute-range parser (Gregorian
    year/month forms).
    """
    m = _MINGUO_YEAR_RE.search(question)
    if m:
        year = int(m.group(1)) + 1911
        return date(year, 1, 1), date(year, 12, 31)
    return _resolve_absolute_range(question)


def _reconcile_chart_fields(question: str, result: BoundaryResult) -> None:
    """Validate and correct chart fields on the boundary output in place.

    The boundary step is the single source of truth for charting. We apply the
    same deterministic keyword logic ``ChartValidator`` uses on the
    classification step, so ``result.chart_data_requirements`` and
    ``result.chart_type`` reflect the final chart decision (what STEP BOUNDRY
    displays). Every correction is logged and never silently discarded.
    """
    vis = derive_visualization(question)
    req: ChartDataRequirement | None = derive_requirement(question) if vis else None
    if vis and req is None:
        req = ChartDataRequirement.PRICE_TREND  # safe default
    if not vis:
        req = None

    if result.chart_data_requirements != req:
        logger.warning(
            "Boundary chart reconciliation: chart_data_requirements %r -> %r",
            result.chart_data_requirements,
            req,
        )
        result.chart_data_requirements = req

    if not vis:
        if result.chart_type is not None:
            logger.warning(
                "Boundary chart reconciliation: chart_type cleared (no visual wording): %r",
                result.chart_type,
            )
            result.chart_type = None
    elif req is not None:
        expected = CHART_TYPE_BY_REQUIREMENT.get(req)
        if expected is not None and result.chart_type != expected:
            logger.warning(
                "Boundary chart reconciliation: chart_type %r -> %r (matches %r)",
                result.chart_type,
                expected.value,
                req.value,
            )
            result.chart_type = expected


class BoundaryExtractionError(Exception):
    """Boundary extraction errors."""


def extract_boundary(
    question: str,
    context: dict[str, Any] | None = None,
    llm_client: LLMClient | None = None,
    stock_resolver: StockResolver | None = None,
) -> BoundaryResult:
    """Extract query boundary and entities from natural language question.

    After the LLM extraction, a deterministic ``StockResolver`` reconciles
    ``company_names`` -> ``stock_codes`` (and reverse-verifies any codes the LLM
    supplied). This fixes name-only queries that previously left ``stock_codes``
    empty and broke the pipeline downstream.

    Args:
        question: User's natural language question
        context: Optional chat context for pronoun resolution
        llm_client: Optional LLM client (creates default if not provided)
        stock_resolver: Optional resolver (uses the global instance if not given)

    Returns:
        BoundaryResult with extracted entities and metadata

    Raises:
        BoundaryExtractionError: If extraction fails
    """
    client = llm_client or LLMClient()
    resolver = stock_resolver or default_resolver

    try:
        prompt = prompt_manager.render_boundary(question=question, context=context)
        result = client.extract_structured(prompt, BoundaryResult)
        resolved_codes, warnings = resolver.resolve(result.company_names, result.stock_codes)
        for warning in warnings:
            logger.warning("Boundary resolution: %s", warning)
        # Derive company_names from the resolved codes via reverse lookup, so the
        # field always carries the canonical Chinese name even when the user
        # queried by code (e.g. "3008" -> "大立光"). Names the resolver could not
        # map are intentionally dropped (they are also absent from stock_codes).
        company_names: list[str] = []
        seen_names: set[str] = set()
        for code in resolved_codes:
            info = resolver.verify_code(code)
            if info["exists"] and info["name"]:
                name = info["name"]
                if name not in seen_names:
                    company_names.append(name)
                    seen_names.add(name)
        result.stock_codes = resolved_codes
        result.company_names = company_names

        # Reconcile a missing or flaky date range: if the LLM left
        # ``date_range`` null (common with the /v1 JSON-mode path) or emitted a
        # RELATIVE range, but the question clearly anchors to a specific
        # calendar year (e.g. "南亞2024年上下游名單", "台塑民國113年度年報",
        # "和益 2024年 1~6月趨勢圖"), re-derive an ABSOLUTE range from the
        # question text. This mirrors how the stock resolver corrects
        # name->code mismatches downstream.
        if _looks_year_anchored(question) and (
            result.date_range is None or result.date_range.type == TimeScope.RELATIVE
        ):
            resolved = _resolve_question_year_range(question)
            if resolved is not None:
                value = f"{resolved[0].isoformat()}/{resolved[1].isoformat()}"
                logger.warning(
                    "Boundary date reconciliation: LLM date_range=%r but the "
                    "question anchors to a specific year; re-derived absolute %r",
                    result.date_range.value if result.date_range else None,
                    value,
                )
                result.date_range = DateRange(type=TimeScope.ABSOLUTE, value=value)
                result.time_scope = TimeScope.ABSOLUTE

        # Validate and correct chart fields so the boundary output is the single
        # source of truth for charting (uses ChartValidator's keyword logic).
        _reconcile_chart_fields(question, result)

        return result
    except LLMError as e:
        raise BoundaryExtractionError(f"Boundary extraction failed: {e}") from e
