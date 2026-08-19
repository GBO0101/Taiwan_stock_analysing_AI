"""Step 1: Boundary Extraction."""

import logging
from typing import Any

import re

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
from classifier.chart_validator import (
    CHART_TYPE_BY_REQUIREMENT,
    derive_requirement,
    derive_visualization,
)

logger = logging.getLogger(__name__)

# A question that anchors a range to a specific calendar year with a month
# (e.g. "2024年 1~6月") is unambiguously ABSOLUTE. The LLM sometimes still
# emits a relative range for these; we re-derive the absolute window from the
# raw question text in that case.
_YEAR_ANCHORED_RE = re.compile(r"\d{4}\s*年[^月]*月")


def _looks_year_anchored(text: str | None) -> bool:
    """True when the text anchors a range to a specific year + month."""
    return bool(_YEAR_ANCHORED_RE.search(text or ""))


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

        # Reconcile a flaky relative/absolute misclassification: if the LLM
        # emitted a RELATIVE range but the question clearly anchors to a
        # specific calendar year with a month range (e.g. "2024年 1~6月"),
        # re-derive an ABSOLUTE range from the question text. This mirrors how
        # the stock resolver corrects name->code mismatches downstream.
        if (
            result.date_range is not None
            and result.date_range.type == TimeScope.RELATIVE
            and _looks_year_anchored(question)
        ):
            resolved = _resolve_absolute_range(question)
            if resolved is not None:
                value = f"{resolved[0].isoformat()}/{resolved[1].isoformat()}"
                logger.warning(
                    "Boundary date reconciliation: LLM emitted relative %r but the "
                    "question anchors to a specific year; re-derived absolute %r",
                    result.date_range.value,
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