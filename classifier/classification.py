"""Step 2: Query Classification."""

import logging

from classifier.chart_validator import ChartValidator
from classifier.chart_validator import chart_validator as default_validator
from classifier.llm_client import LLMClient, LLMError
from classifier.models import (
    BoundaryResult,
    ClassificationResult,
    ClassificationType,
    StockScope,
)
from classifier.prompts import prompt_manager

logger = logging.getLogger(__name__)


class ClassificationError(Exception):
    """Classification errors."""


def _reconcile_query_type(
    boundary: BoundaryResult,
    result: ClassificationResult,
) -> list[str]:
    """Correct a contradictory NON_FINANCIAL classification (in place).

    ``classification.type`` was entirely LLM-decided, and the local model
    (JSON-mode /v1 path) sometimes tags stock-domain queries as
    ``non_financial`` (e.g. "南亞2024年上下游名單"), which makes the pipeline
    skip Steps 3-4 entirely. But a ``non_financial`` verdict contradicts a
    boundary that already resolved real stock/sector/market scope — a query
    naming a known Taiwan stock is by definition in-domain. Such contradictions
    are corrected deterministically to ``factual`` (the conservative in-domain
    default: it still triggers Step 4 stock enumeration without forcing a
    possibly-noisy Step 3 decomposition). Mirrors the StockResolver /
    ChartValidator / boundary date reconciliation philosophy.

    Returns warnings describing every correction.
    """
    if result.type != ClassificationType.NON_FINANCIAL:
        return []
    has_scope = bool(
        boundary.stock_codes
        or boundary.company_names
        or boundary.sectors
        or boundary.stock_scope in (StockScope.SECTOR, StockScope.MARKET)
    )
    if not has_scope:
        return []
    warnings = [
        (
            "classification type corrected from 'non_financial' to 'factual' "
            f"(boundary resolved in-domain scope: "
            f"{boundary.stock_codes or boundary.sectors or boundary.company_names})"
        )
    ]
    result.type = ClassificationType.FACTUAL
    result.confidence = max(result.confidence, 0.5)
    return warnings


def classify_query(
    question: str,
    boundary: BoundaryResult,
    llm_client: LLMClient | None = None,
    chart_validator: ChartValidator | None = None,
) -> ClassificationResult:
    """Classify a query into one of four types and determine visualization requirements.

    After the LLM classification, a deterministic ``ChartValidator`` re-derives
    ``needs_visualization`` and ``chart_data_requirements`` from the question
    keywords (the source of truth) and optionally forces ``chart_type`` to match.
    This fixes misclassifications where a price-trend question was tagged as
    ``indicator_trend`` or where visualization intent was missed.

    Args:
        question: User's natural language question
        boundary: Step 1 boundary extraction result
        llm_client: Optional LLM client (creates default if not provided)
        chart_validator: Optional validator (uses the global instance if not given)

    Returns:
        ClassificationResult with query type and visualization requirements

    Raises:
        ClassificationError: If classification fails
    """
    client = llm_client or LLMClient()
    validator = chart_validator or default_validator

    try:
        prompt = prompt_manager.render_classify(
            question=question,
            boundary=boundary.model_dump(),
        )
        result = client.extract_structured(prompt, ClassificationResult)
        result, warnings = validator.validate(question, result)
        for warning in warnings:
            logger.warning("Chart validation: %s", warning)
        # Deterministic type reconciliation (see ``_reconcile_query_type``):
        # a NON_FINANCIAL verdict cannot stand when the boundary resolved real
        # stock/sector/market scope.
        for warning in _reconcile_query_type(boundary, result):
            logger.warning("Classification type reconciliation: %s", warning)
        return result
    except LLMError as e:
        raise ClassificationError(f"Classification failed: {e}") from e
