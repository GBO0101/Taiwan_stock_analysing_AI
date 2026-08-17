"""Step 2: Query Classification."""

import logging

from classifier.chart_validator import ChartValidator
from classifier.chart_validator import chart_validator as default_validator
from classifier.llm_client import LLMClient, LLMError
from classifier.models import BoundaryResult, ClassificationResult
from classifier.prompts import prompt_manager

logger = logging.getLogger(__name__)


class ClassificationError(Exception):
    """Classification errors."""



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
        return result
    except LLMError as e:
        raise ClassificationError(f"Classification failed: {e}") from e
