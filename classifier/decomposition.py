"""Step 3: Query Decomposition."""

from typing import Any
from classifier.models import (
    BoundaryResult,
    ClassificationResult,
    DecompositionResult,
)
from classifier.llm_client import LLMClient, LLMError
from classifier.prompts import prompt_manager
from classifier.indicator_mapper import IndicatorMapper


class DecompositionError(Exception):
    """Decomposition errors."""

    pass


def decompose_query(
    question: str,
    boundary: BoundaryResult,
    classification: ClassificationResult,
    llm_client: LLMClient | None = None,
    indicator_mapper: IndicatorMapper | None = None,
) -> DecompositionResult:
    """Decompose an analytical query into an ordered executable sub-query DAG.

    Args:
        question: User's natural language question
        boundary: Step 1 boundary extraction result
        classification: Step 2 classification result
        llm_client: Optional LLM client (creates default if not provided)
        indicator_mapper: Optional indicator mapper (creates default if not provided)

    Returns:
        DecompositionResult with ordered sub-queries

    Raises:
        DecompositionError: If decomposition fails
    """
    client = llm_client or LLMClient()
    mapper = indicator_mapper or IndicatorMapper()

    try:
        prompt = prompt_manager.render_decompose(
            question=question,
            boundary=boundary.model_dump(),
            classification=classification.model_dump(),
            indicator_mappings=mapper.to_prompt_format(),
        )
        result = client.extract_structured(prompt, DecompositionResult)
        return result
    except LLMError as e:
        raise DecompositionError(f"Decomposition failed: {e}") from e
