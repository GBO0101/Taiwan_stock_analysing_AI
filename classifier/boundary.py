"""Step 1: Boundary Extraction."""

import logging
from typing import Any

from classifier.llm_client import LLMClient, LLMError
from classifier.models import BoundaryResult
from classifier.prompts import prompt_manager
from classifier.stock_resolver import StockResolver
from classifier.stock_resolver import stock_resolver as default_resolver

logger = logging.getLogger(__name__)


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
        return result
    except LLMError as e:
        raise BoundaryExtractionError(f"Boundary extraction failed: {e}") from e