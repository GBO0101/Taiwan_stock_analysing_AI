"""Pipeline runner: orchestrates Steps 1-3 in strict sequential order."""

from typing import Any
from classifier.models import (
    BoundaryResult,
    ClassificationResult,
    DecompositionResult,
    PipelineResult,
    PipelineStepResult,
    StepStatus,
    ClassificationType,
)
from classifier.boundary import extract_boundary, BoundaryExtractionError
from classifier.classification import classify_query, ClassificationError
from classifier.decomposition import decompose_query, DecompositionError
from classifier.llm_client import LLMClient
from classifier.indicator_mapper import IndicatorMapper


class PipelineError(Exception):
    """Pipeline execution errors."""

    pass


class Pipeline:
    """Strict sequential pipeline runner for Taiwan-stock query understanding.

    Executes Steps 1-3 in order:
    1. Boundary Extraction
    2. Classification
    3. Decomposition (conditional - only for analytical queries)
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        indicator_mapper: IndicatorMapper | None = None,
    ):
        """Initialize pipeline.

        Args:
            llm_client: Optional LLM client (creates default if not provided)
            indicator_mapper: Optional indicator mapper (creates default if not provided)
        """
        self.llm_client = llm_client or LLMClient()
        self.indicator_mapper = indicator_mapper or IndicatorMapper()

    def run(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline on a question.

        Args:
            question: User's natural language question
            context: Optional chat context for pronoun resolution

        Returns:
            PipelineResult with step trace

        Raises:
            PipelineError: If any step fails
        """
        steps: list[PipelineStepResult] = []

        # Step 1: Boundary Extraction
        try:
            boundary = extract_boundary(
                question=question,
                context=context,
                llm_client=self.llm_client,
            )
            steps.append(PipelineStepResult(
                step="boundary",
                status=StepStatus.COMPLETED,
                output=boundary.model_dump(),
            ))
        except BoundaryExtractionError as e:
            steps.append(PipelineStepResult(
                step="boundary",
                status=StepStatus.FAILED,
                output={"error": str(e)},
            ))
            raise PipelineError(f"Step 1 (Boundary) failed: {e}") from e

        # Step 2: Classification
        try:
            classification = classify_query(
                question=question,
                boundary=boundary,
                llm_client=self.llm_client,
            )
            steps.append(PipelineStepResult(
                step="classification",
                status=StepStatus.COMPLETED,
                output=classification.model_dump(),
            ))
        except ClassificationError as e:
            steps.append(PipelineStepResult(
                step="classification",
                status=StepStatus.FAILED,
                output={"error": str(e)},
            ))
            raise PipelineError(f"Step 2 (Classification) failed: {e}") from e

        # Step 3: Decomposition (conditional - only for analytical queries)
        if classification.type == ClassificationType.ANALYTICAL:
            try:
                decomposition = decompose_query(
                    question=question,
                    boundary=boundary,
                    classification=classification,
                    llm_client=self.llm_client,
                    indicator_mapper=self.indicator_mapper,
                )
                steps.append(PipelineStepResult(
                    step="decomposition",
                    status=StepStatus.COMPLETED,
                    output=decomposition.model_dump(),
                ))
            except DecompositionError as e:
                steps.append(PipelineStepResult(
                    step="decomposition",
                    status=StepStatus.FAILED,
                    output={"error": str(e)},
                ))
                raise PipelineError(f"Step 3 (Decomposition) failed: {e}") from e
        else:
            steps.append(PipelineStepResult(
                step="decomposition",
                status=StepStatus.SKIPPED,
                output={"reason": f"Skipped: classification type is {classification.type}"},
            ))

        return PipelineResult(
            question=question,
            steps=steps,
        )
