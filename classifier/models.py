"""Pydantic models for the classify-twse-query pipeline contracts."""

from datetime import date
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class ClassificationType(str, Enum):
    """Query classification types."""

    LIVE = "live"
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    NON_FINANCIAL = "non_financial"


class ChartType(str, Enum):
    """Supported chart types."""

    LINE = "line"
    KLINE = "kline"
    BAR = "bar"


class ChartDataRequirement(str, Enum):
    """Supported chart data requirements."""

    PRICE_TREND = "price_trend"
    PRICE_OHLC = "price_ohlc"
    REVENUE_TREND = "revenue_trend"
    REVENUE_COMPARISON = "revenue_comparison"
    INDICATOR_TREND = "indicator_trend"
    SECTOR_ANALYSIS = "sector_analysis"


class OperationType(str, Enum):
    """Sub-query operation types."""

    FINMIND_QUERY = "finmind_query"
    COMPUTE = "compute"
    SENTIMENT_ANALYSIS = "sentiment_analysis"


class StepStatus(str, Enum):
    """Pipeline step execution status."""

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TimeScope(str, Enum):
    """Time scope types."""

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


class StockScope(str, Enum):
    """Stock scope types."""

    SINGLE_STOCK = "single_stock"
    MULTIPLE_STOCKS = "multiple_stocks"
    SECTOR = "sector"
    MARKET = "market"


class DataDimension(str, Enum):
    """Data dimension types."""

    PRICE = "price"
    REVENUE = "revenue"
    FINANCIAL = "financial"
    NEWS = "news"
    INSTITUTIONAL = "institutional"


class Market(str, Enum):
    """Market types."""

    TWSE = "TWSE"
    TPEx = "TPEx"


class DateRange(BaseModel):
    """Date range specification."""

    type: TimeScope = Field(..., description="Type of date range")
    value: str = Field(..., description="Range value (e.g., '30d', '2024-01-01/2024-12-31')")


class BoundaryResult(BaseModel):
    """Step 1 output: extracted query boundary and entities."""

    stock_codes: list[str] = Field(default_factory=list, description="Stock codes (e.g., ['2330'])")
    company_names: list[str] = Field(default_factory=list, description="Company names (e.g., ['台積電'])")
    sectors: list[str] = Field(default_factory=list, description="Sector names")
    date_range: Optional[DateRange] = Field(default=None, description="Date range specification")
    time_scope: Optional[TimeScope] = Field(default=None, description="Time scope type")
    stock_scope: Optional[StockScope] = Field(default=None, description="Stock scope type")
    data_dimension: Optional[DataDimension] = Field(default=None, description="Data dimension")
    market: Market = Field(default=Market.TWSE, description="Market")
    metrics: list[str] = Field(default_factory=list, description="Requested metrics")
    chart_type: Optional[ChartType] = Field(default=None, description="Chart type hint from query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")


class ClassificationResult(BaseModel):
    """Step 2 output: query classification and visualization requirements."""

    type: ClassificationType = Field(..., description="Query classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    needs_clarification: bool = Field(default=False, description="Whether clarification is needed")
    clarification_question: Optional[str] = Field(default=None, description="Clarification question if needed")
    needs_visualization: bool = Field(default=False, description="Whether visualization is requested")
    chart_type: Optional[ChartType] = Field(default=None, description="Chart type for visualization")
    chart_data_requirements: Optional[ChartDataRequirement] = Field(
        default=None, description="Chart data requirement type"
    )
    target_datasets: list[str] = Field(default_factory=list, description="Target FinMind datasets")


class SubQuery(BaseModel):
    """Single sub-query in the decomposition DAG."""

    id: str = Field(..., description="Unique node ID (e.g., 'q0')")
    operation: OperationType = Field(..., description="Operation type")
    datasets: list[str] = Field(default_factory=list, description="FinMind datasets for finmind_query")
    computed_field: Optional[str] = Field(default=None, description="Computed field name for compute operations")
    formula: Optional[str] = Field(default=None, description="Formula referencing prerequisite node IDs")
    params: dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    depends_on: list[str] = Field(default_factory=list, description="Direct prerequisite node IDs")

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, v: list[str], info) -> list[str]:
        """finmind_query must have at least one dataset."""
        if info.data.get("operation") == OperationType.FINMIND_QUERY and not v:
            raise ValueError("finmind_query operation requires at least one dataset")
        return v


class DecompositionResult(BaseModel):
    """Step 3 output: ordered executable sub-query DAG."""

    sub_queries: list[SubQuery] = Field(..., description="Sub-queries in dependency-respecting order")

    @field_validator("sub_queries")
    @classmethod
    def validate_unique_ids(cls, v: list[SubQuery]) -> list[SubQuery]:
        """Ensure all node IDs are unique."""
        ids = [q.id for q in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Sub-query IDs must be unique")
        return v

    @field_validator("sub_queries")
    @classmethod
    def validate_dependencies(cls, v: list[SubQuery]) -> list[SubQuery]:
        """Validate dependency references exist and no cycles."""
        id_set = {q.id for q in v}
        for query in v:
            for dep in query.depends_on:
                if dep not in id_set:
                    raise ValueError(f"Dependency '{dep}' not found in sub-queries")

        # Check for cycles using topological sort
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False
            visited.add(node_id)
            rec_stack.add(node_id)
            query = next((q for q in v if q.id == node_id), None)
            if query:
                for dep in query.depends_on:
                    if has_cycle(dep):
                        return True
            rec_stack.remove(node_id)
            return False

        for query in v:
            if has_cycle(query.id):
                raise ValueError("Sub-query DAG contains cycles")
        return v

    @field_validator("sub_queries")
    @classmethod
    def validate_topological_order(cls, v: list[SubQuery]) -> list[SubQuery]:
        """Validate prerequisites occur before dependent nodes."""
        seen = set()
        for query in v:
            for dep in query.depends_on:
                if dep not in seen:
                    raise ValueError(f"Prerequisite '{dep}' must appear before dependent '{query.id}'")
            seen.add(query.id)
        return v

    @field_validator("sub_queries")
    @classmethod
    def validate_formulas(cls, v: list[SubQuery]) -> list[SubQuery]:
        """Validate formulas reference valid node IDs."""
        id_set = {q.id for q in v}
        for query in v:
            if query.formula:
                # Simple check: formula should reference existing node IDs
                import re
                refs = re.findall(r'q\d+', query.formula)
                for ref in refs:
                    if ref not in id_set:
                        raise ValueError(f"Formula references unknown node '{ref}'")
        return v


class PipelineStepResult(BaseModel):
    """Single pipeline step result for trace."""

    step: str = Field(..., description="Step name: boundary, classification, decomposition")
    status: StepStatus = Field(..., description="Step execution status")
    output: dict[str, Any] = Field(default_factory=dict, description="Step output data")


class PipelineResult(BaseModel):
    """Complete pipeline execution result."""

    question: str = Field(..., description="Original user question")
    steps: list[PipelineStepResult] = Field(..., description="Pipeline step trace")


class ChartRequest(BaseModel):
    """Chart rendering request."""

    stock_codes: list[str] = Field(..., description="Stock codes to chart")
    chart_data_requirements: ChartDataRequirement = Field(..., description="Chart data requirement type")
    chart_type: ChartType = Field(..., description="Chart type")
    start_date: Optional[date] = Field(default=None, description="Start date (optional, uses defaults)")
    end_date: Optional[date] = Field(default=None, description="End date (optional, uses defaults)")