"""Pydantic models for the classify-twse-query pipeline contracts."""

from datetime import date, datetime, timedelta
import re
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


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


def _parse_date_flex(raw: str) -> date | None:
    """Parse a date string accepting Western and Chinese, day/month/year precision."""
    raw = raw.strip()
    # Chinese forms: 2024年1月1日 / 2024年1月 / 2024年
    m = re.match(r"^\s*(\d{4})\s*年\s*(\d{1,2})?\s*月?\s*(\d{1,2})?\s*日?\s*$", raw)
    if m:
        year = int(m.group(1))
        month = int(m.group(2)) if m.group(2) else 1
        day = int(m.group(3)) if m.group(3) else 1
        try:
            return date(year, month, day)
        except ValueError:
            return None
    # Western forms: 2024-01-01 / 2024-01 / 2024 / 2024/01/01 / 2024/01
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _extract_date_tokens(value: str) -> list[str]:
    """Pull Western date tokens out of a free-form range string.

    Handles 'YYYY-MM-DD', 'YYYY/MM/DD', 'YYYY-MM', 'YYYY/MM' and bare 'YYYY'.
    A trailing word boundary prevents a 2-digit day from over-consuming into a
    following 4-digit year (e.g. when '/' doubles as both separators). Chinese
    ('年/月/日') ranges are handled separately in ``_resolve_chinese_range``.
    """
    pattern = re.compile(
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"  # 2024-01-01 / 2024/01/01
        r"|\d{4}[-/]\d{1,2}\b"           # 2024-01 / 2024/01
        r"|\d{4}"                         # 2024
    )
    return pattern.findall(value)


def _expand_single(d: date, raw: str) -> tuple[date, date]:
    """Expand a single parsed date into a sensible (start, end) range."""
    raw = raw.strip()
    if re.fullmatch(r"\d{4}", raw) or re.fullmatch(r"\d{4}\s*年", raw):
        # year-only -> whole year
        return d.replace(month=1, day=1), d.replace(month=12, day=31)
    if re.fullmatch(r"\d{4}[-/]\d{1,2}", raw) or re.fullmatch(r"\d{4}\s*年\s*\d{1,2}\s*月", raw):
        # month-only -> whole month
        start = d.replace(day=1)
        last = d.replace(day=28) + timedelta(days=4)
        last = last - timedelta(days=last.day)
        return start, last
    # full date -> leave as-is (degenerate single-day range)
    return d, d


def _resolve_chinese_range(value: str) -> tuple[date, date] | None:
    """Resolve a Chinese-style absolute range, e.g. '2024年1月到7月' or '2024年 1~6月'."""
    year_m = re.search(r"(\d{4})\s*年", value)
    if not year_m:
        return None
    year = int(year_m.group(1))
    # Capture each month token, tolerating '月', '~', '-', '到', '至' as separators
    # so '1~6月' and '1月到7月' both yield (start_month, end_month).
    pairs = re.findall(
        r"(\d{1,2})\s*(?:月\s*(?:(\d{1,2})\s*日)?|[~\-到至])", value
    )
    if not pairs:
        return None
    points: list[date] = []
    had_day: list[bool] = []
    for mo_s, day_s in pairs:
        mo = int(mo_s)
        day = int(day_s) if day_s else 1
        try:
            points.append(date(year, mo, day))
            had_day.append(bool(day_s))
        except ValueError:
            continue
    if not points:
        return None
    start = points[0]
    end = points[-1]
    # Expand month-only endpoints to the full month.
    if not had_day[0]:
        start = start.replace(day=1)
    if not had_day[-1]:
        last = end.replace(day=28) + timedelta(days=4)
        last = last - timedelta(days=last.day)
        end = last
    if start > end:
        start, end = end, start
    return start, end


def _resolve_absolute_range(value: str) -> tuple[date, date] | None:
    """Resolve an absolute range value tolerant of many LLM output formats."""
    value = re.sub(r"\s+", "", value)
    # Chinese form (contains 年) gets dedicated handling first.
    if "年" in value:
        result = _resolve_chinese_range(value)
        if result is not None:
            return result
    # Western form: tokenize on date-like patterns.
    tokens = _extract_date_tokens(value)
    if not tokens:
        return None
    parsed: list[date] = []
    for t in tokens:
        d = _parse_date_flex(t)
        if d is not None:
            parsed.append(d)
    if not parsed:
        return None
    if len(parsed) == 1:
        return _expand_single(parsed[0], tokens[0])
    start, end = parsed[0], parsed[-1]
    if start > end:
        start, end = end, start
    return start, end


class DateRange(BaseModel):
    """Date range specification."""

    type: TimeScope = Field(..., description="Type of date range")
    value: str = Field(..., description="Range value (e.g., '30d', '2024-01-01/2024-12-31')")

    def resolve_dates(self) -> tuple[date, date] | None:
        """Resolve this range into concrete ``(start_date, end_date)``.

        Absolute ranges use ``YYYY-MM-DD[/YYYY-MM-DD]`` (month/year precision also
        accepted). Relative ranges use ``<N>[dwmy]`` (days/weeks/months/years)
        counted back from today. Returns ``None`` if the value cannot be parsed.
        """
        try:
            if self.type == TimeScope.ABSOLUTE:
                return _resolve_absolute_range(self.value)
            # Relative: counted back from today.
            end = date.today()
            match = re.match(r"\s*(\d+)\s*([dwmy])\s*$", self.value.lower())
            if not match:
                return None
            num = int(match.group(1))
            unit = match.group(2)
            if unit == "d":
                delta = timedelta(days=num)
            elif unit == "w":
                delta = timedelta(weeks=num)
            elif unit == "m":
                delta = timedelta(days=num * 30)
            else:  # "y"
                delta = timedelta(days=num * 365)
            return end - delta, end
        except (ValueError, AttributeError):
            return None


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
    chart_data_requirements: Optional[ChartDataRequirement] = Field(
        default=None,
        description="Validated chart data requirement; single source of truth for charting",
    )
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
    date_range: Optional[DateRange] = Field(
        default=None, description="Optional date range; resolved into start/end dates when not set"
    )

    @model_validator(mode="after")
    def _resolve_date_range(self) -> "ChartRequest":
        """Derive start/end dates from ``date_range`` when not explicitly given."""
        if self.date_range is not None and (self.start_date is None or self.end_date is None):
            resolved = self.date_range.resolve_dates()
            if resolved is not None:
                if self.start_date is None:
                    self.start_date = resolved[0]
                if self.end_date is None:
                    self.end_date = resolved[1]
        return self