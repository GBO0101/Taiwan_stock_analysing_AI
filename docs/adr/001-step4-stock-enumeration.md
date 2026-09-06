# ADR-001: Step 4 — Stock Enumeration & Related-Stock Lookup

## Status

Proposed

## Context

Users query the pipeline with range-based questions (e.g., 「半導體產業有哪些股票」、「台灣50成分股」、「2024年1月所有上市股票」) or single-stock questions that imply supply-chain context (e.g., 「台積電的上游供應商有哪些」). Currently Steps 1–3 handle query understanding, classification, and decomposition, but never enumerate actual stock codes or related companies. The user needs a structured list of stocks to feed downstream analysis.

## Decision

Add a **Step 4** after Step 3 (Decomposition) in the pipeline.

### Trigger Condition

Step 4 executes when **all** of the following are true:

1. `boundary.stock_codes` is non-empty **OR** boundary extracted a range/topic (sector, index, date range)
2. `classification.type != NON_FINANCIAL`

If condition is not met, Step 4 is skipped (status = `skipped`).

### Two Execution Modes

| Mode | When | Input | Output |
|------|------|-------|--------|
| **Range** | Boundary has range/topic but `stock_codes` is empty or contains only placeholder codes | sector name, index name, or date range | list of `{code, name}` for all matching stocks |
| **Related** | Boundary has exactly 1 stock code (single-stock query) | stock code | upstream suppliers `[{code, name}]` + downstream customers `[{code, name}]` |

If boundary has multiple stock codes (e.g., from a prior range query or explicit multi-stock input), Step 4 runs in **Range** mode for each code group.

### Pydantic Models

Two independent models in `models.py`:

```python
class StockItem(BaseModel):
    code: str          # e.g., "2330"
    name: str          # e.g., "台積電"

class RangeStockResult(BaseModel):
    query_type: str            # "sector" | "index" | "market" | "date_range"
    query_value: str           # e.g., "半導體", "台灣50", "2024年1月"
    stocks: list[StockItem]    # all matching stocks
    time_range: DateRange | None = None  # if date-range based
    source: str                # "twse_api" | "llm" | "combined"
    confidence: float          # 0.0–1.0

class RelatedStockResult(BaseModel):
    stock_code: str            # target stock
    stock_name: str            # target name
    upstream: list[StockItem]  # direct suppliers (1 layer)
    downstream: list[StockItem]  # direct customers (1 layer)
    source: str                # "llm" | "api" | "combined"
    confidence: float          # 0.0–1.0
```

### Pipeline Integration

- Output stored in `PipelineResult.steps[]` as a new step (name: `"stock_enumeration"`)
- `Pipeline.run()` adds Step 4 after Step 3, guarded by the trigger condition
- Step 4 execution is **synchronous** (same as Steps 1–3)

### Caching

- TWSE API data (sector→stocks, index→constituents) cached locally with 7-day TTL
- Cache stored in `classifier/stock_cache.py` or `cache/` directory
- LLM results are **never** cached (query-dependent)

### Frontend

- Step 4 results displayed in the existing step panel, same as Steps 1–3
- Range mode: shows stock list with count
- Related mode: shows upstream/downstream as two sub-lists

## Consequences

### Positive

- Users get structured stock lists without leaving the pipeline
- Two-model design keeps Range and Related concerns cleanly separated
- Caching avoids repeated TWSE API calls
- Confidence scores let downstream steps gauge data quality

### Negative

- New TWSE API dependency (free, no auth, but subject to rate limits and downtime)
- Additional LLM call for validation/supplement (cost + latency)
- Pipeline gains ~1 step of latency
- `PipelineResult.steps[]` grows by 1 entry

### Risks

- TWSE API may change endpoints without notice → cache TTL helps, but needs monitoring
- Supply chain data (upstream/downstream) may be incomplete from free sources → confidence score reflects this
- LLM hallucination on stock lists → API-first + LLM-validation mitigates this
