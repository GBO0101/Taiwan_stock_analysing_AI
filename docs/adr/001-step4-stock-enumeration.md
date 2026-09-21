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
| **Related** | Boundary has exactly 1 stock code (single-stock query) | stock code | upstream suppliers `[{code, name}]` + downstream customers `[{code, name}]`, each with `evidence` (disclosed_by, target_ratio, source_pdf) |

If boundary has multiple stock codes (e.g., from a prior range query or explicit multi-stock input), Step 4 runs in **Range** mode for each code group.

#### Related mode: Annual-report-first + LLM fallback

```
1. 提取目標股票的股東會年報（doc.twse.com.tw，F18 含合併財報優先，F04 次要）。
2. PyMuPDF 指定頁面抽取文字 → LLM 結構化抽取前五大客戶/供應商。
3. 用 StockResolver 交叉驗證每個 counterparty 是否為上市櫃公司。
   - counterparty 可解析 → 併入 upstream/downstream；以 ratio 作為 impact。
   - counterparty 不可解析 → 跳過。
4. Phase 1.5：對 up to 3 家 ratio 最高的 counterparty，抓取其自身年報，
   - 若他們也列出目標股票 → disclosed_by = "both"（雙邊揭露）。
5. 若年報未公告或無可解析 counterparty → fallback 到 LLM 推論（supply_chain.j2）。
```

### Pydantic Models

Models in `models.py` (Step 4 only; `models.py` is the single source of truth):

```python
class StockItem(BaseModel):
    code: str          # e.g., "2330"
    name: str          # e.g., "台積電"

class RangeStockResult(BaseModel):
    query_type: str            # "sector" | "index" | "market"
    query_value: str           # e.g., "半導體", "台灣50"
    stocks: list[StockItem]    # all matching stocks
    source: str                # "combined" | "twse_api" | "llm"
    confidence: float          # 0.0–1.0

class RelatedStockResult(BaseModel):
    stock_code: str            # target stock
    stock_name: str            # target name
    upstream: list[StockItem]  # direct suppliers (1 layer)
    downstream: list[StockItem]  # direct customers (1 layer)
    source: str                # "annual_report" | "llm"
    confidence: float          # 0.0–1.0
    evidence: list[SupplyChainEvidence]  # disclosed_by, target_ratio, source_pdf

class SupplyChainEvidence(BaseModel):
    counterparty_code: str
    counterparty_name: str
    relation: SupplyChainRelation      # UPSTREAM | DOWNSTREAM
    target_ratio: float | None         # Revenue/purchase percentage (annual report only)
    disclosed_by: str                  # "target" | "both" | "llm_inferred"
    source_pdf: str | None             # PDF filename from doc.twse.com.tw

class ImpactStock(BaseModel):
    code: str
    name: str
    relation: StockRelation            # TARGET | UPSTREAM | DOWNSTREAM | RANGE_MEMBER
    impact_score: float                # 0–1; higher = more impacted
    ratio: float | None                # annual-report ratio (%), for related mode
    fiscal_year: int | None            # ROC fiscal year of this data point
    detail_source: str                 # "annual_report" | "llm" | "twse_api"

class QuerySummary(BaseModel):
    topic_kind: QueryTopicKind        # SINGLE_STOCK | RANGE
    topic_code: str | None            # target stock code (related mode)
    topic_name: str | None            # display name
    range_type: RangeQueryType | None # sector/index/market (range mode)
    range_value: str | None           # "半導體", "台灣50" (range mode)
    fiscal_years: list[int]           # ROC fiscal years, newest first
    notes: list[str]                  # coverage notes
    related_stocks: list[ImpactStock] # sorted by impact_score descending
```

### Pipeline Integration

- Output stored in `PipelineResult.steps[]` as a new step (name: `"stock_enumeration"`)
- `Pipeline.run()` adds Step 4 after Step 3, guarded by the trigger condition
- Step 4 execution is **synchronous** (same as Steps 1–3)
- `Pipeline.run()` builds `PipelineResult.summary` (a `QuerySummary`) from the Step 4 result, so downstream agents receive a unified topic + related-stocks structure alongside the step trace

### Caching

- ISIN sector→stock list: 7-day TTL (`classifier/stock_cache.py`)
- TIP index constituents: 7-day TTL
- Annual report disclosure (`AnnualReportDisclosure`): 30-day TTL in `classifier/stock_cache.py`; key = `annual_report_{code}_{fiscal_year}` (raw extracted JSON is cached; per-request PDF fetch still hits doc.twse.com.tw)
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
