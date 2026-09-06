# Glossary — Step 4 Stock Enumeration

| Term | Definition |
|------|-----------|
| **Stock Enumeration** | The process of listing all stocks that match a given criteria (sector, index, date range, or supply-chain relationship). This is Step 4's core function. |
| **Range Query** | A query that asks for all stocks within a category: a sector (產業), an index (指數), or a date range (日期範圍). Produces `RangeStockResult`. |
| **Related Stock Query** | A query for a single stock that also needs its upstream suppliers and downstream customers. Produces `RelatedStockResult`. |
| **StockItem** | `{ code: str, name: str }` — a stock identified by its 4-digit TWSE code and Chinese name. The atomic unit of stock enumeration output. |
| **Upstream (上游)** | Companies that supply raw materials, components, or services to the target stock. Direct suppliers only (1 layer). |
| **Downstream (下游)** | Companies that purchase or use the target stock's products/services. Direct customers only (1 layer). |
| **Source** | Where the stock list came from: `"twse_api"` (authoritative), `"llm"` (inferred), or `"combined"` (API + LLM validated). |
| **Confidence** | A 0.0–1.0 score indicating data reliability. API-sourced data scores higher (0.8–0.9); LLM-only data scores lower (0.3–0.6). |
| **TWSE** | Taiwan Stock Exchange (台灣證券交易所). Provides free open APIs for industry classification and index constituents. |
| **MOPS** | Market Observation Post System (公開資訊觀測站). Taiwan's financial disclosure platform. May have supply-chain data but requires authentication. |
| **Query Type** | For range queries: `"sector"` (產業), `"index"` (指數), `"market"` (全市場), or `"date_range"` (日期範圍). Determines which TWSE API endpoint to call. |
| **RangeStockResult** | Pydantic model for range-query output: query type, value, stock list, optional time range, source, confidence. |
| **RelatedStockResult** | Pydantic model for related-stock output: target stock, upstream list, downstream list, source, confidence. |
| **Cache TTL** | Time-to-live for locally cached TWSE API data. Default: 7 days. TWSE sector/index data changes infrequently. |
| **Step 4** | The pipeline step for stock enumeration. Executes after Step 3 (Decomposition) when boundary has stock codes or range/topic and type is not NON_FINANCIAL. |
