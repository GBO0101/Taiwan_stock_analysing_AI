# classify-twse-query

Taiwan-stock natural-language query understanding → classification → decomposition → supply-chain analysis → visualization tool.

用自然語言查詢台股：解析問題（標的/範圍/時間/指標）→ 分類（即時/事實/分析）→ 拆解分析任務 → 列出成分股或上下游供應鏈 → 產出圖表。

## 開始使用 (Getting Started)

完整使用說明（安裝、LLM/圖表設定、CLI、一鍵啟動、API、前端、測試）請見 [GETTING_STARTED.md](GETTING_STARTED.md)。

快速開始：

```bash
pip install -r requirements.txt
cp .env.example .env        # 編輯 LLM_BASE_URL / LLM_MODEL（本機可用 Ollama，免 key）
python start.py             # 一鍵啟動 API + 前端，並開啟瀏覽器
```

## Project Structure

```
classify-twse-query/
├── classifier/             # importable package: the whole pipeline lives here
│   ├── pipeline.py         # Pipeline.run(): STRICT sequential orchestrator (Steps 1-4)
│   ├── boundary.py         # Step 1: entity/scope/time/metric extraction; chart fields reconciled here (chart source of truth)
│   ├── chart_validator.py  # keyword → ChartDataRequirement + enforced chart_type
│   ├── stock_resolver.py   # authoritative company_name ↔ stock_code resolver (singleton)
│   ├── classification.py   # Step 2: query type + visualization needs
│   ├── decomposition.py    # Step 3 (conditional): analytical → sub-query DAG
│   ├── stock_enumeration.py# Step 4 (conditional): Range (sector/index) & Related (upstream/downstream)
│   ├── annual_report.py    # supply-chain disclosure extraction from 股東會年報 (PDF → LLM structured)
│   ├── data_fetcher.py     # FREE TWSE fetcher (STOCK_DAY, BWIBBU); no key
│   ├── isin_client.py      # TWSE ISIN 出版列表 (Range mode 上市/上櫃/興櫃)
│   ├── doc_twse_client.py  # doc.twse.com.tw 年報 PDF 抓取 (F18 優先 / F04 備援)
│   ├── finmind_client.py   # paid FinMind client (charts only)
│   ├── stock_cache.py      # TTL 快取 (ISIN 7d / TIP 7d / 年報揭露 30d)
│   ├── indicator_mapper.py # loads table.csv → indicator→dataset/field map
│   ├── chart_renderer.py   # matplotlib chart writer → output/charts/*.png; K-line 疊日均線
│   ├── llm_client.py       # vendor-neutral OpenAI-compatible structured-extraction client
│   ├── prompts.py          # Jinja2 PromptManager over prompts/*.j2
│   ├── models.py           # Pydantic contracts + enums (single source of truth)
│   ├── config.py           # pydantic-settings; imports .env
│   ├── cli.py              # `python -m classifier.cli` entry point
│   ├── api.py              # FastAPI app (endpoints below)
│   └── logging_setup.py    # structured logging bootstrap
├── prompts/                # boundary.j2 / classify.j2 / decompose.j2 / annual_report.j2 / stock_validation.j2 / supply_chain.j2 / sentiment.j2
├── frontend/               # plain index.html / app.js / styles.css (static, calls API :8000)
├── docs/                   # ADRs (Step 4 stock enumeration, multi-source data strategy) + glossary
├── tests/
│   ├── unit/               # 每個 classifier 模組一個 test_*.py（LLM/網路 MOCKED）
│   └── integration/        # FastAPI TestClient + pipeline-e2e sequential-contract tests
├── output/
│   └── charts/             # 產出的 PNG（gitignored）
├── table.csv               # indicator→dataset/field mapping (loaded at import)
├── pyproject.toml          # pytest / ruff / mypy config
├── requirements.txt
├── .env.example
├── AGENTS.md
├── GETTING_STARTED.md
├── start.py                # one-click: API :8000 + frontend :8080 + opens browser
└── README.md
```

## Pipeline

1. **Step 1 — Boundary Extraction**: 抽出標的、範圍、時間、指標、市場、視覺化提示
2. **Step 2 — Classification**: 判斷查詢類型（live/factual/analytical/non_financial）與視覺化需求
3. **Step 3 — Decomposition** (conditional): 僅 `ANALYTICAL` 類型將分析問題拆成可執行的 FinMind query DAG
4. **Step 4 — Stock Enumeration** (conditional): 有標的/範圍且非 `NON_FINANCIAL` 時執行
   - **Range mode**: sector/index/market → 完整股票清單（TWSE ISIN + TIP，LLM 驗證）
   - **Related mode**: 單一標的 → upstream/downstream 供應鏈，**年報優先**（doc.twse.com.tw PDF → LLM 結構化抽取 `annual_report.j2`，再以 `StockResolver` 交叉驗證；年報缺漏才 fallback 到 LLM 推論 `supply_chain.j2`）

The **boundary output** is the single source of truth for charting: its `chart_data_requirements` (reconciled by `chart_validator` keywords) decides the chart type, not the classification step.

Pipeline 透過嚴格循序執行，Step 1 失敗即中止 Step 2；Step 3 僅在 Step 2 為 `ANALYTICAL` 時執行；Step 4 依觸發條件執行或標記 `skipped`。Step 4 結果同時彙整為 `QuerySummary`（topic + impact-sorted related stocks）輸出在 `PipelineResult.summary`。

## Configuration

The LLM layer is **vendor-neutral**: it speaks the OpenAI-compatible Chat
Completions API, so any provider exposing `/v1/chat/completions` works — OpenAI,
Ollama, vLLM, Groq, DeepSeek, OpenRouter, LM Studio, and more. Just point
`LLM_BASE_URL` at the provider's `/v1` endpoint.

Required environment variables:
- `LLM_BASE_URL`: OpenAI-compatible endpoint base URL (default: `https://api.openai.com/v1`)
- `LLM_API_KEY` (or legacy `OPENAI_API_KEY`): API key for the provider
- `LLM_MODEL` (or legacy `OPENAI_MODEL`): Model to use (default: `gpt-4o-mini`)
- `LLM_TIMEOUT` (or legacy `OPENAI_TIMEOUT`): Request timeout in seconds (default: 30)
- `FINMIND_API_TOKEN`: **config 層必填**（`Settings` 無預設值，`.env` 沒設會直接 `ValidationError`）。免費路徑（TWSE）不會用到它，僅「月營收圖」與 ChartRenderer 的 FinMind 路由需要，但仍須在 `.env` 定義才能 import。

### Chart data source

Charts are rendered from **free TWSE open APIs** (no API key required):

- Price / K-line / volume → TWSE `STOCK_DAY`
- PE ratio / sector analysis → TWSE `BWIBBU`

The K-line (candlestick) chart overlays a **blue dashed daily-average trend line** (mean of each day's open/high/low/close).

Monthly **revenue** charts are NOT available from the free source (MOPS blocks
unauthenticated access). They require a paid FinMind account:

- `FINMIND_BASE_URL`: FinMind API base URL (default: https://api.finmindtrade.com/api/v4)

### Examples

```bash
# OpenAI (default)
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini

# Local Ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama   # Ollama ignores the key, any non-empty value works
LLM_MODEL=qwen2.5:7b

# Groq
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...
LLM_MODEL=llama-3.1-70b-versatile
```

## License

MIT